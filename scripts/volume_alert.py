"""
5-minute volume breakout alert scanner.

Universe : CoinGecko market-cap rank RANK_MIN..RANK_MAX, matched to OKX USDT spot pairs.
           (Binance's API returns HTTP 451 - blocked - for GitHub Actions runner IPs, so OKX
           is used as the exchange data source instead.)
Trigger  : the latest CONFIRM_CANDLES consecutive closed 5m candles must ALL satisfy: quote-currency
           volume > VOLUME_MULTIPLIER x the average volume of that SAME 5-minute-of-day slot over the
           past HISTORY_DAYS days, AND volume > MIN_TRIGGER_VOLUME_USDT (absolute floor so a spike on
           an illiquid coin's near-zero baseline can't trigger a dollar-meaningless alert), AND the
           candle's close price breaks above the high of the RANGE_LOOKBACK closed candles immediately
           preceding it. CONFIRM_CANDLES is currently 1 (single-candle trigger); raise it later to
           require multiple consecutive confirmations if single-candle noise becomes an issue.
State    : data/volume_history.json persists per-symbol, per-time-slot rolling volume history so
           each run only needs to fetch each symbol's most recent candles (cheap + fast), instead
           of re-downloading days of history every run.
Notify   : Telegram bot (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID from environment / GitHub Secrets).
"""

import os
import json
import time
import statistics
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ---------------- Config ----------------
RANK_MIN = 200
RANK_MAX = 500
VOLUME_MULTIPLIER = 15
RANGE_LOOKBACK = 20          # number of prior closed 5m candles used for breakout range-high
HISTORY_DAYS = 7             # days of same-time-slot history kept for the baseline average
MIN_HISTORY_SAMPLES = 5      # need at least this many days of data before alerting on a slot
MIN_TRIGGER_VOLUME_USDT = 3000  # absolute floor on the triggering candle's volume (filters illiquid noise)
CONFIRM_CANDLES = 1          # number of consecutive closed candles that must ALL confirm the trigger
MAX_WORKERS = 10
MAX_ALERTS_KEPT = 300        # cap on how many past alerts are kept in the history file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE_DIR, "..", "data", "volume_history.json")
ALERTS_FILE = os.path.join(BASE_DIR, "..", "data", "alerts_history.json")

OKX_BASE = "https://www.okx.com"
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

SESSION = requests.Session()


def get_market_cap_universe():
    """CoinGecko market-cap rank RANK_MIN..RANK_MAX -> [{symbol, name, rank}, ...]."""
    coins = []
    for page in (1, 2):
        for attempt in range(3):
            try:
                r = SESSION.get(
                    f"{COINGECKO_BASE}/coins/markets",
                    params={
                        "vs_currency": "usd",
                        "order": "market_cap_desc",
                        "per_page": 250,
                        "page": page,
                        "sparkline": "false",
                    },
                    timeout=20,
                )
                r.raise_for_status()
                coins.extend(r.json())
                break
            except requests.RequestException as e:
                print(f"CoinGecko page {page} attempt {attempt+1} failed: {e}")
                time.sleep(3)
        time.sleep(1.5)  # be polite to the free CoinGecko tier

    ranked = []
    for idx, c in enumerate(coins, start=1):
        if RANK_MIN <= idx <= RANK_MAX:
            ranked.append({"symbol": c["symbol"].upper(), "name": c["name"], "rank": idx})
    return ranked


def get_okx_usdt_instruments():
    """Return set of live OKX spot instIds quoted in USDT, e.g. {'BTC-USDT', ...}."""
    r = SESSION.get(
        f"{OKX_BASE}/api/v5/public/instruments",
        params={"instType": "SPOT"},
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()
    return {
        item["instId"]
        for item in data.get("data", [])
        if item.get("quoteCcy") == "USDT" and item.get("state") == "live"
    }


def fetch_candles(inst_id):
    """
    RANGE_LOOKBACK + CONFIRM_CANDLES closed candles for inst_id, oldest-first.
    Returns a list of that many candles [open_time_ms, open, high, low, close, vol, volCcy, volCcyQuote, confirm]
    or None on failure. OKX returns candles newest-first and the first one or two entries may still
    be unconfirmed (still-forming), so we skip forward to the first confirmed candle.
    """
    needed = RANGE_LOOKBACK + CONFIRM_CANDLES
    limit = needed + 3
    for attempt in range(2):
        try:
            r = SESSION.get(
                f"{OKX_BASE}/api/v5/market/candles",
                params={"instId": inst_id, "bar": "5m", "limit": limit},
                timeout=10,
            )
            if r.status_code == 429:
                time.sleep(2)
                continue
            if r.status_code != 200:
                return None
            payload = r.json()
            if payload.get("code") != "0":
                return None
            data = payload.get("data", [])
            if len(data) < limit:
                return None
            # data[0] = newest; skip any still-forming (confirm == "0") candles
            start = 0
            while start < len(data) and data[start][8] == "0":
                start += 1
            closed = data[start:start + needed]
            if len(closed) < needed:
                return None
            closed.reverse()  # oldest-first: closed[-1] is the latest closed candle
            return closed
        except (requests.RequestException, IndexError, ValueError):
            time.sleep(1)
    return None


def slot_index(open_time_ms):
    minute_of_day = (open_time_ms // 60000) % 1440
    return int(minute_of_day // 5)


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def load_alerts_history():
    if os.path.exists(ALERTS_FILE):
        try:
            with open(ALERTS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
    return []


def save_alerts_history(history):
    os.makedirs(os.path.dirname(ALERTS_FILE), exist_ok=True)
    # newest first, capped
    trimmed = history[-MAX_ALERTS_KEPT:] if len(history) > MAX_ALERTS_KEPT else history
    with open(ALERTS_FILE, "w") as f:
        json.dump(trimmed, f)


def send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials not set; would have sent:\n" + text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        resp = SESSION.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        if resp.status_code != 200:
            print(f"Telegram send failed: {resp.status_code} {resp.text}")
    except requests.RequestException as e:
        print(f"Telegram send failed: {e}")


def process_symbol(symbol, meta, state):
    candles = fetch_candles(symbol)
    if not candles:
        return None, None

    # oldest-first, length RANGE_LOOKBACK + CONFIRM_CANDLES.
    # The last CONFIRM_CANDLES entries are the "confirmation window": each one needs its own
    # RANGE_LOOKBACK-candle range-high and its own same-slot historical average, and ALL of them
    # must satisfy the breakout condition for an alert to fire (filters single-candle noise).
    confirm_set = candles[-CONFIRM_CANDLES:]
    newest = confirm_set[-1]
    open_time = int(newest[0])

    sym_state = state.get(symbol, {"slots": {}, "last_open_time": 0})
    sym_state["name"] = meta["name"]
    sym_state["rank"] = meta["rank"]

    already_seen = open_time <= sym_state.get("last_open_time", 0)

    alert = None
    if not already_seen:
        checks = []
        all_confirmed = True
        for i, c in enumerate(confirm_set):
            idx = len(candles) - CONFIRM_CANDLES + i
            prev_range = candles[idx - RANGE_LOOKBACK:idx]
            if len(prev_range) < RANGE_LOOKBACK:
                all_confirmed = False
                break
            c_open_time = int(c[0])
            c_close = float(c[4])
            c_vol = float(c[6])  # volCcy: volume denominated in USDT
            c_range_high = max(float(x[2]) for x in prev_range)
            slot = str(slot_index(c_open_time))
            hist = sym_state.get("slots", {}).get(slot, [])
            if len(hist) < MIN_HISTORY_SAMPLES:
                all_confirmed = False
                break
            avg_vol = statistics.mean(hist)
            ok = (
                avg_vol > 0
                and c_vol > VOLUME_MULTIPLIER * avg_vol
                and c_vol > MIN_TRIGGER_VOLUME_USDT
                and c_close > c_range_high
            )
            if not ok:
                all_confirmed = False
                break
            checks.append({
                "avg_vol": avg_vol,
                "vol": c_vol,
                "close": c_close,
                "range_high": c_range_high,
                "ratio": c_vol / avg_vol,
            })

        if all_confirmed and len(checks) == CONFIRM_CANDLES:
            last = checks[-1]
            alert = {
                "symbol": symbol,
                "name": meta["name"],
                "rank": meta["rank"],
                "volume": last["vol"],
                "avg_volume": last["avg_vol"],
                "ratio": last["ratio"],
                "close": last["close"],
                "range_high": last["range_high"],
                "candle_time": datetime.fromtimestamp(open_time / 1000, tz=timezone.utc).isoformat(),
                "confirm_candles": CONFIRM_CANDLES,
            }

    if not already_seen:
        # Only the newest candle is new data; older candles in the confirm window were already
        # appended to their own slot's history in a previous run.
        newest_open_time = int(newest[0])
        newest_vol = float(newest[6])
        newest_slot = str(slot_index(newest_open_time))
        history = list(sym_state.get("slots", {}).get(newest_slot, []))
        history.append(newest_vol)
        history = history[-HISTORY_DAYS:]
        sym_state.setdefault("slots", {})[newest_slot] = history
        sym_state["last_open_time"] = newest_open_time

    return alert, sym_state


def main():
    universe = get_market_cap_universe()
    if not universe:
        print("Could not fetch market-cap universe from CoinGecko this run; skipping.")
        return

    valid_symbols = get_okx_usdt_instruments()

    targets = []
    for coin in universe:
        sym = coin["symbol"] + "-USDT"
        if sym in valid_symbols:
            targets.append((sym, coin))

    print(f"Scanning {len(targets)} symbols (CoinGecko rank {RANK_MIN}-{RANK_MAX}, matched on OKX USDT)")

    state = load_state()
    alerts = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_symbol, sym, meta, state): sym for sym, meta in targets}
        for fut in as_completed(futures):
            sym = futures[fut]
            alert, sym_state = fut.result()
            if sym_state is not None:
                state[sym] = sym_state
            if alert:
                alerts.append(alert)

    save_state(state)

    if alerts:
        lines = ["\U0001F6A8 <b>5分鐘成交量爆量突破警報</b>"]
        for a in sorted(alerts, key=lambda x: -x["ratio"]):
            confirm_note = f"\n（連續 {a['confirm_candles']} 根5分K確認）" if a["confirm_candles"] > 1 else ""
            lines.append(
                f"\n#{a['rank']} <b>{a['symbol']}</b> ({a['name']})\n"
                f"量比: {a['ratio']:.1f}x 均值 (現量 {a['volume']:,.0f} / 均量 {a['avg_volume']:,.0f} USDT)\n"
                f"現價 {a['close']:.6g} 突破近{RANGE_LOOKBACK}根高點 {a['range_high']:.6g}"
                f"{confirm_note}"
            )
        send_telegram("\n".join(lines))
        print(f"Sent alert for {len(alerts)} symbol(s).")

        detected_at = datetime.now(timezone.utc).isoformat()
        alerts_history = load_alerts_history()
        for a in sorted(alerts, key=lambda x: -x["ratio"]):
            alerts_history.append({
                "detected_at": detected_at,
                "candle_time": a["candle_time"],
                "symbol": a["symbol"],
                "name": a["name"],
                "rank": a["rank"],
                "volume": a["volume"],
                "avg_volume": a["avg_volume"],
                "ratio": a["ratio"],
                "close": a["close"],
                "range_high": a["range_high"],
            })
        save_alerts_history(alerts_history)
    else:
        print("No alerts this run.")


if __name__ == "__main__":
    main()
