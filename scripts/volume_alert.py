"""
5-minute volume breakout alert scanner.

Universe : CoinGecko market-cap rank RANK_MIN..RANK_MAX, matched to Binance USDT spot pairs.
Trigger  : latest closed 5m candle's quote volume > VOLUME_MULTIPLIER x the average volume of
           the SAME 5-minute-of-day slot over the past HISTORY_DAYS days, AND the candle's close
           price breaks above the high of the previous RANGE_LOOKBACK closed candles.
State    : data/volume_history.json persists per-symbol, per-time-slot rolling volume history so
           each run only needs to fetch each symbol's most recent candles (cheap + fast), instead
           of re-downloading days of history every run.
Notify   : Telegram bot (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID from environment / GitHub Secrets).
"""

import os
import json
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ---------------- Config ----------------
RANK_MIN = 200
RANK_MAX = 500
VOLUME_MULTIPLIER = 8
RANGE_LOOKBACK = 20          # number of prior closed 5m candles used for breakout range-high
HISTORY_DAYS = 7             # days of same-time-slot history kept for the baseline average
MIN_HISTORY_SAMPLES = 3      # need at least this many days of data before alerting on a slot
MAX_WORKERS = 10

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE_DIR, "..", "data", "volume_history.json")

BINANCE_BASE = "https://api.binance.com"
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


def get_binance_usdt_symbols():
    r = SESSION.get(f"{BINANCE_BASE}/api/v3/exchangeInfo", timeout=20)
    r.raise_for_status()
    data = r.json()
    return {
        s["symbol"]
        for s in data["symbols"]
        if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"
    }


def fetch_klines(symbol):
    """Latest RANGE_LOOKBACK+2 candles (last one still forming). None on failure."""
    limit = RANGE_LOOKBACK + 2
    for attempt in range(2):
        try:
            r = SESSION.get(
                f"{BINANCE_BASE}/api/v3/klines",
                params={"symbol": symbol, "interval": "5m", "limit": limit},
                timeout=10,
            )
            if r.status_code == 429:
                time.sleep(2)
                continue
            if r.status_code != 200:
                return None
            data = r.json()
            if len(data) < limit:
                return None
            return data
        except requests.RequestException:
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
    klines = fetch_klines(symbol)
    if not klines:
        return None, None

    latest_closed = klines[-2]
    prev_range = klines[-(RANGE_LOOKBACK + 2):-2]
    open_time = int(latest_closed[0])
    close_price = float(latest_closed[4])
    quote_volume = float(latest_closed[7])
    range_high = max(float(c[2]) for c in prev_range)

    sym_state = state.get(symbol, {"slots": {}, "last_open_time": 0})
    slot = str(slot_index(open_time))

    already_seen = open_time <= sym_state.get("last_open_time", 0)

    history = list(sym_state.get("slots", {}).get(slot, []))
    alert = None
    if not already_seen and len(history) >= MIN_HISTORY_SAMPLES:
        avg_vol = statistics.mean(history)
        if avg_vol > 0 and quote_volume > VOLUME_MULTIPLIER * avg_vol and close_price > range_high:
            alert = {
                "symbol": symbol,
                "name": meta["name"],
                "rank": meta["rank"],
                "volume": quote_volume,
                "avg_volume": avg_vol,
                "ratio": quote_volume / avg_vol,
                "close": close_price,
                "range_high": range_high,
            }

    if not already_seen:
        history.append(quote_volume)
        history = history[-HISTORY_DAYS:]
        sym_state.setdefault("slots", {})[slot] = history
        sym_state["last_open_time"] = open_time

    return alert, sym_state


def main():
    universe = get_market_cap_universe()
    if not universe:
        print("Could not fetch market-cap universe from CoinGecko this run; skipping.")
        return

    valid_symbols = get_binance_usdt_symbols()

    targets = []
    for coin in universe:
        sym = coin["symbol"] + "USDT"
        if sym in valid_symbols:
            targets.append((sym, coin))

    print(f"Scanning {len(targets)} symbols (CoinGecko rank {RANK_MIN}-{RANK_MAX}, matched on Binance USDT)")

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
            lines.append(
                f"\n#{a['rank']} <b>{a['symbol']}</b> ({a['name']})\n"
                f"量比: {a['ratio']:.1f}x 均值 (現量 {a['volume']:,.0f} / 均量 {a['avg_volume']:,.0f} USDT)\n"
                f"現價 {a['close']:.6g} 突破近{RANGE_LOOKBACK}根高點 {a['range_high']:.6g}"
            )
        send_telegram("\n".join(lines))
        print(f"Sent alert for {len(alerts)} symbol(s).")
    else:
        print("No alerts this run.")


if __name__ == "__main__":
    main()
