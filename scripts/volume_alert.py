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
Divergence: independent RSI(14) divergence detector on the same 5m candles. A bullish divergence
           (底背離) fires when price prints a lower swing-low but RSI prints a higher low; a bearish
           divergence (頂背離) fires when price prints a higher swing-high but RSI prints a lower
           high. Swing points are pivot highs/lows confirmed by PIVOT_WINDOW candles on each side,
           and the first pivot's RSI must be in the oversold/overbought zone to filter weak signals.
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

# --- RSI divergence (頂背離/底背離) detection ---
DIVERGENCE_ENABLED = True
RSI_PERIOD = 14
PIVOT_WINDOW = 2             # candles required on EACH side of a swing high/low to confirm the pivot
DIVERGENCE_LOOKBACK = 60     # recent closed candles scanned for divergence pivots (~5 hours of 5m)
DIVERGENCE_MIN_GAP = 5       # min candles between the two compared pivots
DIVERGENCE_MAX_GAP = 40      # max candles between the two compared pivots
BULL_DIV_RSI_MAX = 30        # first pivot's RSI must be below this for a bullish (bottom) divergence
BEAR_DIV_RSI_MIN = 80        # first pivot's RSI must be above this for a bearish (top) divergence

# Candle count fetched per symbol: enough for the breakout check AND RSI warmup + divergence scan.
CANDLES_NEEDED = max(
    RANGE_LOOKBACK + CONFIRM_CANDLES,
    DIVERGENCE_LOOKBACK + RSI_PERIOD * 3,  # ~3x period of Wilder-smoothing warmup before the scan window
)

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
    Up to CANDLES_NEEDED closed candles for inst_id, oldest-first.
    Returns a list of candles [open_time_ms, open, high, low, close, vol, volCcy, volCcyQuote, confirm]
    or None on failure. OKX returns candles newest-first and the first one or two entries may still
    be unconfirmed (still-forming), so we skip forward to the first confirmed candle. Symbols with a
    short trading history may return fewer than CANDLES_NEEDED candles; the breakout check still
    needs its minimum, but divergence detection simply skips symbols without enough data.
    """
    limit = min(CANDLES_NEEDED + 3, 300)  # OKX per-request cap is 300
    min_needed = RANGE_LOOKBACK + CONFIRM_CANDLES
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
            # data[0] = newest; skip any still-forming (confirm == "0") candles
            start = 0
            while start < len(data) and data[start][8] == "0":
                start += 1
            closed = data[start:start + CANDLES_NEEDED]
            if len(closed) < min_needed:
                return None
            closed.reverse()  # oldest-first: closed[-1] is the latest closed candle
            return closed
        except (requests.RequestException, IndexError, ValueError):
            time.sleep(1)
    return None


def compute_rsi(closes, period=RSI_PERIOD):
    """Wilder-smoothed RSI. Returns a list aligned with closes (None before warmup), or None if too short."""
    if len(closes) < period + 1:
        return None
    rsi = [None] * len(closes)
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        gains += max(d, 0.0)
        losses += max(-d, 0.0)
    avg_gain = gains / period
    avg_loss = losses / period
    rsi[period] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(d, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-d, 0.0)) / period
        rsi[i] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return rsi


def find_pivots(values, window, is_low):
    """Indices of swing lows (is_low) or swing highs: extreme within +/- window candles."""
    pivots = []
    for i in range(window, len(values) - window):
        segment = values[i - window:i + window + 1]
        if (is_low and values[i] == min(segment)) or (not is_low and values[i] == max(segment)):
            pivots.append(i)
    return pivots


def detect_divergence(candles):
    """
    RSI divergence on oldest-first candles. A signal only fires when the newest closed candle is the
    one that just confirmed a new pivot (pivot index == len - 1 - PIVOT_WINDOW), so each divergence
    is reported exactly once, PIVOT_WINDOW candles after the actual swing point.

    bullish (底背離): price makes a lower low but RSI makes a higher low (first pivot RSI oversold-ish)
    bearish (頂背離): price makes a higher high but RSI makes a lower high (first pivot RSI overbought-ish)
    Returns a list of signal dicts (0-2 entries).
    """
    closes = [float(c[4]) for c in candles]
    rsi = compute_rsi(closes)
    if rsi is None:
        return []
    n = len(candles)
    confirm_idx = n - 1 - PIVOT_WINDOW
    scan_start = max(RSI_PERIOD + 1, n - DIVERGENCE_LOOKBACK)
    if confirm_idx <= scan_start:
        return []

    signals = []
    for is_low, div_type, extreme_ok in (
        (True, "bullish_divergence", lambda r: r < BULL_DIV_RSI_MAX),
        (False, "bearish_divergence", lambda r: r > BEAR_DIV_RSI_MIN),
    ):
        series = [float(c[3]) for c in candles] if is_low else [float(c[2]) for c in candles]
        pivots = [i for i in find_pivots(series, PIVOT_WINDOW, is_low) if i >= scan_start]
        if len(pivots) < 2 or pivots[-1] != confirm_idx:
            continue
        p2 = pivots[-1]
        p1 = pivots[-2]
        gap = p2 - p1
        if gap < DIVERGENCE_MIN_GAP or gap > DIVERGENCE_MAX_GAP:
            continue
        if not extreme_ok(rsi[p1]):
            continue
        if is_low:
            diverged = series[p2] < series[p1] and rsi[p2] > rsi[p1]
        else:
            diverged = series[p2] > series[p1] and rsi[p2] < rsi[p1]
        if diverged:
            signals.append({
                "type": div_type,
                "price1": series[p1],
                "price2": series[p2],
                "rsi1": rsi[p1],
                "rsi2": rsi[p2],
                "pivot_time": int(candles[p2][0]),
                "close": closes[-1],
                "gap_candles": gap,
            })
    return signals


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
        return None, [], None

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

    div_alerts = []
    if DIVERGENCE_ENABLED and not already_seen:
        last_div = sym_state.get("last_divergence", {})
        for sig in detect_divergence(candles):
            # dedupe on the confirming pivot's candle time, per divergence type
            if sig["pivot_time"] <= last_div.get(sig["type"], 0):
                continue
            last_div[sig["type"]] = sig["pivot_time"]
            div_alerts.append({
                "symbol": symbol,
                "name": meta["name"],
                "rank": meta["rank"],
                "candle_time": datetime.fromtimestamp(sig["pivot_time"] / 1000, tz=timezone.utc).isoformat(),
                **sig,
            })
        if last_div:
            sym_state["last_divergence"] = last_div

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

    return alert, div_alerts, sym_state


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
    div_alerts = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_symbol, sym, meta, state): sym for sym, meta in targets}
        for fut in as_completed(futures):
            sym = futures[fut]
            alert, sym_divs, sym_state = fut.result()
            if sym_state is not None:
                state[sym] = sym_state
            if alert:
                alerts.append(alert)
            div_alerts.extend(sym_divs)

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
        print("No volume alerts this run.")

    if div_alerts:
        lines = ["\U0001F4C8\U0001F4C9 <b>5分鐘 RSI 背離警報</b>"]
        for a in sorted(div_alerts, key=lambda x: (x["type"], x["rank"])):
            if a["type"] == "bullish_divergence":
                label = "\U0001F4C8 底背離"
                price_note = f"價格創新低 {a['price1']:.6g} → {a['price2']:.6g}，RSI 走高 {a['rsi1']:.1f} → {a['rsi2']:.1f}"
            else:
                label = "\U0001F4C9 頂背離"
                price_note = f"價格創新高 {a['price1']:.6g} → {a['price2']:.6g}，RSI 走低 {a['rsi1']:.1f} → {a['rsi2']:.1f}"
            lines.append(
                f"\n{label} #{a['rank']} <b>{a['symbol']}</b> ({a['name']})\n"
                f"{price_note}\n"
                f"現價 {a['close']:.6g}（兩個轉折點相隔 {a['gap_candles']} 根5分K）"
            )
        send_telegram("\n".join(lines))
        print(f"Sent divergence alert for {len(div_alerts)} signal(s).")

        detected_at = datetime.now(timezone.utc).isoformat()
        alerts_history = load_alerts_history()
        for a in sorted(div_alerts, key=lambda x: (x["type"], x["rank"])):
            alerts_history.append({
                "detected_at": detected_at,
                "candle_time": a["candle_time"],
                "type": a["type"],
                "symbol": a["symbol"],
                "name": a["name"],
                "rank": a["rank"],
                "close": a["close"],
                "price1": a["price1"],
                "price2": a["price2"],
                "rsi1": a["rsi1"],
                "rsi2": a["rsi2"],
                "gap_candles": a["gap_candles"],
            })
        save_alerts_history(alerts_history)
    else:
        print("No divergence alerts this run.")


if __name__ == "__main__":
    main()
