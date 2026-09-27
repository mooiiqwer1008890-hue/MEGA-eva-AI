"""
quant_alert.py  (combined version)
===================================
Runs BOTH of our tools as ONE process, so a single Railway service
(free-tier friendly) covers everything:

1. The statistical alert (Z-score + Garman-Klass) — sent to Telegram
   every cycle, exactly as before.
2. Paper-trading execution on Binance Testnet — using ONLY the
   buy-side signal our backtest validated (see backtest_zscore.py
   results: buy signal had a real historical edge; sell signal did
   not, so it is deliberately never traded here).

WHY COMBINED INTO ONE FILE:
Railway's free trial only allows one service per project without
paying. Rather than two separate always-on processes, this single
loop does the market fetch ONCE per cycle and then does both jobs
with that same data — alert and (optionally) execution.

GRACEFUL DEGRADATION:
Telegram alerting is the core, must-always-work feature. Testnet
execution is optional and additive: if BINANCE_TESTNET_API_KEY /
BINANCE_TESTNET_API_SECRET are not set, the bot logs that execution
is disabled and continues sending alerts normally — it does NOT
crash the whole process over a missing execution-only variable.
"""

import os
import time
import hmac
import hashlib
import json
import logging
import requests
import numpy as np

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
BOT_TOKEN = os.environ.get("TG_BOT_TOKEN")
CHAT_ID = os.environ.get("TG_CHAT_ID")
SYMBOLS = [s.strip() for s in os.environ.get(
    "SYMBOLS", "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT"
).split(",") if s.strip()]
INTERVAL = os.environ.get("INTERVAL", "15m")
LOOKBACK = int(os.environ.get("LOOKBACK", "20"))
CANDLE_LIMIT = int(os.environ.get("CANDLE_LIMIT", "100"))
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "900"))  # 15 min default

# Testnet execution (optional — only activates if both are set)
TESTNET_API_KEY = os.environ.get("BINANCE_TESTNET_API_KEY")
TESTNET_API_SECRET = os.environ.get("BINANCE_TESTNET_API_SECRET")
TESTNET_BASE_URL = "https://testnet.binance.vision"
EXECUTION_ENABLED = bool(TESTNET_API_KEY and TESTNET_API_SECRET)

Z_THRESHOLD = 2.0
HOLD_CANDLES = 10  # matches the backtest horizon that showed a real edge
TRADE_NOTIONAL_USDT = float(os.environ.get("TRADE_NOTIONAL_USDT", "15"))


def position_file(symbol):
    return f"position_state_{symbol}.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("quant_alert")


def require_config():
    # Only the alerting variables are hard-required. Execution is optional.
    missing = [n for n, v in [("TG_BOT_TOKEN", BOT_TOKEN), ("TG_CHAT_ID", CHAT_ID)] if not v]
    if missing:
        raise RuntimeError(f"Missing environment variable(s): {', '.join(missing)}")


# ---------------------------------------------------------------------
# MARKET DATA (public, no auth — used for both alerts and signal calc)
# ---------------------------------------------------------------------
def fetch_candles(symbol, interval, limit):
    url = "https://data-api.binance.vision/api/v3/klines"
    resp = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=10)
    resp.raise_for_status()
    raw = resp.json()
    return np.array([[float(k[1]), float(k[2]), float(k[3]), float(k[4])] for k in raw])


# ---------------------------------------------------------------------
# INDICATORS
# ---------------------------------------------------------------------
def z_score(closes, period):
    window = closes[-period:]
    mean, std = window.mean(), window.std()
    return 0.0 if std == 0 else (closes[-1] - mean) / std


def garman_klass_volatility(candles, period):
    window = candles[-period:]
    o, h, l, c = window[:, 0], window[:, 1], window[:, 2], window[:, 3]
    log_hl = np.log(h / l)
    log_co = np.log(c / o)
    variance = (0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2).mean()
    return np.sqrt(max(variance, 0)) * 100.0


def classify_signal(z):
    if z < -2.0:
        return "🟢 شراء إحصائي (انحراف سلبي قوي عن المتوسط)"
    elif z > 2.0:
        return "🔴 بيع إحصائي (انحراف إيجابي قوي عن المتوسط)"
    return "⚪ محايد - لا يوجد شذوذ إحصائي واضح"


# ---------------------------------------------------------------------
# TELEGRAM
# ---------------------------------------------------------------------
def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, data={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}, timeout=10)
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        log.error(f"Telegram send failed: {e}")
        return False


def build_coin_section(symbol, price, z, vol):
    signal = classify_signal(z)
    return (
        f"*{symbol}*  السعر: ${price:,.4f}\n"
        f"Z-Score: {z:.2f} | GK Vol: {vol:.3f}%\n"
        f"{signal}"
    )


def build_report(interval, sections):
    body = "\n\n".join(sections)
    return (
        f"🧠 *تحليل إحصائي متعدد العملات ({interval})*\n"
        f"=====================================\n"
        f"{body}\n"
        f"=====================================\n"
        f"⚠️ _تذكير: هذه إشارات أولية. BTC فقط تم اختبارها تاريخياً بجدية "
        f"(Backtested) - باقي العملات قيد المراقبة فقط حالياً، بلا تنفيذ حقيقي._"
    )


# ---------------------------------------------------------------------
# BINANCE TESTNET EXECUTION (optional layer, buy-side only)
# ---------------------------------------------------------------------
def signed_request(method, path, params=None):
    params = params or {}
    params["timestamp"] = int(time.time() * 1000)
    params["recvWindow"] = 10000
    query = "&".join(f"{k}={v}" for k, v in params.items())
    signature = hmac.new(TESTNET_API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    params["signature"] = signature
    headers = {"X-MBX-APIKEY": TESTNET_API_KEY}
    resp = requests.request(method, f"{TESTNET_BASE_URL}{path}", params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json()


def place_market_order(symbol, side, quote_order_qty=None, quantity=None):
    params = {"symbol": symbol, "side": side, "type": "MARKET"}
    if quote_order_qty is not None:
        params["quoteOrderQty"] = quote_order_qty
    if quantity is not None:
        params["quantity"] = quantity
    return signed_request("POST", "/api/v3/order", params)


def load_position(symbol):
    path = position_file(symbol)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_position(symbol, position):
    with open(position_file(symbol), "w") as f:
        json.dump(position, f)


def clear_position(symbol):
    path = position_file(symbol)
    if os.path.exists(path):
        os.remove(path)


def run_execution_step(symbol, price, z):
    """Buy-only paper trading on Testnet, independent per symbol. Never
    trades the sell side — our backtest showed it has no predictive
    value. Each symbol has its own position file, so BTC's trade
    never affects ETH's, etc."""
    position = load_position(symbol)

    if position is None:
        if z < -Z_THRESHOLD:
            try:
                order = place_market_order(symbol, "BUY", quote_order_qty=TRADE_NOTIONAL_USDT)
                fills = order.get("fills", [])
                entry_price = float(fills[0]["price"]) if fills else price
                qty = float(order["executedQty"])
                save_position(symbol, {"entry_price": entry_price, "qty": qty, "opened_candle_count": 0})
                send_telegram_message(
                    f"🟢 *دخول تجريبي (Testnet) — {symbol}*\n"
                    f"السعر: ${entry_price:,.4f} | Z-Score: {z:.2f}\n"
                    f"الكمية: {qty} | سيُغلق بعد {HOLD_CANDLES} شمعة"
                )
                log.info(f"[EXEC:{symbol}] Opened test position: entry={entry_price} qty={qty}")
            except requests.RequestException as e:
                log.error(f"[EXEC:{symbol}] Order failed: {e}")
                pnl_pct = (exit_price - position["entry_price"]) / position["entry_price"] * 100
                send_telegram_message(
                    f"🔴 *خروج تجريبي (Testnet) — {symbol}*\n"
                    f"دخول: ${position['entry_price']:,.4f} | خروج: ${exit_price:,.4f}\n"
                    f"النتيجة: {pnl_pct:+.3f}% (بدون احتساب عمولة)"
                )
                log.info(f"[EXEC:{symbol}] Closed test position: exit={exit_price} pnl={pnl_pct:.3f}%")
                clear_position(symbol)
            except requests.RequestException as e:
                log.error(f"[EXEC:{symbol}] Close order failed: {e}")
        else:
            save_position(symbol, position)
            log.info(f"[EXEC:{symbol}] Holding ({position['opened_candle_count']}/{HOLD_CANDLES})")


# ---------------------------------------------------------------------
# MAIN CYCLE
# ---------------------------------------------------------------------
def run_cycle():
    sections = []
    per_symbol_data = []

    for symbol in SYMBOLS:
        try:
            candles = fetch_candles(symbol, INTERVAL, CANDLE_LIMIT)
            closes = candles[:, 3]
            price = closes[-1]
            z = z_score(closes, LOOKBACK)
            vol = garman_klass_volatility(candles, LOOKBACK)
            sections.append(build_coin_section(symbol, price, z, vol))
            per_symbol_data.append((symbol, price, z))
            log.info(f"{symbol} price={price:.4f} z={z:.2f} gk_vol={vol:.3f}%")
        except requests.RequestException as e:
            log.error(f"Failed to fetch {symbol} this cycle: {e}")

    if sections:
        report = build_report(INTERVAL, sections)
        sent = send_telegram_message(report)
        log.info(f"Combined report sent: {sent}")

    if EXECUTION_ENABLED:
        # BTC only: the backtested, validated buy-side edge. Other coins
        # are alert-only for now until each is individually backtested —
        # this is deliberate, not a bug (see our earlier discussion about
        # not extending an unvalidated signal to more assets).
        for symbol, price, z in per_symbol_data:
            if symbol == "BTCUSDT":
                run_execution_step(symbol, price, z)
    else:
        log.info("[EXEC] Testnet keys not set — execution layer disabled, alerts only.")


def main():
    require_config()
    log.info(
        f"Starting quant_alert for {', '.join(SYMBOLS)} ({INTERVAL}), polling every {POLL_SECONDS}s. "
        f"Testnet execution: {'ENABLED' if EXECUTION_ENABLED else 'disabled'}"
    )
    while True:
        try:
            run_cycle()
        except requests.RequestException as e:
            log.error(f"Network error this cycle, will retry next cycle: {e}")
        except Exception as e:
            log.exception(f"Unexpected error this cycle: {e}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
