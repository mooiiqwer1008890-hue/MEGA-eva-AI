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
SYMBOL = os.environ.get("SYMBOL", "BTCUSDT")
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
POSITION_FILE = "position_state.json"

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


def build_report(symbol, interval, price, z, vol):
    signal = classify_signal(z)
    return (
        f"🧠 *تحليل إحصائي - {symbol} ({interval})*\n"
        f"-------------------------------------\n"
        f"💵 *السعر:* ${price:,.2f}\n\n"
        f"📐 *المؤشرات:*\n"
        f"• Z-Score: {z:.2f}\n"
        f"• تقلب Garman-Klass: {vol:.3f}%\n\n"
        f"-------------------------------------\n"
        f"🎯 *الإشارة:* {signal}\n"
        f"-------------------------------------\n"
        f"⚠️ _تذكير: هذه إشارة إحصائية أولية غير مُختبرة تاريخياً بالكامل. "
        f"لا تُستخدم كقرار تنفيذ مباشر._"
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


def load_position():
    if os.path.exists(POSITION_FILE):
        with open(POSITION_FILE) as f:
            return json.load(f)
    return None


def save_position(position):
    with open(POSITION_FILE, "w") as f:
        json.dump(position, f)


def clear_position():
    if os.path.exists(POSITION_FILE):
        os.remove(POSITION_FILE)


def run_execution_step(price, z):
    """Buy-only paper trading on Testnet. Never trades the sell side —
    our backtest showed it has no predictive value."""
    position = load_position()

    if position is None:
        if z < -Z_THRESHOLD:
            try:
                order = place_market_order(SYMBOL, "BUY", quote_order_qty=TRADE_NOTIONAL_USDT)
                fills = order.get("fills", [])
                entry_price = float(fills[0]["price"]) if fills else price
                qty = float(order["executedQty"])
                save_position({"entry_price": entry_price, "qty": qty, "opened_candle_count": 0})
                send_telegram_message(
                    f"🟢 *دخول تجريبي (Testnet) — {SYMBOL}*\n"
                    f"السعر: ${entry_price:,.2f} | Z-Score: {z:.2f}\n"
                    f"الكمية: {qty} | سيُغلق بعد {HOLD_CANDLES} شمعة"
                )
                log.info(f"[EXEC] Opened test position: entry={entry_price} qty={qty}")
            except requests.RequestException as e:
                log.error(f"[EXEC] Order failed: {e}")
    else:
        position["opened_candle_count"] += 1
        if position["opened_candle_count"] >= HOLD_CANDLES:
            try:
                order = place_market_order(SYMBOL, "SELL", quantity=position["qty"])
                fills = order.get("fills", [])
                exit_price = float(fills[0]["price"]) if fills else price
                pnl_pct = (exit_price - position["entry_price"]) / position["entry_price"] * 100
                send_telegram_message(
                    f"🔴 *خروج تجريبي (Testnet) — {SYMBOL}*\n"
                    f"دخول: ${position['entry_price']:,.2f} | خروج: ${exit_price:,.2f}\n"
                    f"النتيجة: {pnl_pct:+.3f}% (بدون احتساب عمولة)"
                )
                log.info(f"[EXEC] Closed test position: exit={exit_price} pnl={pnl_pct:.3f}%")
                clear_position()
            except requests.RequestException as e:
                log.error(f"[EXEC] Close order failed: {e}")
        else:
            save_position(position)
            log.info(f"[EXEC] Holding ({position['opened_candle_count']}/{HOLD_CANDLES})")


# ---------------------------------------------------------------------
# MAIN CYCLE
# ---------------------------------------------------------------------
def run_cycle():
    candles = fetch_candles(SYMBOL, INTERVAL, CANDLE_LIMIT)
    closes = candles[:, 3]
    price = closes[-1]
    z = z_score(closes, LOOKBACK)
    vol = garman_klass_volatility(candles, LOOKBACK)

    report = build_report(SYMBOL, INTERVAL, price, z, vol)
    sent = send_telegram_message(report)
    log.info(f"{SYMBOL} price={price:.2f} z={z:.2f} gk_vol={vol:.3f}% telegram_sent={sent}")

    if EXECUTION_ENABLED:
        run_execution_step(price, z)
    else:
        log.info("[EXEC] Testnet keys not set — execution layer disabled, alerts only.")


def main():
    require_config()
    log.info(
        f"Starting quant_alert for {SYMBOL} ({INTERVAL}), polling every {POLL_SECONDS}s. "
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
