"""
quant_alert.py (combined version - with Risk Management)
=========================================================
1. Statistical Alert (Z-score + Garman-Klass) - sent to Telegram every cycle.
2. Paper-trading execution on Binance Testnet.
3. Risk Management using Kelly Criterion.
4. Daily Performance Report.
"""

import os
import time
import hmac
import hashlib
import json
import logging
import requests
import numpy as np
import pandas as pd

# استيراد نظام إدارة المخاطر
from risk_manager import (
    calculate_kelly_position_size,
    calculate_var,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    generate_performance_report
)

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
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "900"))

# Testnet execution (optional)
TESTNET_API_KEY = os.environ.get("BINANCE_TESTNET_API_KEY")
TESTNET_API_SECRET = os.environ.get("BINANCE_TESTNET_API_SECRET")
TESTNET_BASE_URL = "https://testnet.binance.vision"
EXECUTION_ENABLED = bool(TESTNET_API_KEY and TESTNET_API_SECRET)

Z_THRESHOLD = 2.0
HOLD_CANDLES = 10

# Risk Management
WIN_RATE = float(os.environ.get("WIN_RATE", "0.55"))
WIN_LOSS_RATIO = float(os.environ.get("WIN_LOSS_RATIO", "1.5"))
KELLY_FRACTION = float(os.environ.get("KELLY_FRACTION", "0.5"))
BALANCE = float(os.environ.get("BALANCE", "1000"))

# Report schedule
REPORT_INTERVAL_SECONDS = 86400  # 24 hours


def position_file(symbol):
    return f"position_state_{symbol}.json"


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("quant_alert")


def require_config():
    missing = [n for n, v in [("TG_BOT_TOKEN", BOT_TOKEN), ("TG_CHAT_ID", CHAT_ID)] if not v]
    if missing:
        raise RuntimeError(f"Missing environment variable(s): {', '.join(missing)}")


# ---------------------------------------------------------------------
# MARKET DATA
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


def z_series(closes, period):
    out = []
    for i in range(period, len(closes) + 1):
        w = closes[i - period:i]
        s = w.std()
        out.append(0.0 if s == 0 else (w[-1] - w.mean()) / s)
    return np.array(out)


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


def send_telegram_photo(png_bytes, caption):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    try:
        resp = requests.post(
            url,
            data={"chat_id": CHAT_ID, "caption": caption[:1000]},
            files={"photo": ("chart.png", png_bytes, "image/png")},
            timeout=30,
        )
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        log.error(f"Telegram photo send failed: {e}")
        return False


def make_chart_png(symbol, candles, period, entry_price=None):
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    o, h, l, c = candles[:, 0], candles[:, 1], candles[:, 2], candles[:, 3]
    n = len(c)
    zs = z_series(c, period)
    zx = np.arange(period - 1, n)
    ma = np.array([c[i - period + 1:i + 1].mean() for i in zx])
    sd = np.array([c[i - period + 1:i + 1].std() for i in zx])

    bg, fg = "#131722", "#d1d4dc"
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]}
    )
    fig.patch.set_facecolor(bg)
    for ax in (ax1, ax2):
        ax.set_facecolor(bg)
        ax.tick_params(colors=fg)
        for spine in ax.spines.values():
            spine.set_color("#2a2e39")
        ax.grid(color="#2a2e39", linewidth=0.5)

    for i in range(n):
        color = "#26a69a" if c[i] >= o[i] else "#ef5350"
        ax1.plot([i, i], [l[i], h[i]], color=color, linewidth=1)
        ax1.add_patch(Rectangle((i - 0.35, min(o[i], c[i])), 0.7, max(abs(c[i] - o[i]), 1e-12), color=color))

    ax1.plot(zx, ma, color="#f5c542", linewidth=1.2, label=f"Mean ({period})")
    ax1.fill_between(zx, ma - 2 * sd, ma + 2 * sd, color="#5c6bc0", alpha=0.15, label="±2σ band")

    pad = (h.max() - l.min()) * 0.05
    ax1.set_ylim(l.min() - pad, h.max() + pad)
    buy_idx = zx[zs < -Z_THRESHOLD]
    if len(buy_idx):
        ax1.scatter(buy_idx, l[buy_idx] - pad * 0.5, marker="^", color="#00e676", s=70, zorder=5,
                    label="Z < -2 (buy zone)")
    if entry_price:
        ax1.axhline(entry_price, color="#00e676", linestyle="--", linewidth=1.3, label=f"Paper entry {entry_price:,.4f}")

    ax1.set_title(f"{symbol}  {INTERVAL}  |  last price {c[-1]:,.4f}  |  Z = {zs[-1]:.2f}", color=fg)
    ax1.legend(loc="upper left", facecolor=bg, edgecolor="#2a2e39", labelcolor=fg, fontsize=8)

    ax2.plot(zx, zs, color="#42a5f5", linewidth=1.2)
    ax2.axhline(Z_THRESHOLD, color="#ef5350", linestyle="--", linewidth=0.9)
    ax2.axhline(-Z_THRESHOLD, color="#00e676", linestyle="--", linewidth=0.9)
    ax2.axhline(0, color="#787b86", linewidth=0.6)
    ax2.set_ylabel("Z-Score", color=fg)
    ax2.set_xlabel(f"last {n} candles ({INTERVAL})", color=fg)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def send_chart(symbol, candles, caption, entry_price=None):
    try:
        png = make_chart_png(symbol, candles, LOOKBACK, entry_price=entry_price)
    except Exception as e:
        log.exception(f"Chart rendering failed for {symbol}: {e}")
        return False
    return send_telegram_photo(png, caption)


# ---------------------------------------------------------------------
# BINANCE TESTNET EXECUTION
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


def run_execution_step(symbol, price, z, candles):
    """Buy-only paper trading on Testnet with Kelly Position Sizing."""
    position = load_position(symbol)

    if position is None:
        if z < -Z_THRESHOLD:
            # استخدام Kelly لتحديد حجم الصفقة
            position_size = calculate_kelly_position_size(
                WIN_RATE, WIN_LOSS_RATIO, BALANCE, KELLY_FRACTION
            )
            if position_size <= 0:
                log.info(f"[EXEC:{symbol}] Kelly position size = 0. Skipping trade.")
                return
            
            try:
                order = place_market_order(symbol, "BUY", quote_order_qty=position_size)
            except requests.RequestException as e:
                log.error(f"[EXEC:{symbol}] BUY failed: {e}")
                return
            
            fills = order.get("fills", [])
            entry_price = float(fills[0]["price"]) if fills else price
            qty = float(order["executedQty"])
            save_position(symbol, {"entry_price": entry_price, "qty": qty, "opened_candle_count": 0})
            
            send_chart(
                symbol, candles,
                f"PAPER ENTRY (Testnet) {symbol}\nprice {entry_price:,.4f} | Z {z:.2f}\n"
                f"qty {qty} | size ${position_size:.2f} | auto-exit after {HOLD_CANDLES} candles",
                entry_price=entry_price,
            )
            send_telegram_message(
                f"🟢 *دخول تجريبي (Testnet) — {symbol}*\n"
                f"السعر: ${entry_price:,.4f} | Z-Score: {z:.2f}\n"
                f"الكمية: {qty} | حجم الصفقة: ${position_size:.2f}\n"
                f"سيُغلق بعد {HOLD_CANDLES} شمعة"
            )
            log.info(f"[EXEC:{symbol}] Opened position: entry={entry_price} qty={qty} size=${position_size:.2f}")
    else:
        position["opened_candle_count"] += 1
        if position["opened_candle_count"] >= HOLD_CANDLES:
            try:
                order = place_market_order(symbol, "SELL", quantity=position["qty"])
            except requests.RequestException as e:
                log.error(f"[EXEC:{symbol}] SELL failed: {e}")
                save_position(symbol, position)
                return
            
            fills = order.get("fills", [])
            exit_price = float(fills[0]["price"]) if fills else price
            pnl_pct = (exit_price - position["entry_price"]) / position["entry_price"] * 100
            pnl = (exit_price - position["entry_price"]) * position["qty"]
            
            send_chart(
                symbol, candles,
                f"PAPER EXIT (Testnet) {symbol}\nentry {position['entry_price']:,.4f} -> exit {exit_price:,.4f}\n"
                f"result {pnl_pct:+.3f}% (before fees)",
                entry_price=position["entry_price"],
            )
            send_telegram_message(
                f"🔴 *خروج تجريبي (Testnet) — {symbol}*\n"
                f"دخول: ${position['entry_price']:,.4f} | خروج: ${exit_price:,.4f}\n"
                f"النتيجة: {pnl_pct:+.3f}% (${pnl:+.2f})"
            )
            log.info(f"[EXEC:{symbol}] Closed position: exit={exit_price} pnl={pnl_pct:.3f}%")
            clear_position(symbol)
        else:
            save_position(symbol, position)
            log.info(f"[EXEC:{symbol}] Holding ({position['opened_candle_count']}/{HOLD_CANDLES})")


# ---------------------------------------------------------------------
# MAIN CYCLE
# ---------------------------------------------------------------------
_in_buy_zone = {}


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
        f"⚠️ _تذكير: هذه إشارات أولية. BTC فقط تم اختبارها تاريخياً بجدية._"
    )


def send_daily_report():
    """يرسل تقريراً يومياً بالأداء إلى Telegram."""
    try:
        # جمع الصفقات من ملفات الحالة
        trades = []
        # ملاحظة: الصفقات المغلقة تُمسح من الملفات، لذا هذا التقرير مبسط
        # في الإصدارات المستقبلية، سنحفظ الصفقات المغلقة في ملف منفصل
        
        # نعرض فقط رصيد الحساب الحالي
        balance_msg = (
            f"📊 *تقرير الأداء اليومي*\n"
            f"========================\n"
            f"💰 رأس المال الحالي: ${BALANCE:.2f}\n"
            f"🎯 نسبة الصفقات الرابحة: {WIN_RATE:.2%}\n"
            f"📈 نسبة الربح/الخسارة: {WIN_LOSS_RATIO:.2f}\n"
            f"⚙️ معامل كيلي: {KELLY_FRACTION:.2f} (Half-Kelly)\n"
            f"========================\n"
            f"⚠️ _هذا تقرير آلي - ليس نصيحة استثمارية._"
        )
        send_telegram_message(balance_msg)
        log.info("Daily report sent successfully.")
    except Exception as e:
        log.exception(f"Failed to send daily report: {e}")


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
            per_symbol_data.append((symbol, price, z, candles))
            log.info(f"{symbol} price={price:.4f} z={z:.2f} gk_vol={vol:.3f}%")
        except requests.RequestException as e:
            log.error(f"Failed to fetch {symbol} this cycle: {e}")

    if sections:
        report = build_report(INTERVAL, sections)
        sent = send_telegram_message(report)
        log.info(f"Combined report sent: {sent}")

    # Chart on entering buy zone
    for symbol, price, z, candles in per_symbol_data:
        now_in_zone = z < -Z_THRESHOLD
        was_in_zone = _in_buy_zone.get(symbol, False)
        if now_in_zone and not was_in_zone:
            pos = load_position(symbol)
            send_chart(
                symbol, candles,
                f"{symbol}: Z = {z:.2f} entered the buy zone (< -{Z_THRESHOLD}).\n"
                f"Statistical signal only - not advice.",
                entry_price=pos["entry_price"] if pos else None,
            )
        _in_buy_zone[symbol] = now_in_zone

    if EXECUTION_ENABLED:
        for symbol, price, z, candles in per_symbol_data:
            if symbol == "BTCUSDT":
                run_execution_step(symbol, price, z, candles)
    else:
        log.info("[EXEC] Testnet keys not set — execution layer disabled, alerts only.")


def main():
    require_config()
    log.info(
        f"Starting quant_alert for {', '.join(SYMBOLS)} ({INTERVAL}), polling every {POLL_SECONDS}s. "
        f"Testnet execution: {'ENABLED' if EXECUTION_ENABLED else 'disabled'}"
    )
    
    last_report_time = time.time()
    
    while True:
        try:
            run_cycle()
            
            # إرسال التقرير اليومي كل 24 ساعة
            if time.time() - last_report_time >= REPORT_INTERVAL_SECONDS:
                send_daily_report()
                last_report_time = time.time()
                
        except requests.RequestException as e:
            log.error(f"Network error this cycle, will retry next cycle: {e}")
        except Exception as e:
            log.exception(f"Unexpected error this cycle: {e}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
