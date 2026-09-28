"""
quant_alert.py (combined version - with HMM Regime Detection)
==============================================================
1. Statistical Alert (Z-score + Garman-Klass) - sent to Telegram.
2. Paper Trading (local simulation) via paper_trader.py.
3. Risk Management via Kelly Criterion.
4. HMM Regime Detection - filter trades in Bear markets.
5. Daily Performance Report.
"""

import os
import time
import json
import logging
import requests
import numpy as np
import pandas as pd

from risk_manager import (
    calculate_kelly_position_size,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    generate_performance_report,
)
from paper_trader import (
    open_paper_position,
    check_and_close_position,
    load_balance,
    get_stats,
    TRADES_LOG,
)
from hmm_regime import (
    detect_regimes,
    get_current_regime,
    filter_signals_by_regime,
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
CANDLE_LIMIT = int(os.environ.get("CANDLE_LIMIT", "200"))
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "900"))

# HMM Config
HMM_INTERVAL = os.environ.get("HMM_INTERVAL", "4h")
HMM_LIMIT = int(os.environ.get("HMM_LIMIT", "500"))
HMM_STATES = int(os.environ.get("HMM_STATES", "2"))
USE_HMM_FILTER = os.environ.get("USE_HMM_FILTER", "true").lower() == "true"

Z_THRESHOLD = 2.0
REPORT_INTERVAL_SECONDS = 86400  # 24 hours

# ---------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------
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
    """يجلب الشموع من Binance Data API."""
    url = "https://data-api.binance.vision/api/v3/klines"
    resp = requests.get(
        url,
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=15,
    )
    resp.raise_for_status()
    raw = resp.json()
    return np.array([[float(k[1]), float(k[2]), float(k[3]), float(k[4])] for k in raw])


def fetch_closes(symbol, interval="4h", limit=500):
    """يجلب أسعار الإغلاق فقط (لـ HMM)."""
    url = "https://data-api.binance.vision/api/v3/klines"
    resp = requests.get(
        url,
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=20,
    )
    resp.raise_for_status()
    raw = resp.json()
    closes = pd.Series([float(k[4]) for k in raw])
    returns = np.log(closes / closes.shift(1)).dropna()
    return returns


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
# HMM REGIME DETECTION
# ---------------------------------------------------------------------
def get_market_regime(symbol):
    """
    يحصل على النظام الحالي للسوق (Bull/Bear) باستخدام HMM.
    """
    if not USE_HMM_FILTER:
        return {"current_regime": "Bull", "regime_prob": 1.0}

    try:
        returns = fetch_closes(symbol, interval=HMM_INTERVAL, limit=HMM_LIMIT)
        if len(returns) < 100:
            log.warning(f"[HMM:{symbol}] Not enough data ({len(returns)}). Skipping filter.")
            return {"current_regime": "Bull", "regime_prob": 1.0}

        result = get_current_regime(returns, n_states=HMM_STATES)
        log.info(
            f"[HMM:{symbol}] Regime={result['current_regime']} "
            f"(prob={result['regime_prob']:.2%})"
        )
        return result
    except Exception as e:
        log.exception(f"[HMM:{symbol}] Failed: {e}")
        return {"current_regime": "Bull", "regime_prob": 1.0}


# ---------------------------------------------------------------------
# TELEGRAM
# ---------------------------------------------------------------------
def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(
            url,
            data={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
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


def make_chart_png(symbol, candles, period, entry_price=None, regime=None):
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
        ax1.axhline(entry_price, color="#00e676", linestyle="--", linewidth=1.3, label=f"Entry {entry_price:,.4f}")

    # إضافة النظام إلى العنوان
    regime_text = f" | Regime: {regime}" if regime else ""
    ax1.set_title(
        f"{symbol}  {INTERVAL}  |  last price {c[-1]:,.4f}  |  Z = {zs[-1]:.2f}{regime_text}",
        color=fg
    )
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


def send_chart(symbol, candles, caption, entry_price=None, regime=None):
    try:
        png = make_chart_png(symbol, candles, LOOKBACK, entry_price=entry_price, regime=regime)
    except Exception as e:
        log.exception(f"Chart rendering failed for {symbol}: {e}")
        return False
    return send_telegram_photo(png, caption)


# ---------------------------------------------------------------------
# EXECUTION (Paper Trading + HMM Filter)
# ---------------------------------------------------------------------
def run_execution_step(symbol, price, z, candles):
    """تشغيل منطق التداول الورقي مع فلتر HMM."""
    position = None
    pos_file = f"paper_position_{symbol}.json"
    if os.path.exists(pos_file):
        with open(pos_file) as f:
            position = json.load(f)

    # الحصول على النظام الحالي
    regime_info = get_market_regime(symbol)
    current_regime = regime_info["current_regime"]

    if position is None:
        # فتح صفقة جديدة إذا Z < -2
        if z < -Z_THRESHOLD:
            # فلترة الإشارة بواسطة HMM
            signal = filter_signals_by_regime(1, current_regime)

            if signal == 0:
                log.info(
                    f"[EXEC:{symbol}] Buy signal BLOCKED by HMM "
                    f"(Regime={current_regime})"
                )
                send_telegram_message(
                    f"⚠️ *إشارة شراء مرفوضة — {symbol}*\n"
                    f"Z-Score: {z:.2f} (منطقة الشراء)\n"
                    f"النظام: *{current_regime}*\n"
                    f"_HMM يمنع الشراء في السوق الهابط._"
                )
                return

            # تنفيذ الصفقة
            new_pos = open_paper_position(symbol, price, z)
            if new_pos is None:
                return
            balance = load_balance()
            send_chart(
                symbol, candles,
                f"PAPER ENTRY {symbol}\nprice {new_pos['entry_price']:,.4f} | Z {z:.2f}\n"
                f"qty {new_pos['quantity']:.6f} | size ${new_pos['position_size_usd']:.2f}\n"
                f"SL {new_pos['stop_loss']:,.4f} | TP {new_pos['take_profit']:,.4f}\n"
                f"Balance: ${balance:.2f} | Regime: {current_regime}",
                entry_price=new_pos['entry_price'],
                regime=current_regime,
            )
            send_telegram_message(
                f"🟢 *فتح صفقة (Paper) — {symbol}*\n"
                f"السعر: ${new_pos['entry_price']:,.4f} | Z: {z:.2f}\n"
                f"الكمية: {new_pos['quantity']:.6f}\n"
                f"حجم الصفقة: ${new_pos['position_size_usd']:.2f} (Kelly)\n"
                f"وقف الخسارة: ${new_pos['stop_loss']:,.4f}\n"
                f"هدف الربح: ${new_pos['take_profit']:,.4f}\n"
                f"النظام: *{current_regime}* | الرصيد: ${balance:.2f}"
            )
    else:
        # إدارة الصفقة المفتوحة
        trade = check_and_close_position(symbol, price)
        if trade:
            balance = load_balance()
            send_chart(
                symbol, candles,
                f"PAPER EXIT {symbol} ({trade['exit_reason']})\n"
                f"entry {trade['entry_price']:,.4f} -> exit {trade['exit_price']:,.4f}\n"
                f"P&L {trade['pnl_pct']:+.3f}% (${trade['pnl_usd']:+.4f})\n"
                f"Balance: ${balance:.2f}",
                entry_price=trade['entry_price'],
                regime=current_regime,
            )
            emoji = "🟢" if trade['pnl_usd'] > 0 else "🔴"
            send_telegram_message(
                f"{emoji} *إغلاق صفقة (Paper) — {symbol}*\n"
                f"السبب: {trade['exit_reason']}\n"
                f"دخول: ${trade['entry_price']:,.4f}\n"
                f"خروج: ${trade['exit_price']:,.4f}\n"
                f"النتيجة: {trade['pnl_pct']:+.3f}% (${trade['pnl_usd']:+.4f})\n"
                f"الرصيد: ${balance:.2f}"
            )


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


def send_daily_report():
    """يرسل تقرير الأداء اليومي من ملف الصفقات."""
    try:
        stats = get_stats()
        if stats is None:
            return
        message = (
            f"📊 *التقرير اليومي للأداء*\n"
            f"========================\n"
            f"💰 الرصيد: ${stats['balance']:.2f} "
            f"(من ${stats['initial_balance']:.2f})\n"
            f"📈 العائد: {stats['total_return_pct']:+.3f}%\n"
            f"💵 صافي الربح: ${stats['net_pnl_usd']:+.4f}\n"
            f"📋 عدد الصفقات: {stats['total_trades']}\n"
            f"🎯 نسبة الرابحة: {stats['win_rate']:.2%}\n"
            f"✅ صفقات رابحة: {stats['winning_trades']}\n"
            f"❌ صفقات خاسرة: {stats['losing_trades']}\n"
            f"📊 متوسط الربح: ${stats['avg_win_usd']:.4f}\n"
            f"📊 متوسط الخسارة: ${stats['avg_loss_usd']:.4f}\n"
            f"========================\n"
            f"⚠️ _تقرير آلي - ليس نصيحة استثمارية._"
        )
        send_telegram_message(message)
        log.info("Daily report sent successfully.")
    except Exception as e:
        log.exception(f"Failed to send daily report: {e}")


def run_cycle():
    per_symbol_data = []

    for symbol in SYMBOLS:
        try:
            candles = fetch_candles(symbol, INTERVAL, CANDLE_LIMIT)
            closes = candles[:, 3]
            price = closes[-1]
            z = z_score(closes, LOOKBACK)
            vol = garman_klass_volatility(candles, LOOKBACK)
            per_symbol_data.append((symbol, price, z, candles))
            log.info(f"{symbol} price={price:.4f} z={z:.2f} gk_vol={vol:.3f}%")
        except requests.RequestException as e:
            log.error(f"Failed to fetch {symbol} this cycle: {e}")

    # Chart & Alert on entering buy zone
    for symbol, price, z, candles in per_symbol_data:
        now_in_zone = z < -Z_THRESHOLD
        was_in_zone = _in_buy_zone.get(symbol, False)
        if now_in_zone and not was_in_zone:
            regime_info = get_market_regime(symbol)
            send_chart(
                symbol, candles,
                f"{symbol}: Z = {z:.2f} entered the buy zone (< -{Z_THRESHOLD}).\n"
                f"Regime: {regime_info['current_regime']}\n"
                f"Statistical signal only - not advice.",
                regime=regime_info['current_regime'],
            )
            send_telegram_message(
                f"🟢 *{symbol} دخل منطقة الشراء*\n"
                f"السعر: ${price:,.4f} | Z: {z:.2f}\n"
                f"النظام: *{regime_info['current_regime']}*\n"
                f"⚠️ _إشارة إحصائية - Paper Trading قيد التنفيذ._"
            )
        _in_buy_zone[symbol] = now_in_zone

    # تشغيل التداول الورقي
    for symbol, price, z, candles in per_symbol_data:
        run_execution_step(symbol, price, z, candles)


def main():
    require_config()
    balance = load_balance()
    log.info(
        f"Starting quant_alert (Paper + HMM) for {', '.join(SYMBOLS)} "
        f"({INTERVAL}), polling every {POLL_SECONDS}s. Balance=${balance:.2f}"
    )

    send_telegram_message(
        f"🚀 *البوت يعمل الآن (Paper + HMM)*\n"
        f"========================\n"
        f"💰 الرصيد: ${balance:.2f}\n"
        f"📊 العملات: {', '.join(SYMBOLS)}\n"
        f"⏱️ الفترة: {INTERVAL}\n"
        f"🧠 فلتر HMM: {'مُفعّل' if USE_HMM_FILTER else 'مُعطّل'}\n"
        f"📊 HMM Interval: {HMM_INTERVAL} ({HMM_LIMIT} شمعة)\n"
        f"========================\n"
        f"⚠️ *تداول وهمي - لا مخاطر مالية.*\n"
        f"📌 _HMM يمنع التداول في النظام Bear._"
    )

    last_report_time = time.time()

    while True:
        try:
            run_cycle()
            if time.time() - last_report_time >= REPORT_INTERVAL_SECONDS:
                send_daily_report()
                last_report_time = time.time()
        except requests.RequestException as e:
            log.error(f"Network error this cycle: {e}")
        except Exception as e:
            log.exception(f"Unexpected error this cycle: {e}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
