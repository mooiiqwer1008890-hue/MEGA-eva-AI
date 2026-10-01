"""
quant_alert.py (Full Stack — Fast + Full + HMM + GARCH + Sentiment + Correlation + FFC)
======================================================================================

النظام الكامل مع 6 طبقات حماية:
1. Z-score < -2 (الإشارة الأساسية)
2. HMM Regime (يجب Bull)
3. GARCH Volatility (يجب غير HIGH)
4. Sentiment (يجب غير BEARISH قوي) — CoinMarketCap Keyless API
5. Correlation (يجب غير مرتبط)
6. FFC (يجب نشط)

+ Fast Exit Check: يفحص SL/TP كل 5 دقائق (وليس كل 30)
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
    load_position,
)
from hmm_regime import (
    get_current_regime,
    filter_signals_by_regime,
    fetch_returns as hmm_fetch_returns,
)
from garch_model import (
    fetch_returns as garch_fetch_returns,
    forecast_volatility,
    classify_volatility,
    adjust_position_size_by_volatility,
)

# ═══════════════════════════════════════════════════════════════
# إضافات: Correlation Filter + FFC + Sentiment Filter
# ═══════════════════════════════════════════════════════════════
from correlation_filter import get_correlation_filter
from ffc import get_ffc
from sentiment_filter import get_sentiment_filter

correlation_filter = get_correlation_filter()
ffc = get_ffc()
sentiment_filter = get_sentiment_filter()
# ═══════════════════════════════════════════════════════════════


# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
BOT_TOKEN = os.environ.get("TG_BOT_TOKEN")
CHAT_ID = os.environ.get("TG_CHAT_ID")
SYMBOLS = [s.strip() for s in os.environ.get(
    "SYMBOLS", "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT"
).split(",") if s.strip()]
INTERVAL = os.environ.get("INTERVAL", "15m")
LOOKBACK = int(os.environ.get("LOOKBACK", "20"))
CANDLE_LIMIT = int(os.environ.get("CANDLE_LIMIT", "200"))

# Fast/Slow Cycle Config
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "1800"))
FAST_POLL_SECONDS = int(os.environ.get("FAST_POLL_SECONDS", "300"))

# HMM Config
HMM_INTERVAL = os.environ.get("HMM_INTERVAL", "4h")
HMM_LIMIT = int(os.environ.get("HMM_LIMIT", "300"))
HMM_STATES = int(os.environ.get("HMM_STATES", "2"))
USE_HMM_FILTER = os.environ.get("USE_HMM_FILTER", "true").lower() == "true"

# GARCH Config
GARCH_INTERVAL = os.environ.get("GARCH_INTERVAL", "4h")
GARCH_LIMIT = int(os.environ.get("GARCH_LIMIT", "500"))
USE_GARCH_FILTER = os.environ.get("USE_GARCH_FILTER", "true").lower() == "true"

# Sentiment Config (بدون API Key — CoinMarketCap Keyless)
USE_SENTIMENT_FILTER = os.environ.get("USE_SENTIMENT_FILTER", "true").lower() == "true"

Z_THRESHOLD = 2.0
Z_STRONG_THRESHOLD = 2.5
REPORT_INTERVAL_SECONDS = 86400  # 24 hours


# -----------------------------------------------------------
# LOGGING
# -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("quant_alert")


def require_config():
    missing = [n for n, v in [("TG_BOT_TOKEN", BOT_TOKEN),
                               ("TG_CHAT_ID", CHAT_ID)] if not v]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")


# -----------------------------------------------------------
# MARKET DATA
# -----------------------------------------------------------
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
    return np.array([[float(k[1]), float(k[2]), float(k[3]),
                      float(k[4]), float(k[5])] for k in raw])


# -----------------------------------------------------------
# INDICATORS
# -----------------------------------------------------------
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
    variance = (0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2)
    return np.sqrt(max(variance.mean(), 0)) * 100.0


def classify_signal(z):
    if z < -2.0:
        return "🟢 شراء إحصائي (انحراف سلبي قوي عن المتوسط)"
    elif z > 2.0:
        return "🔴 بيع إحصائي (انحراف إيجابي قوي عن المتوسط)"
    return "⚪ لا إشارة"


# -----------------------------------------------------------
# HMM REGIME
# -----------------------------------------------------------
def get_market_regime(symbol):
    """يحصل على النظام الحالي (Bull/Bear) باستخدام HMM."""
    if not USE_HMM_FILTER:
        return {"current_regime": "Bull", "regime_prob": 1.0}
    try:
        returns = hmm_fetch_returns(symbol, interval=HMM_INTERVAL, limit=HMM_LIMIT)
        if len(returns) < 100:
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


# -----------------------------------------------------------
# GARCH VOLATILITY
# -----------------------------------------------------------
def get_volatility_info(symbol):
    """يحصل على معلومات التقلب باستخدام GARCH."""
    if not USE_GARCH_FILTER:
        return {
            "vol_ratio": 1.0, "vol_regime": "NORMAL",
            "annualized_vol": 0.0, "current_vol": 0.0, "forecast_vol": 0.0,
        }
    try:
        returns = garch_fetch_returns(symbol, interval=GARCH_INTERVAL, limit=GARCH_LIMIT)
        if len(returns) < 100:
            return {
                "vol_ratio": 1.0, "vol_regime": "NORMAL",
                "annualized_vol": 0.0, "current_vol": 0.0, "forecast_vol": 0.0,
            }
        forecast = forecast_volatility(returns, horizon=1)
        vol_ratio = forecast["vol_ratio"]
        vol_regime = classify_volatility(vol_ratio)
        return {
            "vol_ratio": forecast["vol_ratio"],
            "vol_regime": vol_regime,
            "annualized_vol": forecast["annualized_vol"],
            "current_vol": forecast["current_vol"],
            "forecast_vol": forecast["forecast_vol"],
        }
    except Exception as e:
        log.exception(f"[GARCH:{symbol}] Failed: {e}")
        return {
            "vol_ratio": 1.0, "vol_regime": "NORMAL",
            "annualized_vol": 0.0, "current_vol": 0.0, "forecast_vol": 0.0,
        }


# -----------------------------------------------------------
# TELEGRAM
# -----------------------------------------------------------
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
            data={"chat_id": CHAT_ID, "caption": caption, "parse_mode": "Markdown"},
            files={"photo": ("chart.png", png_bytes, "image/png")},
            timeout=30,
        )
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        log.error(f"Telegram photo send failed: {e}")
        return False


# -----------------------------------------------------------
# CHART
# -----------------------------------------------------------
def make_chart_png(symbol, candles, period, entry_price=None,
                   regime=None, vol_regime=None):
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    o, h, l, c = candles[:, 0], candles[:, 1], candles[:, 2], candles[:, 3]
    n = len(c)
    zs = z_series(c, period)
    zx = np.arange(period - 1, n)
    ma = np.array([c[i - period + 1:i + 1].mean() for i in range(period - 1, n)])
    sd = np.array([c[i - period + 1:i + 1].std() for i in range(period - 1, n)])

    bg, fg = "#131722", "#d1d4dc"
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
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
        ax1.add_patch(Rectangle((i - 0.35, min(o[i], c[i])),
                                0.7, abs(c[i] - o[i]) or 1e-9,
                                facecolor=color, edgecolor=color))

    ax1.plot(zx, ma, color="#f5c542", linewidth=1.2, label="Mean (20)")
    ax1.fill_between(zx, ma - 2 * sd, ma + 2 * sd, color="#2962ff",
                     alpha=0.08, label="±2σ band")

    pad = (h.max() - l.min()) * 0.05
    ax1.set_ylim(l.min() - pad, h.max() + pad)
    buy_idx = zx[zs < -Z_THRESHOLD]
    if len(buy_idx):
        ax1.scatter(buy_idx, l[buy_idx] - pad * 0.5, marker="^",
                    color="#26a69a", s=40, label="Z < -2 (buy zone)")
    if entry_price:
        ax1.axhline(entry_price, color="#00e676", linestyle="--",
                    linewidth=1, label=f"Entry {entry_price:.4f}")

    title = f"{symbol}  {INTERVAL}  |  last {c[-1]:.4f}  |  Z = {zs[-1]:.2f}"
    if regime:
        title += f"  |  {regime}"
    if vol_regime:
        title += f"  |  Vol: {vol_regime}"
    ax1.set_title(title, color=fg, fontsize=11)

    ax2.plot(zx, zs, color="#2962ff", linewidth=1.2)
    ax2.axhline(-Z_THRESHOLD, color="#ef5350", linestyle="--", linewidth=0.8)
    ax2.axhline(Z_THRESHOLD, color="#ef5350", linestyle="--", linewidth=0.8)
    ax2.axhline(0, color="#787b86", linewidth=0.6)
    ax2.set_ylabel("Z-Score", color=fg, fontsize=9)
    ax2.set_xlabel(f"last {len(c)} candles ({INTERVAL})", color=fg, fontsize=9)
    ax1.legend(facecolor=bg, edgecolor="#2a2e39", labelcolor=fg, fontsize=8)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=bg, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def send_chart(symbol, candles, caption, entry_price=None,
               regime=None, vol_regime=None):
    try:
        png = make_chart_png(
            symbol, candles, LOOKBACK,
            entry_price=entry_price, regime=regime, vol_regime=vol_regime,
        )
    except Exception as e:
        log.exception(f"Chart rendering failed for {symbol}: {e}")
        return False
    return send_telegram_photo(png, caption)


# -----------------------------------------------------------
# HELPERS
# -----------------------------------------------------------
def _get_open_position_symbols():
    """جلب قائمة العملات التي لديها صفقات مفتوحة."""
    open_symbols = []
    for symbol in SYMBOLS:
        pos_file = f"paper_position_{symbol}.json"
        if os.path.exists(pos_file):
            open_symbols.append(symbol)
    return open_symbols


# -----------------------------------------------------------
# EXECUTION
# -----------------------------------------------------------
def run_execution_step(symbol, price, z, candles):
    """
    تشغيل منطق التداول الورقي مع 6 فلاتر.
    """
    # ═══ 0. FFC Check ═══
    if not ffc.can_open_position():
        log.info(f"[EXEC:{symbol}] 🔴 FFC OFF — تم إيقاف التداول")
        return

    position = None
    pos_file = f"paper_position_{symbol}.json"
    if os.path.exists(pos_file):
        with open(pos_file) as f:
            position = json.load(f)

    regime_info = get_market_regime(symbol)
    current_regime = regime_info["current_regime"]

    vol_info = get_volatility_info(symbol)
    vol_ratio = vol_info["vol_ratio"]
    vol_regime = vol_info["vol_regime"]

    # ═══ لا توجد صفقة مفتوحة ═══
    if position is None:
        if z < -Z_THRESHOLD:
            # ─── 1. HMM Filter ───
            signal = filter_signals_by_regime(1, current_regime)
            if signal == 0:
                log.info(f"[EXEC:{symbol}] Buy BLOCKED by HMM ({current_regime})")
                send_telegram_message(
                    f"⚠️ *إشارة شراء مرفوضة — {symbol}*\n"
                    f"Z-Score: `{z:.2f}`\n"
                    f"النظام: *{current_regime}*\n"
                    f"_HMM يمنع الشراء في السوق الهابط._"
                )
                return

            # ─── 2. GARCH Filter ───
            if vol_regime == "HIGH" and vol_ratio > 2.0:
                log.info(f"[EXEC:{symbol}] Buy BLOCKED by GARCH ({vol_regime})")
                send_telegram_message(
                    f"⚠️ *إشارة شراء مرفوضة — {symbol}*\n"
                    f"Z-Score: `{z:.2f}`\n"
                    f"النظام: *{current_regime}*\n"
                    f"التقلب: *{vol_regime}* (ratio={vol_ratio:.2f})\n"
                    f"_GARCH يمنع الشراء في التقلب المرتفع._"
                )
                return

            # ─── 3. Sentiment Filter ───
            sentiment_multiplier = 1.0
            sentiment_info = {"label": "neutral", "score": 0.0, "count": 0}
            if USE_SENTIMENT_FILTER:
                sent_result = sentiment_filter.filter_signal("BUY", symbol)
                sentiment_info = sent_result["sentiment"]
                if not sent_result["allowed"]:
                    log.info(f"[EXEC:{symbol}] Buy BLOCKED by Sentiment: {sent_result['reason']}")
                    send_telegram_message(
                        f"⚠️ *إشارة شراء مرفوضة — {symbol}*\n"
                        f"Z-Score: `{z:.2f}`\n"
                        f"السبب: *الأخبار سلبية*\n"
                        f"`{sent_result['reason']}`\n"
                        f"_Sentiment Filter يمنع الشراء في الأخبار السيئة._"
                    )
                    return
                sentiment_multiplier = sent_result["multiplier"]
                log.info(
                    f"[EXEC:{symbol}] Sentiment OK: {sent_result['reason']} "
                    f"(mult={sentiment_multiplier:.2f})"
                )

            # ─── 4. Correlation Filter ───
            open_symbols = _get_open_position_symbols()
            can_open, reason = correlation_filter.can_open_position(symbol, open_symbols)
            if not can_open:
                log.info(f"[EXEC:{symbol}] Buy BLOCKED by Correlation: {reason}")
                send_telegram_message(
                    f"⚠️ *إشارة شراء مرفوضة — {symbol}*\n"
                    f"Z-Score: `{z:.2f}`\n"
                    f"السبب: *ارتباط عالي*\n"
                    f"`{reason}`\n"
                    f"_Correlation Filter يمنع الصفقات المترابطة._"
                )
                return

            # ✅ كل الفلاتر نجحت — افتح الصفقة
            new_pos = open_paper_position(symbol, price, z)
            if new_pos is None:
                return

            original_size = new_pos["position_size_usd"]
            garch_size = adjust_position_size_by_volatility(original_size, vol_ratio)
            final_size = garch_size * sentiment_multiplier
            new_pos["quantity"] = final_size / price
            new_pos["position_size_usd"] = final_size

            with open(pos_file, "w") as f:
                json.dump(new_pos, f)

            balance = load_balance()
            sent_label = sentiment_info.get("label", "neutral")
            sent_score = sentiment_info.get("score", 0.0)
            sent_count = sentiment_info.get("count", 0)

            send_chart(
                symbol, candles,
                f"PAPER ENTRY {symbol}\n"
                f"price {new_pos['entry_price']:.4f} | Z {z:.2f}\n"
                f"qty {new_pos['quantity']:.6f} | size ${final_size:.2f}\n"
                f"Sentiment: {sent_label} ({sent_score:+.2f}, {sent_count} news)\n"
                f"Balance: ${balance:.2f} | {current_regime} | {vol_regime}",
                entry_price=new_pos["entry_price"],
                regime=current_regime,
                vol_regime=vol_regime,
            )
            send_telegram_message(
                f"🟢 *فتح صفقة (Paper) — {symbol}*\n"
                f"السعر: `${new_pos['entry_price']:.4f}` | Z: `{z:.2f}`\n"
                f"الكمية: `{new_pos['quantity']:.6f}`\n"
                f"حجم الصفقة: `${final_size:.2f}`\n"
                f"GARCH × Sentiment: `×{sentiment_multiplier:.2f}`\n"
                f"النظام: *{current_regime}* | التقلب: *{vol_regime}*\n"
                f"الأخبار: {sent_label} ({sent_score:+.2f}, {sent_count} items)\n"
                f"الرصيد: `${balance:.2f}`"
            )
        else:
            pass

    else:
        # ─── صفقة موجودة — فكر في إغلاقها ───
        trade = check_and_close_position(symbol, price)
        if trade:
            balance = load_balance()
            ffc.update_after_trade(trade["pnl_usd"], balance)

            send_chart(
                symbol, candles,
                f"PAPER EXIT {symbol} ({trade['exit_reason']})\n"
                f"entry {trade['entry_price']:.4f} -> exit {trade['exit_price']:.4f}\n"
                f"P&L {trade['pnl_pct']:+.3f}% (${trade['pnl_usd']:+.2f})\n"
                f"Balance: ${balance:.2f}",
                entry_price=trade["entry_price"],
                regime=current_regime,
                vol_regime=vol_regime,
            )
            emoji = "🟢" if trade["pnl_usd"] > 0 else "🔴"
            send_telegram_message(
                f"{emoji} *إغلاق صفقة (Paper) — {symbol}*\n"
                f"السبب: `{trade['exit_reason']}`\n"
                f"دخول: `${trade['entry_price']:.4f}`\n"
                f"خروج: `${trade['exit_price']:.4f}`\n"
                f"النتيجة: `{trade['pnl_pct']:+.3f}%` (${trade['pnl_usd']:+.2f})\n"
                f"الرصيد: `${balance:.2f}`"
            )


# -----------------------------------------------------------
# MAIN CYCLE
# -----------------------------------------------------------
_in_buy_zone = {}


def check_urgent_signals(per_symbol_data):
    """يفحص الإشارات العاجلة."""
    for symbol, price, z, candles in per_symbol_data:
        if z < -Z_STRONG_THRESHOLD:
            log.info(f"[URGENT:{symbol}] Strong buy signal z={z:.2f}")
            send_telegram_message(
                f"🚨 *إشارة عاجلة — {symbol}*\n"
                f"Z-Score: *{z:.2f}* (انحراف قوي جداً)\n"
                f"السعر: `${price:.4f}`\n"
                f"_إشارة شراء قوية._"
            )

        now_in_zone = z < -Z_THRESHOLD
        was_in_zone = _in_buy_zone.get(symbol, False)

        if now_in_zone and not was_in_zone:
            send_telegram_message(
                f"🟢 *{symbol}* دخل منطقة الشراء\n"
                f"Z-Score: `{z:.2f}`\n"
                f"السعر: `${price:.4f}`\n"
                f"_إشارة إحصائية - لا نصيحة._"
            )

        _in_buy_zone[symbol] = now_in_zone


def run_cycle_fast():
    """
    دورة سريعة (كل 5 دقائق).
    ✅ تفحص الإشارات العاجلة + تفحص SL/TP للصفقات المفتوحة.
    """
    per_symbol_data = []
    for symbol in SYMBOLS:
        try:
            candles = fetch_candles(symbol, INTERVAL, CANDLE_LIMIT)
            closes = candles[:, 3]
            price = closes[-1]
            z = z_score(closes, LOOKBACK)
            per_symbol_data.append((symbol, price, z, candles))
            log.info(f"[FAST] {symbol} price={price:.4f} z={z:.2f}")
        except requests.RequestException as e:
            log.error(f"[FAST] Failed to fetch {symbol}: {e}")

    if per_symbol_data:
        check_urgent_signals(per_symbol_data)

    # ═══════════════════════════════════════════════════════════
    # 🆕 فحص سريع لـ SL/TP (حماية فورية كل 5 دقائق)
    # ═══════════════════════════════════════════════════════════
    for symbol, price, z, candles in per_symbol_data:
        try:
            position = load_position(symbol)
            if position is not None:
                trade = check_and_close_position(symbol, price)
                if trade:
                    balance = load_balance()
                    ffc.update_after_trade(trade["pnl_usd"], balance)
                    log.info(
                        f"[FAST-EXIT:{symbol}] {trade['exit_reason']} "
                        f"P&L={trade['pnl_pct']:+.3f}% "
                        f"duration={trade.get('duration_minutes', 0):.1f}min"
                    )
                    emoji = "🟢" if trade["pnl_usd"] > 0 else "🔴"
                    send_telegram_message(
                        f"{emoji} *إغلاق سريع (Paper) — {symbol}*\n"
                        f"السبب: `{trade['exit_reason']}`\n"
                        f"النتيجة: `{trade['pnl_pct']:+.3f}%` "
                        f"(${trade['pnl_usd']:+.2f})\n"
                        f"المدة: `{trade.get('duration_minutes', 0):.0f} دقيقة`\n"
                        f"الرصيد: `${balance:.2f}`"
                    )
        except Exception as e:
            log.warning(f"[FAST-EXIT:{symbol}] فشل فحص الخروج: {e}")


def run_cycle_full():
    """دورة كاملة (كل 30 دقيقة) — كل الفلاتر."""
    per_symbol_data = []

    # تحديث Correlation Filter
    for symbol in SYMBOLS:
        try:
            candles = fetch_candles(symbol, INTERVAL, CANDLE_LIMIT)
            closes = candles[:, 3]
            correlation_filter.update_prices(symbol, closes)
        except Exception as e:
            log.warning(f"[FULL] فشل تحديث correlation لـ {symbol}: {e}")

    # تحديث cache الأخبار
    if USE_SENTIMENT_FILTER:
        try:
            from news_fetcher import update_news_cache
            bases = [s.replace("USDT", "") for s in SYMBOLS]
            update_news_cache(symbols=bases)
            log.info("[FULL] تم تحديث news cache")
        except Exception as e:
            log.warning(f"[FULL] فشل تحديث news cache: {e}")

    for symbol in SYMBOLS:
        try:
            candles = fetch_candles(symbol, INTERVAL, CANDLE_LIMIT)
            closes = candles[:, 3]
            price = closes[-1]
            z = z_score(closes, LOOKBACK)
            vol = garman_klass_volatility(candles, LOOKBACK)
            per_symbol_data.append((symbol, price, z, candles))
            log.info(f"[FULL] {symbol} price={price:.4f} z={z:.2f}")
        except requests.RequestException as e:
            log.error(f"[FULL] Failed to fetch {symbol}: {e}")

    for symbol, price, z, candles in per_symbol_data:
        run_execution_step(symbol, price, z, candles)


def send_daily_report():
    """يرسل التقرير اليومي."""
    try:
        stats = get_stats()
        if stats is None:
            return

        ffc_info = ffc.status()
        ffc_emoji = "🟢" if ffc_info["is_live"] else "🔴"

        # حالة Sentiment
        sent_lines = []
        if USE_SENTIMENT_FILTER:
            for symbol in SYMBOLS:
                s = sentiment_filter.get_sentiment(symbol)
                emoji = "🟢" if s["label"] == "bullish" else "🔴" if s["label"] == "bearish" else "⚪"
                sent_lines.append(
                    f"   {emoji} {symbol}: {s['label']} "
                    f"({s['score']:+.2f}, {s['count']} خبر)"
                )
            sent_summary = "\n".join(sent_lines) if sent_lines else "   (لا توجد بيانات)"
            sent_section = (
                f"════════════════════\n"
                f"📰 *حالة الأخبار*\n"
                f"{sent_summary}\n"
            )
        else:
            sent_section = "📰 *Sentiment*: `معطل`\n"

        message = (
            f"📊 *التقرير اليومي للأداء*\n"
            f"════════════════════\n"
            f"💰 الرصيد: `${stats['balance']:.2f}` "
            f"(من `${stats['initial_balance']:.2f}`)\n"
            f"📈 العائد: `{stats['total_return_pct']:+.3f}%`\n"
            f"💵 صافي الربح: `${stats['net_pnl_usd']:+.4f}`\n"
            f"📊 عدد الصفقات: `{stats['total_trades']}`\n"
            f"🎯 نسبة الرابحة: `{stats['win_rate']:.2%}`\n"
            f"✅ صفقات رابحة: `{stats['winning_trades']}`\n"
            f"❌ صفقات خاسرة: `{stats['losing_trades']}`\n"
            f"📈 متوسط الربح: `${stats['avg_win_usd']:.4f}`\n"
            f"📉 متوسط الخسارة: `${stats['avg_loss_usd']:.4f}`\n"
            f"════════════════════\n"
            f"{ffc_emoji} *حالة FFC (حماية رأس المال)*\n"
            f"الحالة: {'نشط' if ffc_info['is_live'] else 'متوقف'}\n"
            f"الأداء: `{ffc_info['fitness']:+.2f}%`\n"
            f"مرات الإيقاف: `{ffc_info['total_halts']}`\n"
            f"════════════════════\n"
            f"{sent_section}"
            f"════════════════════\n"
            f"⚠️ _تقرير آلي — ليس نصيحة استثمارية._"
        )
        send_telegram_message(message)
        log.info("Daily report sent successfully.")
    except Exception as e:
        log.exception(f"Failed to send daily report: {e}")


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
def main():
    require_config()
    balance = load_balance()

    sentiment_status = "مفعل" if USE_SENTIMENT_FILTER else "معطل"

    log.info(
        f"Starting quant_alert (Full Stack) for {', '.join(SYMBOLS)} "
        f"({INTERVAL}), fast every {FAST_POLL_SECONDS}s, "
        f"full every {POLL_SECONDS}s, "
        f"Balance=${balance:.2f}, "
        f"Sentiment={sentiment_status}"
    )

    send_telegram_message(
        f"🚀 *البوت يعمل الآن (Full Stack)*\n"
        f"════════════════════\n"
        f"💰 الرصيد: `${balance:.2f}`\n"
        f"📊 العملات: `{', '.join(SYMBOLS)}`\n"
        f"⏱ الفترة: `{INTERVAL}`\n"
        f"⚡ فحص سريع: كل `{FAST_POLL_SECONDS // 60}` دقيقة\n"
        f"🔄 دورة كاملة: كل `{POLL_SECONDS // 60}` دقيقة\n"
        f"════════════════════\n"
        f"🧠 HMM: `{'مفعل' if USE_HMM_FILTER else 'معطل'}`\n"
        f"📉 GARCH: `{'مفعل' if USE_GARCH_FILTER else 'معطل'}`\n"
        f"📰 Sentiment: `{sentiment_status}`\n"
        f"🔗 Correlation: `مفعل`\n"
        f"🛡 FFC: `مفعل`\n"
        f"════════════════════\n"
        f"⚠️ _تداول وهمي — لا مخاطر مالية._"
    )

    last_full_cycle = time.time()
    last_report_time = time.time()

    while True:
        try:
            run_cycle_fast()

            if time.time() - last_full_cycle >= POLL_SECONDS:
                log.info("[MAIN] Starting full cycle (HMM + GARCH + Sentiment + Correlation + FFC)")
                run_cycle_full()
                last_full_cycle = time.time()

            if time.time() - last_report_time >= REPORT_INTERVAL_SECONDS:
                send_daily_report()
                last_report_time = time.time()

        except requests.RequestException as e:
            log.error(f"Network error this cycle: {e}")
        except Exception as e:
            log.exception(f"Unexpected error this cycle: {e}")

        time.sleep(FAST_POLL_SECONDS)


if __name__ == "__main__":
    main()
