"""
quant_alert.py
==============
Full-stack crypto analysis / paper-trading engine.

Layers:
1) Z-Score mean-reversion signal
2) HMM market-regime filter
3) GARCH volatility filter / position-size adjustment
4) Sentiment filter
5) Correlation / concentration filter
6) FFC (Fitness Feedback Control) as the LIVE execution gate

Important design rule:
- Paper trading continues even when FFC is HALTED. This is intentional:
  FFC needs ongoing Paper results to evaluate recovery and decide whether
  LIVE execution may resume.
- This file does NOT contain a real exchange-order executor. Therefore the
  default mode is PAPER. Setting TRADING_MODE=live is rejected unless a live
  executor is added explicitly.

Environment variables (main):
TG_BOT_TOKEN / TELEGRAM_BOT_TOKEN
TG_CHAT_ID   / TELEGRAM_CHAT_ID
SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT
INTERVAL=15m
LOOKBACK=20
CANDLE_LIMIT=200
POLL_SECONDS=1800
FAST_POLL_SECONDS=300
TRADING_MODE=paper
USE_HMM_FILTER=true
USE_GARCH_FILTER=true
USE_SENTIMENT_FILTER=true
HMM_INTERVAL=4h
HMM_LIMIT=300
HMM_STATES=2
GARCH_INTERVAL=4h
GARCH_LIMIT=500
Z_THRESHOLD=2.0
Z_STRONG_THRESHOLD=2.5
URGENT_ALERT_COOLDOWN_SECONDS=1800
REPORT_INTERVAL_SECONDS=86400

FFC variables are consumed by ffc.py, e.g.:
PHI_OFF_PCT, PHI_ON_PCT, LOOKBACK_TRADES, MIN_TRADES_TO_JUDGE,
RESUME_CONFIRM_TRADES, MAX_CONSECUTIVE_LOSSES, MAX_DRAWDOWN_PCT,
RESUME_COOLDOWN_TRADES.
"""

from __future__ import annotations

import html
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

from paper_trader import (
    check_and_close_position,
    get_stats,
    load_balance,
    open_paper_position,
)
from hmm_regime import (
    fetch_returns as hmm_fetch_returns,
    filter_signals_by_regime,
    get_current_regime,
)
from garch_model import (
    adjust_position_size_by_volatility,
    classify_volatility,
    fetch_returns as garch_fetch_returns,
    forecast_volatility,
)
from correlation_filter import get_correlation_filter
from ffc import get_ffc
from sentiment_filter import get_sentiment_filter


# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("quant_alert")


# ============================================================
# CONFIG HELPERS
# ============================================================
def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean for {name}: {raw!r}")


def _env_int(name: str, default: int, minimum: Optional[int] = None) -> int:
    raw = os.environ.get(name)
    try:
        value = int(raw) if raw is not None else default
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer for {name}: {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _env_float(name: str, default: float, minimum: Optional[float] = None) -> float:
    raw = os.environ.get(name)
    try:
        value = float(raw) if raw is not None else default
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid number for {name}: {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


# ============================================================
# CONFIG
# ============================================================
BOT_TOKEN = (
    os.environ.get("TG_BOT_TOKEN")
    or os.environ.get("TELEGRAM_BOT_TOKEN")
    or os.environ.get("BOT_TOKEN")
)
CHAT_ID = (
    os.environ.get("TG_CHAT_ID")
    or os.environ.get("TELEGRAM_CHAT_ID")
    or os.environ.get("CHAT_ID")
)

SYMBOLS = [
    s.strip().upper()
    for s in os.environ.get(
        "SYMBOLS", "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT"
    ).split(",")
    if s.strip()
]

INTERVAL = os.environ.get("INTERVAL", "15m").strip()
LOOKBACK = _env_int("LOOKBACK", 20, 5)
CANDLE_LIMIT = _env_int("CANDLE_LIMIT", 200, LOOKBACK + 2)

POLL_SECONDS = _env_int("POLL_SECONDS", 1800, 30)
FAST_POLL_SECONDS = _env_int("FAST_POLL_SECONDS", 300, 15)
REPORT_INTERVAL_SECONDS = _env_int("REPORT_INTERVAL_SECONDS", 86400, 60)
URGENT_ALERT_COOLDOWN_SECONDS = _env_int(
    "URGENT_ALERT_COOLDOWN_SECONDS", 1800, 60
)

TRADING_MODE = os.environ.get("TRADING_MODE", "paper").strip().lower()
if TRADING_MODE not in {"paper", "live"}:
    raise ValueError("TRADING_MODE must be 'paper' or 'live'")

HMM_INTERVAL = os.environ.get("HMM_INTERVAL", "4h").strip()
HMM_LIMIT = _env_int("HMM_LIMIT", 300, 100)
HMM_STATES = _env_int("HMM_STATES", 2, 2)
USE_HMM_FILTER = _env_bool("USE_HMM_FILTER", True)

GARCH_INTERVAL = os.environ.get("GARCH_INTERVAL", "4h").strip()
GARCH_LIMIT = _env_int("GARCH_LIMIT", 500, 100)
USE_GARCH_FILTER = _env_bool("USE_GARCH_FILTER", True)

USE_SENTIMENT_FILTER = _env_bool("USE_SENTIMENT_FILTER", True)

Z_THRESHOLD = _env_float("Z_THRESHOLD", 2.0, 0.1)
Z_STRONG_THRESHOLD = _env_float("Z_STRONG_THRESHOLD", 2.5, Z_THRESHOLD)

# Conservative safety policy: data-dependent filters fail CLOSED on errors.
FAIL_CLOSED_ON_FILTER_ERROR = _env_bool("FAIL_CLOSED_ON_FILTER_ERROR", True)

# If False, the bot can fetch/display reports while halted, but it will not
# create new Paper positions. Default True so FFC can actually learn recovery.
PAPER_CONTINUE_WHEN_FFC_HALTED = _env_bool(
    "PAPER_CONTINUE_WHEN_FFC_HALTED", True
)

BINANCE_DATA_URL = "https://data-api.binance.vision/api/v3/klines"

# Reuse HTTP connections.
HTTP = requests.Session()
HTTP.headers.update({"User-Agent": "quant-alert/3.0"})

correlation_filter = get_correlation_filter()
ffc = get_ffc()
sentiment_filter = get_sentiment_filter()

_in_buy_zone: Dict[str, bool] = {}
_last_urgent_alert_at: Dict[str, float] = {}


# ============================================================
# VALIDATION
# ============================================================
def require_config() -> None:
    missing = []
    if not BOT_TOKEN:
        missing.append("TG_BOT_TOKEN (or TELEGRAM_BOT_TOKEN)")
    if not CHAT_ID:
        missing.append("TG_CHAT_ID (or TELEGRAM_CHAT_ID)")

    if missing:
        raise RuntimeError(
            "Missing environment variables: " + ", ".join(missing)
        )

    if not SYMBOLS:
        raise RuntimeError("SYMBOLS is empty")

    if POLL_SECONDS < FAST_POLL_SECONDS:
        raise RuntimeError(
            "POLL_SECONDS must be >= FAST_POLL_SECONDS "
            "so the full cycle is not scheduled more frequently than the fast cycle."
        )

    if TRADING_MODE == "live":
        raise RuntimeError(
            "TRADING_MODE=live is intentionally disabled in this file because "
            "no real exchange-order executor is implemented. Use TRADING_MODE=paper."
        )


# ============================================================
# SMALL HELPERS
# ============================================================
def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _format_price(price: float) -> str:
    if price >= 1000:
        return f"{price:,.2f}"
    if price >= 1:
        return f"{price:.4f}"
    if price >= 0.01:
        return f"{price:.6f}"
    return f"{price:.10f}"


def _position_path(symbol: str) -> Path:
    return Path(f"paper_position_{symbol}.json")


def _load_position(symbol: str) -> Optional[Dict[str, Any]]:
    path = _position_path(symbol)
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("position JSON is not an object")
        return data
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        log.error("[%s] Invalid position file %s: %s", symbol, path, exc)
        return None


def _get_open_position_symbols() -> List[str]:
    """Return symbols with valid open Paper position files."""
    result: List[str] = []
    for symbol in SYMBOLS:
        if _load_position(symbol) is not None:
            result.append(symbol)
    return result


def _filter_result(
    name: str,
    allowed: bool,
    reason: str,
    **details: Any,
) -> Dict[str, Any]:
    return {
        "name": name,
        "allowed": bool(allowed),
        "reason": reason,
        "details": details,
    }


def _format_filter_summary(filters: List[Dict[str, Any]]) -> str:
    lines = []
    for item in filters:
        icon = "✅" if item["allowed"] else "❌"
        lines.append(f"{icon} {item['name']}: {item['reason']}")
    return "\n".join(lines)


# ============================================================
# MARKET DATA
# ============================================================
def fetch_candles(symbol: str, interval: str, limit: int) -> np.ndarray:
    """Fetch OHLCV candles from Binance Data API."""
    response = HTTP.get(
        BINANCE_DATA_URL,
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=15,
    )
    response.raise_for_status()
    raw = response.json()

    if not isinstance(raw, list) or not raw:
        raise ValueError(f"No candle data returned for {symbol} {interval}")

    rows = []
    for k in raw:
        if not isinstance(k, list) or len(k) < 6:
            continue
        try:
            rows.append(
                [float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])]
            )
        except (TypeError, ValueError):
            continue

    candles = np.asarray(rows, dtype=float)
    if len(candles) < LOOKBACK:
        raise ValueError(
            f"Insufficient candles for {symbol}: {len(candles)} < {LOOKBACK}"
        )
    if not np.isfinite(candles).all():
        raise ValueError(f"Non-finite candle data for {symbol}")
    if np.any(candles[:, 0] <= 0) or np.any(candles[:, 3] <= 0):
        raise ValueError(f"Invalid OHLC prices for {symbol}")
    return candles


# ============================================================
# INDICATORS
# ============================================================
def z_score(closes: np.ndarray, period: int) -> float:
    if len(closes) < period:
        raise ValueError("Not enough closes for z-score")
    window = closes[-period:]
    mean = float(window.mean())
    std = float(window.std(ddof=0))
    if std <= 0 or not math.isfinite(std):
        return 0.0
    return float((closes[-1] - mean) / std)


def z_series(closes: np.ndarray, period: int) -> np.ndarray:
    if len(closes) < period:
        return np.array([], dtype=float)
    values: List[float] = []
    for i in range(period, len(closes) + 1):
        window = closes[i - period:i]
        mean = float(window.mean())
        std = float(window.std(ddof=0))
        values.append(0.0 if std <= 0 else float((window[-1] - mean) / std))
    return np.asarray(values, dtype=float)


def garman_klass_volatility(candles: np.ndarray, period: int) -> float:
    if len(candles) < period:
        return 0.0
    window = candles[-period:]
    o, h, l, c = window[:, 0], window[:, 1], window[:, 2], window[:, 3]

    with np.errstate(divide="ignore", invalid="ignore"):
        log_hl = np.log(h / l)
        log_co = np.log(c / o)
        variance = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2

    variance = variance[np.isfinite(variance)]
    if len(variance) == 0:
        return 0.0
    return float(np.sqrt(max(float(variance.mean()), 0.0)) * 100.0)


def classify_signal(z: float) -> str:
    if z < -Z_THRESHOLD:
        return "BUY_STATISTICAL"
    if z > Z_THRESHOLD:
        return "SELL_STATISTICAL"
    return "NO_SIGNAL"


# ============================================================
# HMM
# ============================================================
def get_market_regime(symbol: str) -> Dict[str, Any]:
    if not USE_HMM_FILTER:
        return {
            "available": True,
            "current_regime": "DISABLED",
            "regime_prob": 1.0,
            "reason": "HMM disabled",
        }

    try:
        returns = hmm_fetch_returns(
            symbol, interval=HMM_INTERVAL, limit=HMM_LIMIT
        )
        if len(returns) < 100:
            raise ValueError(f"Only {len(returns)} HMM returns available")

        result = get_current_regime(returns, n_states=HMM_STATES)
        raw_regime = str(result.get("current_regime", "UNKNOWN"))
        probability = _safe_float(result.get("regime_prob"), 0.0)
        regime_key = raw_regime.strip().upper()
        if regime_key == "BULL":
            regime = "Bull"
        elif regime_key == "BEAR":
            regime = "Bear"
        else:
            regime = "UNKNOWN"

        log.info(
            "[HMM:%s] Regime=%s prob=%.2f%%",
            symbol,
            regime,
            probability * 100,
        )
        return {
            "available": True,
            "current_regime": regime,
            "regime_prob": probability,
            "reason": "HMM evaluated",
        }
    except Exception as exc:
        log.exception("[HMM:%s] Failed: %s", symbol, exc)
        return {
            "available": False,
            "current_regime": "UNKNOWN",
            "regime_prob": 0.0,
            "reason": f"HMM unavailable: {exc}",
        }


# ============================================================
# GARCH
# ============================================================
def get_volatility_info(symbol: str) -> Dict[str, Any]:
    if not USE_GARCH_FILTER:
        return {
            "available": True,
            "vol_ratio": 1.0,
            "vol_regime": "DISABLED",
            "annualized_vol": 0.0,
            "current_vol": 0.0,
            "forecast_vol": 0.0,
            "reason": "GARCH disabled",
        }

    try:
        returns = garch_fetch_returns(
            symbol, interval=GARCH_INTERVAL, limit=GARCH_LIMIT
        )
        if len(returns) < 100:
            raise ValueError(f"Only {len(returns)} GARCH returns available")

        forecast = forecast_volatility(returns, horizon=1)
        vol_ratio = _safe_float(forecast.get("vol_ratio"), float("nan"))
        annualized = _safe_float(forecast.get("annualized_vol"), 0.0)
        current = _safe_float(forecast.get("current_vol"), 0.0)
        forecast_value = _safe_float(forecast.get("forecast_vol"), 0.0)

        if not math.isfinite(vol_ratio):
            raise ValueError("GARCH returned a non-finite vol_ratio")

        regime = str(classify_volatility(vol_ratio)).upper()
        return {
            "available": True,
            "vol_ratio": vol_ratio,
            "vol_regime": regime,
            "annualized_vol": annualized,
            "current_vol": current,
            "forecast_vol": forecast_value,
            "reason": "GARCH evaluated",
        }
    except Exception as exc:
        log.exception("[GARCH:%s] Failed: %s", symbol, exc)
        return {
            "available": False,
            "vol_ratio": float("nan"),
            "vol_regime": "UNKNOWN",
            "annualized_vol": 0.0,
            "current_vol": 0.0,
            "forecast_vol": 0.0,
            "reason": f"GARCH unavailable: {exc}",
        }


# ============================================================
# TELEGRAM
# ============================================================
def _telegram_request(
    endpoint: str,
    *,
    data: Optional[Dict[str, Any]] = None,
    files: Optional[Dict[str, Any]] = None,
    timeout: int = 10,
) -> bool:
    if not BOT_TOKEN or not CHAT_ID:
        log.error("Telegram is not configured")
        return False

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{endpoint}"
    try:
        response = HTTP.post(
            url,
            data=data,
            files=files,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok", False):
            log.error("Telegram API rejected request: %s", payload)
            return False
        return True
    except (requests.RequestException, ValueError) as exc:
        body = ""
        try:
            body = response.text[:500]  # type: ignore[possibly-undefined]
        except Exception:
            pass
        log.error("Telegram %s failed: %s %s", endpoint, exc, body)
        return False


def send_telegram_message(text: str) -> bool:
    # HTML is safer than legacy Markdown for dynamic symbols/reasons.
    return _telegram_request(
        "sendMessage",
        data={
            "chat_id": CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": "true",
        },
        timeout=10,
    )


def send_telegram_photo(png_bytes: bytes, caption: str) -> bool:
    return _telegram_request(
        "sendPhoto",
        data={
            "chat_id": CHAT_ID,
            "caption": caption,
            "parse_mode": "HTML",
        },
        files={"photo": ("chart.png", png_bytes, "image/png")},
        timeout=30,
    )


def tg(value: Any) -> str:
    """HTML-escape dynamic Telegram content."""
    return html.escape(str(value), quote=False)


# ============================================================
# CHART
# ============================================================
def make_chart_png(
    symbol: str,
    candles: np.ndarray,
    period: int,
    entry_price: Optional[float] = None,
    regime: Optional[str] = None,
    vol_regime: Optional[str] = None,
) -> bytes:
    import io
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    o, h, l, c = candles[:, 0], candles[:, 1], candles[:, 2], candles[:, 3]
    n = len(c)
    zs = z_series(c, period)
    zx = np.arange(period - 1, n)
    ma = np.array(
        [c[i - period + 1:i + 1].mean() for i in range(period - 1, n)],
        dtype=float,
    )
    sd = np.array(
        [c[i - period + 1:i + 1].std() for i in range(period - 1, n)],
        dtype=float,
    )

    bg, fg = "#131722", "#d1d4dc"
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
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
        ax1.add_patch(
            Rectangle(
                (i - 0.35, min(o[i], c[i])),
                0.7,
                abs(c[i] - o[i]) or 1e-9,
                facecolor=color,
                edgecolor=color,
            )
        )

    ax1.plot(zx, ma, color="#f5c542", linewidth=1.2, label=f"Mean ({period})")
    ax1.fill_between(
        zx,
        ma - 2 * sd,
        ma + 2 * sd,
        color="#2962ff",
        alpha=0.08,
        label="±2σ band",
    )

    price_range = max(float(h.max() - l.min()), 1e-12)
    pad = price_range * 0.05
    ax1.set_ylim(l.min() - pad, h.max() + pad)

    buy_idx = zx[zs < -Z_THRESHOLD]
    if len(buy_idx):
        ax1.scatter(
            buy_idx,
            l[buy_idx] - pad * 0.5,
            marker="^",
            color="#26a69a",
            s=40,
            label=f"Z < -{Z_THRESHOLD:g}",
        )

    if entry_price is not None:
        ax1.axhline(
            entry_price,
            color="#00e676",
            linestyle="--",
            linewidth=1,
            label=f"Entry {_format_price(entry_price)}",
        )

    title = (
        f"{symbol}  {INTERVAL} | last {_format_price(c[-1])} | "
        f"Z={zs[-1]:.2f}"
    )
    if regime:
        title += f" | {regime}"
    if vol_regime:
        title += f" | Vol: {vol_regime}"
    ax1.set_title(title, color=fg, fontsize=11)

    ax2.plot(zx, zs, color="#2962ff", linewidth=1.2)
    ax2.axhline(-Z_THRESHOLD, color="#ef5350", linestyle="--", linewidth=0.8)
    ax2.axhline(Z_THRESHOLD, color="#ef5350", linestyle="--", linewidth=0.8)
    ax2.axhline(0, color="#787b86", linewidth=0.6)
    ax2.set_ylabel("Z-Score", color=fg, fontsize=9)
    ax2.set_xlabel(
        f"last {len(c)} candles ({INTERVAL})", color=fg, fontsize=9
    )
    ax1.legend(
        facecolor=bg,
        edgecolor="#2a2e39",
        labelcolor=fg,
        fontsize=8,
    )

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", facecolor=bg, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def send_chart(
    symbol: str,
    candles: np.ndarray,
    caption: str,
    entry_price: Optional[float] = None,
    regime: Optional[str] = None,
    vol_regime: Optional[str] = None,
) -> bool:
    try:
        png = make_chart_png(
            symbol,
            candles,
            LOOKBACK,
            entry_price=entry_price,
            regime=regime,
            vol_regime=vol_regime,
        )
    except Exception as exc:
        log.exception("Chart rendering failed for %s: %s", symbol, exc)
        return False
    return send_telegram_photo(png, caption)


# ============================================================
# FILTER STACK
# ============================================================
def evaluate_buy_filters(
    symbol: str,
    z: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Run the non-Z filters and return structured diagnostics."""
    filters: List[Dict[str, Any]] = []

    # Layer 1: Z-score
    z_allowed = z < -Z_THRESHOLD
    filters.append(
        _filter_result(
            "Z-Score",
            z_allowed,
            f"Z={z:.2f}",
        )
    )

    # Layer 2: HMM
    regime_info = get_market_regime(symbol)
    if not USE_HMM_FILTER:
        hmm_allowed = True
        hmm_reason = "disabled"
    elif not regime_info["available"]:
        hmm_allowed = not FAIL_CLOSED_ON_FILTER_ERROR
        hmm_reason = regime_info["reason"]
    else:
        current_regime = str(regime_info["current_regime"]).upper()
        hmm_signal = filter_signals_by_regime(1, current_regime)
        hmm_allowed = hmm_signal == 1
        hmm_reason = f"regime={current_regime}"
    filters.append(_filter_result("HMM", hmm_allowed, hmm_reason))

    # Layer 3: GARCH
    vol_info = get_volatility_info(symbol)
    if not USE_GARCH_FILTER:
        garch_allowed = True
        garch_reason = "disabled"
    elif not vol_info["available"]:
        garch_allowed = not FAIL_CLOSED_ON_FILTER_ERROR
        garch_reason = vol_info["reason"]
    else:
        # User's rule: HIGH volatility blocks entry.
        garch_allowed = str(vol_info["vol_regime"]).upper() != "HIGH"
        garch_reason = (
            f"regime={vol_info['vol_regime']} "
            f"ratio={vol_info['vol_ratio']:.2f}"
        )
    filters.append(_filter_result("GARCH", garch_allowed, garch_reason))

    # Layer 4: Sentiment
    sentiment_result: Dict[str, Any] = {
        "allowed": True,
        "multiplier": 1.0,
        "sentiment": {"label": "disabled", "score": 0.0, "count": 0},
        "reason": "disabled",
    }
    if USE_SENTIMENT_FILTER:
        try:
            sentiment_result = sentiment_filter.filter_signal("BUY", symbol)
            if not isinstance(sentiment_result, dict):
                raise ValueError("Sentiment filter returned non-dict result")
            sentiment_allowed = bool(sentiment_result.get("allowed", False))
            sentiment_reason = str(
                sentiment_result.get("reason", "no reason returned")
            )
        except Exception as exc:
            log.exception("[SENTIMENT:%s] Failed: %s", symbol, exc)
            sentiment_result = {
                "allowed": not FAIL_CLOSED_ON_FILTER_ERROR,
                "multiplier": 1.0,
                "sentiment": {
                    "label": "unknown",
                    "score": 0.0,
                    "count": 0,
                },
                "reason": f"sentiment unavailable: {exc}",
            }
            sentiment_allowed = bool(sentiment_result["allowed"])
            sentiment_reason = sentiment_result["reason"]
    else:
        sentiment_allowed = True
        sentiment_reason = "disabled"

    filters.append(
        _filter_result("Sentiment", sentiment_allowed, sentiment_reason)
    )

    # Layer 5: Correlation / concentration
    try:
        open_symbols = _get_open_position_symbols()
        correlation_allowed, correlation_reason = correlation_filter.can_open_position(
            symbol, open_symbols
        )
    except Exception as exc:
        log.exception("[CORRELATION:%s] Failed: %s", symbol, exc)
        correlation_allowed = not FAIL_CLOSED_ON_FILTER_ERROR
        correlation_reason = f"correlation unavailable: {exc}"
    filters.append(
        _filter_result("Correlation", correlation_allowed, str(correlation_reason))
    )

    return filters, regime_info, vol_info, sentiment_result


# ============================================================
# EXECUTION / PAPER ENGINE
# ============================================================
def _save_position(symbol: str, position: Dict[str, Any]) -> None:
    path = _position_path(symbol)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(position, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, path)


def run_execution_step(
    symbol: str,
    price: float,
    z: float,
    candles: np.ndarray,
) -> None:
    """
    Run one symbol through the Paper Engine.

    Crucial FFC behavior:
    - Existing Paper positions are always monitored and may be closed.
    - New Paper positions may continue while FFC is HALTED if configured.
    - FFC is the LIVE gate, not a kill-switch for Paper data collection.
    """
    position = _load_position(symbol)

    # Always manage an existing Paper position first.
    if position is not None:
        try:
            trade = check_and_close_position(symbol, price)
        except Exception as exc:
            log.exception("[EXIT:%s] Failed to evaluate position: %s", symbol, exc)
            return

        if not trade:
            return

        try:
            balance = load_balance()
            ffc_state = ffc.update_after_trade(
                trade["pnl_usd"],
                balance,
                source="paper",
            )
        except Exception as exc:
            log.exception("[FFC:%s] Failed after Paper trade: %s", symbol, exc)
            return

        current_regime = "N/A"
        vol_regime = "N/A"
        emoji = "🟢" if trade["pnl_usd"] > 0 else "🔴"
        caption = (
            f"<b>PAPER EXIT {tg(symbol)}</b> ({tg(trade['exit_reason'])})\n"
            f"entry {tg(_format_price(trade['entry_price']))} → "
            f"exit {tg(_format_price(trade['exit_price']))}\n"
            f"P&amp;L {trade['pnl_pct']:+.3f}% (${trade['pnl_usd']:+.2f})\n"
            f"Balance: ${balance:.2f}\n"
            f"FFC: {'LIVE' if ffc_state['is_live'] else 'HALTED'}"
        )
        send_chart(
            symbol,
            candles,
            caption,
            entry_price=trade["entry_price"],
            regime=current_regime,
            vol_regime=vol_regime,
        )
        send_telegram_message(
            f"{emoji} <b>إغلاق صفقة Paper — {tg(symbol)}</b>\n"
            f"السبب: <code>{tg(trade['exit_reason'])}</code>\n"
            f"دخول: <code>{tg(_format_price(trade['entry_price']))}</code>\n"
            f"خروج: <code>{tg(_format_price(trade['exit_price']))}</code>\n"
            f"النتيجة: <code>{trade['pnl_pct']:+.3f}% (${trade['pnl_usd']:+.2f})</code>\n"
            f"الرصيد: <code>${balance:.2f}</code>\n"
            f"FFC: <b>{'نشط' if ffc_state['is_live'] else 'متوقف'}</b>"
        )
        return

    # No open position: check the statistical trigger first.
    if z >= -Z_THRESHOLD:
        return

    # Evaluate the complete filter stack.
    filters, regime_info, vol_info, sentiment_result = evaluate_buy_filters(symbol, z)

    # FFC is a LIVE gate. For this Paper-only implementation, it is never
    # used to prevent the Paper data stream unless the user explicitly opts in.
    ffc_live_allowed = ffc.can_open_position("live")
    # Paper mode is normally allowed. When explicitly disabled, it follows
    # the FFC state so recovery can be paused intentionally.
    ffc_paper_allowed = (
        True
        if PAPER_CONTINUE_WHEN_FFC_HALTED
        else ffc.can_open_position("live")
    )

    all_market_filters_allowed = all(item["allowed"] for item in filters)

    if not ffc_paper_allowed:
        log.info("[EXEC:%s] Paper opening disabled while FFC halted", symbol)
        return

    if not all_market_filters_allowed:
        summary = _format_filter_summary(filters)
        log.info("[EXEC:%s] BUY BLOCKED\n%s", symbol, summary)
        send_telegram_message(
            f"⚠️ <b>إشارة شراء مرفوضة — {tg(symbol)}</b>\n"
            f"Z-Score: <code>{z:.2f}</code>\n"
            f"<pre>{tg(summary)}</pre>\n"
            f"FFC LIVE: <b>{'مسموح' if ffc_live_allowed else 'متوقف'}</b>"
        )
        return

    # ========================================================
    # Paper entry
    # ========================================================
    try:
        new_pos = open_paper_position(symbol, price, z)
    except Exception as exc:
        log.exception("[ENTRY:%s] open_paper_position failed: %s", symbol, exc)
        return

    if not new_pos:
        log.warning("[ENTRY:%s] Paper engine returned no position", symbol)
        return

    try:
        original_size = float(new_pos["position_size_usd"])
        vol_ratio = _safe_float(vol_info.get("vol_ratio"), 1.0)
        garch_size = adjust_position_size_by_volatility(original_size, vol_ratio)
        sentiment_multiplier = _safe_float(
            sentiment_result.get("multiplier", 1.0), 1.0
        )
        if sentiment_multiplier < 0:
            sentiment_multiplier = 0.0

        final_size = max(0.0, float(garch_size) * sentiment_multiplier)
        if final_size <= 0:
            log.warning("[ENTRY:%s] Final position size is zero", symbol)
            return

        new_pos["quantity"] = final_size / price
        new_pos["position_size_usd"] = final_size
        _save_position(symbol, new_pos)
    except Exception as exc:
        log.exception("[ENTRY:%s] Position sizing/save failed: %s", symbol, exc)
        return

    balance = load_balance()
    sent_data = sentiment_result.get("sentiment", {})
    sent_label = sent_data.get("label", "unknown")
    sent_score = _safe_float(sent_data.get("score"), 0.0)
    sent_count = int(_safe_float(sent_data.get("count"), 0))
    current_regime = regime_info.get("current_regime", "UNKNOWN")
    vol_regime = vol_info.get("vol_regime", "UNKNOWN")

    send_chart(
        symbol,
        candles,
        f"<b>PAPER ENTRY {tg(symbol)}</b>\n"
        f"price {_format_price(new_pos['entry_price'])} | Z {z:.2f}\n"
        f"qty {new_pos['quantity']:.8f} | size ${final_size:.2f}\n"
        f"Sentiment: {tg(sent_label)} ({sent_score:+.2f}, {sent_count} news)\n"
        f"Balance: ${balance:.2f} | {tg(current_regime)} | {tg(vol_regime)}",
        entry_price=new_pos["entry_price"],
        regime=current_regime,
        vol_regime=vol_regime,
    )

    send_telegram_message(
        f"🟢 <b>فتح صفقة Paper — {tg(symbol)}</b>\n"
        f"السعر: <code>{tg(_format_price(new_pos['entry_price']))}</code> | "
        f"Z: <code>{z:.2f}</code>\n"
        f"الكمية: <code>{new_pos['quantity']:.8f}</code>\n"
        f"حجم الصفقة: <code>${final_size:.2f}</code>\n"
        f"GARCH × Sentiment: <code>×{_safe_float(vol_info.get('vol_ratio'), 1.0):.2f} / "
        f"×{sentiment_multiplier:.2f}</code>\n"
        f"النظام: <b>{tg(current_regime)}</b> | التقلب: <b>{tg(vol_regime)}</b>\n"
        f"الأخبار: {tg(sent_label)} ({sent_score:+.2f}, {sent_count} items)\n"
        f"FFC LIVE gate: <b>{'مسموح' if ffc_live_allowed else 'متوقف'}</b>\n"
        f"الرصيد: <code>${balance:.2f}</code>"
    )


# ============================================================
# ALERTS
# ============================================================
def check_urgent_signals(
    per_symbol_data: List[Tuple[str, float, float, np.ndarray]]
) -> None:
    now = time.monotonic()

    for symbol, price, z, _candles in per_symbol_data:
        if z < -Z_STRONG_THRESHOLD:
            last = _last_urgent_alert_at.get(symbol, 0.0)
            if now - last >= URGENT_ALERT_COOLDOWN_SECONDS:
                log.info("[URGENT:%s] strong statistical signal z=%.2f", symbol, z)
                send_telegram_message(
                    f"🚨 <b>إشارة إحصائية قوية — {tg(symbol)}</b>\n"
                    f"Z-Score: <b>{z:.2f}</b>\n"
                    f"السعر: <code>{tg(_format_price(price))}</code>\n"
                    f"النوع: <b>Z-Score mean-reversion</b>\n"
                    f"⚠️ هذه <b>إشارة أولية</b> وليست موافقة نهائية؛ يجب أن تمر عبر HMM/GARCH/Sentiment/Correlation/FFC."
                )
                _last_urgent_alert_at[symbol] = now

        now_in_zone = z < -Z_THRESHOLD
        was_in_zone = _in_buy_zone.get(symbol, False)

        if now_in_zone and not was_in_zone:
            send_telegram_message(
                f"🟢 <b>{tg(symbol)}</b> دخل منطقة Z-Score السلبية\n"
                f"Z-Score: <code>{z:.2f}</code>\n"
                f"السعر: <code>{tg(_format_price(price))}</code>\n"
                f"ℹ️ إشارة إحصائية أولية — ليست نصيحة استثمارية."
            )

        _in_buy_zone[symbol] = now_in_zone


# ============================================================
# CYCLES
# ============================================================
def run_cycle_fast() -> None:
    """Fast cycle: price + Z-score + early alerts."""
    per_symbol_data: List[Tuple[str, float, float, np.ndarray]] = []

    for symbol in SYMBOLS:
        try:
            candles = fetch_candles(symbol, INTERVAL, CANDLE_LIMIT)
            closes = candles[:, 3]
            price = float(closes[-1])
            z = z_score(closes, LOOKBACK)
            per_symbol_data.append((symbol, price, z, candles))
            log.info("[FAST] %s price=%s z=%.2f", symbol, _format_price(price), z)
        except (requests.RequestException, ValueError) as exc:
            log.error("[FAST] Failed to fetch %s: %s", symbol, exc)

    if per_symbol_data:
        check_urgent_signals(per_symbol_data)


def run_cycle_full() -> None:
    """Full cycle: market data + correlation + news + all six layers."""
    per_symbol_data: List[Tuple[str, float, float, np.ndarray]] = []

    # Fetch each symbol ONCE. The old version fetched the same 15m candles
    # twice in this cycle, unnecessarily doubling Binance requests.
    for symbol in SYMBOLS:
        try:
            candles = fetch_candles(symbol, INTERVAL, CANDLE_LIMIT)
            closes = candles[:, 3]
            price = float(closes[-1])
            z = z_score(closes, LOOKBACK)
            per_symbol_data.append((symbol, price, z, candles))

            try:
                correlation_filter.update_prices(symbol, closes)
            except Exception as exc:
                log.warning(
                    "[FULL] Correlation update failed for %s: %s", symbol, exc
                )

            log.info(
                "[FULL] %s price=%s z=%.2f GK-vol=%.3f%%",
                symbol,
                _format_price(price),
                z,
                garman_klass_volatility(candles, LOOKBACK),
            )
        except (requests.RequestException, ValueError) as exc:
            log.error("[FULL] Failed to fetch %s: %s", symbol, exc)

    # Refresh news cache before executing the signal stack.
    if USE_SENTIMENT_FILTER:
        try:
            from news_fetcher import update_news_cache

            bases = [s.replace("USDT", "") for s in SYMBOLS]
            update_news_cache(symbols=bases)
            log.info("[FULL] News cache updated")
        except Exception as exc:
            log.warning("[FULL] News cache update failed: %s", exc)
            if FAIL_CLOSED_ON_FILTER_ERROR:
                log.warning(
                    "[FULL] Sentiment is configured fail-closed; entries may be blocked."
                )

    for symbol, price, z, candles in per_symbol_data:
        run_execution_step(symbol, price, z, candles)


# ============================================================
# REPORTING
# ============================================================
def send_daily_report() -> None:
    try:
        stats = get_stats()
        if not stats:
            return

        ffc_info = ffc.status()
        ffc_emoji = "🟢" if ffc_info["is_live"] else "🔴"

        sent_lines: List[str] = []
        if USE_SENTIMENT_FILTER:
            for symbol in SYMBOLS:
                try:
                    s = sentiment_filter.get_sentiment(symbol)
                    label = s.get("label", "unknown")
                    if label == "bullish":
                        emoji = "🟢"
                    elif label == "bearish":
                        emoji = "🔴"
                    else:
                        emoji = "⚪"
                    sent_lines.append(
                        f"{emoji} {tg(symbol)}: {tg(label)} "
                        f"({s.get('score', 0.0):+.2f}, {s.get('count', 0)} خبر)"
                    )
                except Exception as exc:
                    sent_lines.append(
                        f"⚪ {tg(symbol)}: unknown ({tg(exc)})"
                    )
            sent_section = (
                "════════════════════\n"
                "📰 <b>حالة الأخبار</b>\n"
                + "\n".join(sent_lines)
                + "\n"
            )
        else:
            sent_section = "📰 <b>Sentiment:</b> <code>معطل</code>\n"

        message = (
            "📊 <b>التقرير اليومي للأداء</b>\n"
            "════════════════════\n"
            f"💰 الرصيد: <code>${_safe_float(stats.get('balance')):.2f}</code> "
            f"(من <code>${_safe_float(stats.get('initial_balance')):.2f}</code>)\n"
            f"📈 العائد: <code>{_safe_float(stats.get('total_return_pct')):+.3f}%</code>\n"
            f"💵 صافي الربح: <code>${_safe_float(stats.get('net_pnl_usd')):+.4f}</code>\n"
            f"📊 عدد الصفقات: <code>{int(_safe_float(stats.get('total_trades')))}</code>\n"
            f"🎯 نسبة الرابحة: <code>{_safe_float(stats.get('win_rate')):.2%}</code>\n"
            f"✅ رابحة: <code>{int(_safe_float(stats.get('winning_trades')))}</code>\n"
            f"❌ خاسرة: <code>{int(_safe_float(stats.get('losing_trades')))}</code>\n"
            f"📈 متوسط الربح: <code>${_safe_float(stats.get('avg_win_usd')):.4f}</code>\n"
            f"📉 متوسط الخسارة: <code>${_safe_float(stats.get('avg_loss_usd')):.4f}</code>\n"
            "════════════════════\n"
            f"{ffc_emoji} <b>FFC — حماية رأس المال</b>\n"
            f"الحالة: <b>{'LIVE GATE' if ffc_info['is_live'] else 'HALTED'}</b>\n"
            f"Fitness: <code>{_safe_float(ffc_info.get('fitness')):+.2f}%</code>\n"
            f"Drawdown: <code>{_safe_float(ffc_info.get('drawdown_pct')):.2f}%</code>\n"
            f"Win rate: <code>{_safe_float(ffc_info.get('win_rate_pct')):.1f}%</code>\n"
            f"Loss streak: <code>{int(_safe_float(ffc_info.get('consecutive_losses')))}</code>\n"
            f"Halt count: <code>{int(_safe_float(ffc_info.get('total_halts')))}</code>\n"
            f"Resume count: <code>{int(_safe_float(ffc_info.get('total_resumes')))}</code>\n"
            + sent_section
            + "════════════════════\n"
            "⚠️ <i>تقرير آلي — ليس نصيحة استثمارية.</i>"
        )

        send_telegram_message(message)
        log.info("Daily report sent successfully")
    except Exception as exc:
        log.exception("Failed to send daily report: %s", exc)


# ============================================================
# STARTUP MESSAGE
# ============================================================
def send_startup_message(balance: float) -> None:
    ffc_info = ffc.status()
    send_telegram_message(
        "🚀 <b>البوت يعمل الآن — Full Stack v3</b>\n"
        "════════════════════\n"
        f"💰 الرصيد: <code>${balance:.2f}</code>\n"
        f"📊 العملات: <code>{tg(', '.join(SYMBOLS))}</code>\n"
        f"⏱ الفترة: <code>{tg(INTERVAL)}</code>\n"
        f"⚡ فحص سريع: كل <code>{FAST_POLL_SECONDS // 60}</code> دقيقة\n"
        f"🔄 دورة كاملة: كل <code>{POLL_SECONDS // 60}</code> دقيقة\n"
        "════════════════════\n"
        f"🧠 HMM: <code>{'مفعل' if USE_HMM_FILTER else 'معطل'}</code>\n"
        f"📉 GARCH: <code>{'مفعل' if USE_GARCH_FILTER else 'معطل'}</code>\n"
        f"📰 Sentiment: <code>{'مفعل' if USE_SENTIMENT_FILTER else 'معطل'}</code>\n"
        "🔗 Correlation: <code>مفعل</code>\n"
        "🛡 FFC: <code>LIVE gate</code>\n"
        f"🧪 Paper engine أثناء FFC HALT: <code>{'مستمر' if PAPER_CONTINUE_WHEN_FFC_HALTED else 'متوقف'}</code>\n"
        f"🟢 FFC state: <code>{'LIVE' if ffc_info['is_live'] else 'HALTED'}</code>\n"
        "════════════════════\n"
        "⚠️ <i>الوضع الحالي: تداول وهمي Paper فقط.</i>"
    )


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    require_config()
    balance = float(load_balance())

    log.info(
        "Starting quant_alert v3 | symbols=%s interval=%s fast=%ss full=%ss "
        "mode=%s HMM=%s GARCH=%s Sentiment=%s FFC=%s",
        ",".join(SYMBOLS),
        INTERVAL,
        FAST_POLL_SECONDS,
        POLL_SECONDS,
        TRADING_MODE,
        USE_HMM_FILTER,
        USE_GARCH_FILTER,
        USE_SENTIMENT_FILTER,
        ffc.status().get("is_live"),
    )

    send_startup_message(balance)

    # Run the expensive full stack immediately at startup instead of waiting
    # the first 30 minutes.
    try:
        run_cycle_full()
    except Exception as exc:
        log.exception("Initial full cycle failed: %s", exc)

    last_full_cycle = time.monotonic()
    last_report_time = time.monotonic()

    while True:
        cycle_started = time.monotonic()
        try:
            run_cycle_fast()

            now = time.monotonic()
            if now - last_full_cycle >= POLL_SECONDS:
                log.info("[MAIN] Starting full cycle")
                run_cycle_full()
                last_full_cycle = now

            if now - last_report_time >= REPORT_INTERVAL_SECONDS:
                send_daily_report()
                last_report_time = now

        except requests.RequestException as exc:
            log.error("Network error in main cycle: %s", exc)
        except Exception as exc:
            log.exception("Unexpected error in main cycle: %s", exc)

        elapsed = time.monotonic() - cycle_started
        sleep_for = max(1.0, FAST_POLL_SECONDS - elapsed)
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
