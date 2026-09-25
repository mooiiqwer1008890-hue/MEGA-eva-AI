"""
quant_alert.py
==============

BTCUSDT Statistical + Market Microstructure Research Engine

Purpose
-------
This program does NOT execute trades.

It collects public Binance Spot market data and constructs a
multi-factor statistical market report for BTCUSDT.

Main layers
-----------
1. OHLCV market data
2. Log returns
3. Classical and robust Z-scores
4. Momentum / trend structure
5. Efficiency ratio
6. Garman-Klass volatility
7. Parkinson volatility
8. Close-to-close volatility
9. ATR
10. Relative volume
11. Taker-buy ratio
12. Rolling VWAP
13. Order-book spread
14. Order-book depth imbalance
15. Multi-timeframe confirmation
16. Market-regime classification
17. Evidence score
18. Contradiction / risk detection
19. Telegram reporting

Important
---------
The score is NOT a probability of profit.

The system is an analytical research engine.
It does not guarantee direction, target price, or future return.
"""

import logging
import math
import os
import time
from html import escape
from typing import Dict, Tuple

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# 1. CONFIGURATION
# ============================================================

SYMBOL = os.getenv("SYMBOL", "BTCUSDT").upper()

# Main analytical timeframe.
PRIMARY_INTERVAL = os.getenv("INTERVAL", "15m")

# Statistical lookback.
LOOKBACK = int(os.getenv("LOOKBACK", "50"))

# Number of candles requested from Binance.
CANDLE_LIMIT = int(os.getenv("CANDLE_LIMIT", "300"))

# How often the complete analysis is repeated.
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "300"))

# HTTP timeout.
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "10"))

# Order-book depth.
DEPTH_LIMIT = int(os.getenv("DEPTH_LIMIT", "20"))

# Telegram credentials are environment variables.
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Binance public market-data endpoint.
# No Binance API key is required for these public endpoints.
BINANCE_BASE_URL = os.getenv(
    "BINANCE_BASE_URL",
    "https://data-api.binance.vision"
)

TELEGRAM_API_URL = "https://api.telegram.org"


# ============================================================
# 2. LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

log = logging.getLogger("quant_alert")


# ============================================================
# 3. INTERVAL DEFINITIONS
# ============================================================

INTERVAL_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
}


# ============================================================
# 4. HTTP SESSION
# ============================================================

def make_session() -> requests.Session:
    """
    Creates a persistent HTTP session.

    Retries temporary server/network errors rather than
    immediately killing the analysis cycle.
    """

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )

    session = requests.Session()

    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=retry
        )
    )

    session.headers.update(
        {
            "User-Agent":
                "BTC-Statistical-Research-Bot/1.0"
        }
    )

    return session


SESSION = make_session()


# ============================================================
# 5. CONFIGURATION VALIDATION
# ============================================================

def require_config() -> None:
    """
    Validate configuration before the main loop starts.
    """

    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError(
            "Missing TELEGRAM_BOT_TOKEN environment variable."
        )

    if not TELEGRAM_CHAT_ID:
        raise RuntimeError(
            "Missing TELEGRAM_CHAT_ID environment variable."
        )

    if PRIMARY_INTERVAL not in INTERVAL_SECONDS:
        raise ValueError(
            f"Unsupported INTERVAL: {PRIMARY_INTERVAL}"
        )

    if LOOKBACK < 20:
        raise ValueError(
            "LOOKBACK must be >= 20."
        )

    if CANDLE_LIMIT < LOOKBACK + 30:
        raise ValueError(
            "CANDLE_LIMIT must be larger than LOOKBACK by at least 30."
        )

    if DEPTH_LIMIT not in (5, 10, 20, 50, 100):
        log.warning(
            "DEPTH_LIMIT=%s is unusual. Binance may reject unsupported values.",
            DEPTH_LIMIT
        )


# ============================================================
# 6. GENERIC BINANCE REQUEST
# ============================================================

def get_json(
    path: str,
    params: Dict = None
):
    """
    GET request against Binance public market-data API.
    """

    url = f"{BINANCE_BASE_URL}{path}"

    response = SESSION.get(
        url,
        params=params,
        timeout=HTTP_TIMEOUT
    )

    if response.status_code == 429:
        raise RuntimeError(
            "Binance rate limit reached (HTTP 429)."
        )

    response.raise_for_status()

    return response.json()


# ============================================================
# 7. FETCH CANDLE DATA
# ============================================================

def fetch_klines(
    interval: str,
    limit: int = CANDLE_LIMIT
) -> np.ndarray:
    """
    Download Binance Spot kline data.

    Returned columns:

    0 = open time
    1 = open
    2 = high
    3 = low
    4 = close
    5 = base volume
    6 = quote volume
    7 = number of trades
    8 = taker-buy base volume
    9 = taker-buy quote volume

    The currently-forming candle is removed to avoid
    partial-candle bias.
    """

    raw = get_json(
        "/api/v3/klines",
        {
            "symbol": SYMBOL,
            "interval": interval,
            "limit": limit
        }
    )

    if not isinstance(raw, list):
        raise RuntimeError(
            f"Invalid kline response for {interval}."
        )

    if len(raw) < LOOKBACK + 5:
        raise RuntimeError(
            f"Insufficient data for {interval}: {len(raw)} candles."
        )

    rows = []

    for candle in raw:

        rows.append(
            [
                float(candle[0]),   # open time
                float(candle[1]),   # open
                float(candle[2]),   # high
                float(candle[3]),   # low
                float(candle[4]),   # close
                float(candle[5]),   # volume
                float(candle[7]),   # quote volume
                float(candle[8]),   # trades
                float(candle[9]),   # taker buy base
                float(candle[10]),  # taker buy quote
            ]
        )

    data = np.asarray(
        rows,
        dtype=float
    )

    # Remove current incomplete candle.
    if len(data) > LOOKBACK + 5:
        data = data[:-1]

    return data


# ============================================================
# 8. ORDER BOOK
# ============================================================

def fetch_order_book():
    """
    Fetch Binance Spot order book.

    Returns:
        bids: [[price, quantity], ...]
        asks: [[price, quantity], ...]
    """

    data = get_json(
        "/api/v3/depth",
        {
            "symbol": SYMBOL,
            "limit": DEPTH_LIMIT
        }
    )

    bids = np.asarray(
        [
            [
                float(price),
                float(quantity)
            ]
            for price, quantity in data["bids"]
        ],
        dtype=float
    )

    asks = np.asarray(
        [
            [
                float(price),
                float(quantity)
            ]
            for price, quantity in data["asks"]
        ],
        dtype=float
    )

    if len(bids) == 0 or len(asks) == 0:
        raise RuntimeError(
            "Order book is empty."
        )

    return bids, asks


# ============================================================
# 9. BASIC STATISTICS
# ============================================================

def safe_mean(x):
    x = np.asarray(x, dtype=float)

    if len(x) == 0:
        return float("nan")

    return float(np.mean(x))


def safe_std(
    x,
    ddof: int = 1
):
    x = np.asarray(x, dtype=float)

    if len(x) <= ddof:
        return float("nan")

    return float(
        np.std(
            x,
            ddof=ddof
        )
    )


# ============================================================
# 10. CLASSICAL Z-SCORE
# ============================================================

def zscore_last(x) -> float:
    """
    Classical standardized distance:

        Z = (x - mean) / standard deviation
    """

    x = np.asarray(
        x,
        dtype=float
    )

    if len(x) < 2:
        return 0.0

    mean = safe_mean(x)
    std = safe_std(x)

    if not np.isfinite(std) or std == 0:
        return 0.0

    return float(
        (x[-1] - mean) / std
    )


# ============================================================
# 11. ROBUST Z-SCORE
# ============================================================

def robust_zscore_last(x) -> float:
    """
    Robust standardized distance based on:

        median
        MAD = median absolute deviation

    Scale factor 1.4826 makes MAD comparable to standard
    deviation under a Gaussian reference distribution.

    Useful when extreme observations distort the ordinary mean
    and standard deviation.
    """

    x = np.asarray(
        x,
        dtype=float
    )

    if len(x) < 3:
        return 0.0

    median = np.median(x)

    mad = np.median(
        np.abs(x - median)
    )

    if mad <= 1e-12:
        return 0.0

    return float(
        (x[-1] - median)
        / (1.4826 * mad)
    )


# ============================================================
# 12. LOG RETURNS
# ============================================================

def log_returns(closes):
    """
    Logarithmic return:

        r_t = ln(P_t / P_{t-1})
    """

    closes = np.asarray(
        closes,
        dtype=float
    )

    return np.diff(
        np.log(closes)
    )


# ============================================================
# 13. MULTI-HORIZON RETURNS
# ============================================================

def rolling_return(
    closes,
    n: int
) -> float:

    closes = np.asarray(
        closes,
        dtype=float
    )

    if len(closes) <= n:
        return 0.0

    return float(
        closes[-1] /
        closes[-1 - n]
        - 1.0
    )


# ============================================================
# 14. NORMALIZED LINEAR TREND
# ============================================================

def linear_slope_normalized(
    closes,
    n: int
) -> float:
    """
    Linear regression slope normalized by average price.

    This is a descriptive trend-strength measure.
    """

    y = np.asarray(
        closes[-n:],
        dtype=float
    )

    if len(y) < 3:
        return 0.0

    x = np.arange(
        len(y),
        dtype=float
    )

    slope = np.polyfit(
        x,
        y,
        1
    )[0]

    normalized = (
        slope /
        max(abs(np.mean(y)), 1e-12)
        * 100.0
    )

    return float(normalized)


# ============================================================
# 15. EFFICIENCY RATIO
# ============================================================

def efficiency_ratio(
    closes,
    n: int = 20
) -> float:
    """
    Kaufman-style efficiency ratio:

        net movement / total path movement

    Near 1:
        directional movement

    Near 0:
        noisy / oscillatory movement
    """

    closes = np.asarray(
        closes,
        dtype=float
    )

    if len(closes) <= n:
        return 0.0

    direction = abs(
        closes[-1] -
        closes[-1 - n]
    )

    path = np.sum(
        np.abs(
            np.diff(
                closes[-1 - n:]
            )
        )
    )

    if path <= 0:
        return 0.0

    return float(
        direction / path
    )


# ============================================================
# 16. RSI
# ============================================================

def rsi(
    closes,
    n: int = 14
) -> float:

    returns = np.diff(
        np.asarray(
            closes,
            dtype=float
        )
    )

    if len(returns) < n:
        return 50.0

    gains = np.maximum(
        returns,
        0
    )

    losses = np.maximum(
        -returns,
        0
    )

    avg_gain = np.mean(
        gains[-n:]
    )

    avg_loss = np.mean(
        losses[-n:]
    )

    if avg_loss == 0:

        if avg_gain > 0:
            return 100.0

        return 50.0

    rs = avg_gain / avg_loss

    return float(
        100 -
        100 / (1 + rs)
    )


# ============================================================
# 17. PARKINSON VOLATILITY
# ============================================================

def parkinson_vol(
    ohlc,
    n: int
) -> float:
    """
    Parkinson range-based volatility estimator.

        sigma^2 =
        mean[
            ln(H/L)^2 / (4 ln 2)
        ]

    Output is percentage volatility per candle window.
    """

    data = ohlc[-n:]

    high = data[:, 2]
    low = data[:, 3]

    variance = (
        np.log(high / low) ** 2
    ) / (
        4.0 * math.log(2.0)
    )

    return float(
        math.sqrt(
            max(
                np.mean(variance),
                0.0
            )
        ) * 100
    )


# ============================================================
# 18. GARMAN-KLASS VOLATILITY
# ============================================================

def garman_klass_vol(
    ohlc,
    n: int
) -> float:
    """
    Garman-Klass OHLC volatility estimator.

        0.5 * ln(H/L)^2
        -
        (2 ln 2 - 1) * ln(C/O)^2

    Output is percentage volatility per candle window.
    """

    data = ohlc[-n:]

    open_price = data[:, 1]
    high = data[:, 2]
    low = data[:, 3]
    close = data[:, 4]

    variance = (
        0.5 *
        np.log(high / low) ** 2
        -
        (
            2 * math.log(2) - 1
        ) *
        np.log(close / open_price) ** 2
    )

    variance = np.maximum(
        variance,
        0.0
    )

    return float(
        math.sqrt(
            max(
                np.mean(variance),
                0.0
            )
        ) * 100
    )


# ============================================================
# 19. CLOSE-TO-CLOSE VOLATILITY
# ============================================================

def close_vol(
    ohlc,
    n: int
) -> float:

    closes = ohlc[
        -(n + 1):,
        4
    ]

    returns = log_returns(
        closes
    )

    if len(returns) <= 1:
        return 0.0

    return float(
        np.std(
            returns,
            ddof=1
        ) * 100
    )


# ============================================================
# 20. RELATIVE VOLUME
# ============================================================

def relative_volume(
    ohlc,
    n: int = 30
) -> float:

    volume = ohlc[:, 5]

    if len(volume) <= n:
        return 1.0

    historical = volume[
        -n - 1:-1
    ]

    mean_volume = np.mean(
        historical
    )

    if mean_volume <= 0:
        return 1.0

    return float(
        volume[-1] /
        mean_volume
    )


# ============================================================
# 21. TAKER BUY RATIO
# ============================================================

def taker_buy_ratio(
    ohlc,
    n: int = 20
) -> float:
    """
    Ratio of taker-buy base volume to total base volume.

    > 0.50:
        buy-side taker volume dominates.

    < 0.50:
        sell-side taker volume dominates.

    This is evidence, not a standalone trading signal.
    """

    buy_volume = ohlc[
        -n:,
        8
    ]

    total_volume = ohlc[
        -n:,
        5
    ]

    denominator = np.sum(
        total_volume
    )

    if denominator <= 0:
        return 0.5

    return float(
        np.sum(buy_volume) /
        denominator
    )


# ============================================================
# 22. ROLLING VWAP
# ============================================================

def vwap(
    ohlc,
    n: int = 50
) -> float:

    data = ohlc[-n:]

    typical_price = (
        data[:, 2] +
        data[:, 3] +
        data[:, 4]
    ) / 3.0

    volume = data[:, 5]

    denominator = np.sum(
        volume
    )

    if denominator <= 0:
        return float(
            data[-1, 4]
        )

    return float(
        np.sum(
            typical_price *
            volume
        ) /
        denominator
    )


# ============================================================
# 23. ATR
# ============================================================

def atr_percent(
    ohlc,
    n: int = 14
) -> float:
    """
    Average True Range normalized by price.
    """

    data = ohlc[
        -(n + 1):
    ]

    previous_close = data[
        :-1,
        4
    ]

    current_high = data[
        1:,
        2
    ]

    current_low = data[
        1:,
        3
    ]

    true_range = np.maximum(
        current_high - current_low,
        np.maximum(
            np.abs(
                current_high -
                previous_close
            ),
            np.abs(
                current_low -
                previous_close
            )
        )
    )

    if len(true_range) == 0:
        return 0.0

    current_price = data[
        -1,
        4
    ]

    return float(
        np.mean(
            true_range[-n:]
        ) /
        current_price *
        100
    )


# ============================================================
# 24. ORDER BOOK MICROSTRUCTURE
# ============================================================

def orderbook_metrics(
    bids,
    asks
) -> Dict[str, float]:

    best_bid = bids[
        0,
        0
    ]

    best_ask = asks[
        0,
        0
    ]

    mid_price = (
        best_bid +
        best_ask
    ) / 2.0

    spread_pct = (
        (best_ask - best_bid)
        / mid_price
        * 100
    )

    bid_depth = np.sum(
        bids[:, 1]
    )

    ask_depth = np.sum(
        asks[:, 1]
    )

    depth_imbalance = (
        bid_depth -
        ask_depth
    ) / max(
        bid_depth +
        ask_depth,
        1e-12
    )

    top5_bid = np.sum(
        bids[:5, 1]
    )

    top5_ask = np.sum(
        asks[:5, 1]
    )

    top5_imbalance = (
        top5_bid -
        top5_ask
    ) / max(
        top5_bid +
        top5_ask,
        1e-12
    )

    return {
        "bid": float(best_bid),
        "ask": float(best_ask),
        "mid": float(mid_price),
        "spread_pct": float(spread_pct),
        "depth_imbalance": float(
            depth_imbalance
        ),
        "top5_imbalance": float(
            top5_imbalance
        ),
        "bid_depth": float(
            bid_depth
        ),
        "ask_depth": float(
            ask_depth
        ),
    }


# ============================================================
# 25. TIMEFRAME ANALYSIS
# ============================================================

def analyze_timeframe(
    ohlc
) -> Dict[str, float]:

    closes = ohlc[
        :,
        4
    ]

    effective_lookback = min(
        LOOKBACK,
        len(closes) - 1
    )

    returns = log_returns(
        closes
    )

    return {
        "price":
            float(closes[-1]),

        "z_price":
            zscore_last(
                closes[
                    -effective_lookback:
                ]
            ),

        "robust_z_price":
            robust_zscore_last(
                closes[
                    -effective_lookback:
                ]
            ),

        "z_return":
            zscore_last(
                returns[
                    -effective_lookback:
                ]
            ),

        "r1":
            rolling_return(
                closes,
                1
            ),

        "r5":
            rolling_return(
                closes,
                min(
                    5,
                    len(closes) - 1
                )
            ),

        "r20":
            rolling_return(
                closes,
                min(
                    20,
                    len(closes) - 1
                )
            ),

        "slope":
            linear_slope_normalized(
                closes,
                min(
                    30,
                    len(closes)
                )
            ),

        "efficiency":
            efficiency_ratio(
                closes,
                min(
                    20,
                    len(closes) - 1
                )
            ),

        "rsi":
            rsi(
                closes,
                min(
                    14,
                    len(closes) - 1
                )
            ),

        "gk":
            garman_klass_vol(
                ohlc,
                effective_lookback
            ),

        "parkinson":
            parkinson_vol(
                ohlc,
                effective_lookback
            ),

        "close_vol":
            close_vol(
                ohlc,
                effective_lookback
            ),

        "atr":
            atr_percent(
                ohlc,
                min(
                    14,
                    len(closes) - 1
                )
            ),

        "relative_volume":
            relative_volume(
                ohlc,
                min(
                    30,
                    len(closes) - 1
                )
            ),

        "taker_buy_ratio":
            taker_buy_ratio(
                ohlc,
                min(
                    20,
                    len(closes)
                )
            ),

        "vwap":
            vwap(
                ohlc,
                min(
                    50,
                    len(closes)
                )
            ),
    }


# ============================================================
# 26. MARKET REGIME
# ============================================================

def classify_regime(
    metrics
) -> str:

    efficiency = metrics[
        "efficiency"
    ]

    volatility = metrics[
        "gk"
    ]

    close_volatility = metrics[
        "close_vol"
    ]

    relative_volume_value = metrics[
        "relative_volume"
    ]

    slope = metrics[
        "slope"
    ]

    # Volatility expansion.
    if (
        close_volatility > 0
        and volatility >
        2.0 * close_volatility
    ):
        return "VOLATILITY_EXPANSION"

    # Sideways / noisy environment.
    if (
        efficiency < 0.25
        and abs(slope) < 0.03
    ):
        return "MEAN_REVERSION_RANGE"

    # Directional upward structure.
    if (
        efficiency >= 0.45
        and slope > 0.02
    ):
        return "TREND_UP"

    # Directional downward structure.
    if (
        efficiency >= 0.45
        and slope < -0.02
    ):
        return "TREND_DOWN"

    # Low participation.
    if relative_volume_value < 0.70:
        return "LOW_PARTICIPATION"

    return "TRANSITION"


# ============================================================
# 27. EVIDENCE ENGINE
# ============================================================

def score_signal(
    primary,
    tf1h,
    tf4h,
    book
):

    score = 0.0

    supporting_evidence = []

    contradictions = []

    # --------------------------------------------------------
    # 27.1 4H STRUCTURE
    # --------------------------------------------------------

    if tf4h["slope"] > 0.02:

        score += 1.5

        supporting_evidence.append(
            "4H normalized trend slope is positive."
        )

    elif tf4h["slope"] < -0.02:

        score -= 1.5

        supporting_evidence.append(
            "4H normalized trend slope is negative."
        )

    # --------------------------------------------------------
    # 27.2 1H STRUCTURE
    # --------------------------------------------------------

    if tf1h["slope"] > 0.02:

        score += 1.0

        supporting_evidence.append(
            "1H normalized trend slope is positive."
        )

    elif tf1h["slope"] < -0.02:

        score -= 1.0

        supporting_evidence.append(
            "1H normalized trend slope is negative."
        )

    # --------------------------------------------------------
    # 27.3 SHORT-TERM RETURN
    # --------------------------------------------------------

    if primary["r5"] > 0:

        score += 0.5

        supporting_evidence.append(
            "Short-term return is positive."
        )

    elif primary["r5"] < 0:

        score -= 0.5

        supporting_evidence.append(
            "Short-term return is negative."
        )

    # --------------------------------------------------------
    # 27.4 RELATIVE VOLUME
    # --------------------------------------------------------

    if (
        primary["relative_volume"] > 1.25
        and primary["r5"] > 0
    ):

        score += 1.0

        supporting_evidence.append(
            "Positive price movement is accompanied by above-normal volume."
        )

    elif (
        primary["relative_volume"] > 1.25
        and primary["r5"] < 0
    ):

        score -= 1.0

        supporting_evidence.append(
            "Negative price movement is accompanied by above-normal volume."
        )

    # --------------------------------------------------------
    # 27.5 TAKER BUY PRESSURE
    # --------------------------------------------------------

    if primary[
        "taker_buy_ratio"
    ] > 0.55:

        score += 0.5

        supporting_evidence.append(
            "Taker-buy volume dominates recent volume."
        )

    elif primary[
        "taker_buy_ratio"
    ] < 0.45:

        score -= 0.5

        supporting_evidence.append(
            "Taker-sell side dominates recent volume."
        )

    # --------------------------------------------------------
    # 27.6 ORDER BOOK
    # --------------------------------------------------------

    if book[
        "top5_imbalance"
    ] > 0.15:

        score += 1.0

        supporting_evidence.append(
            "Top-5 order-book depth favors bids."
        )

    elif book[
        "top5_imbalance"
    ] < -0.15:

        score -= 1.0

        supporting_evidence.append(
            "Top-5 order-book depth favors asks."
        )

    # --------------------------------------------------------
    # 27.7 STATISTICAL EXTREMES
    # --------------------------------------------------------

    if (
        primary["robust_z_price"] > 2.5
        and primary["slope"] < 0
    ):

        score -= 0.75

        contradictions.append(
            "Price is statistically stretched upward while local slope is weakening."
        )

    elif (
        primary["robust_z_price"] < -2.5
        and primary["slope"] > 0
    ):

        score += 0.75

        contradictions.append(
            "Price is statistically stretched downward while local slope is improving."
        )

    # --------------------------------------------------------
    # 27.8 VWAP
    # --------------------------------------------------------

    if primary[
        "price"
    ] > primary[
        "vwap"
    ]:

        score += 0.25

        supporting_evidence.append(
            "Price is above rolling VWAP."
        )

    else:

        score -= 0.25

        supporting_evidence.append(
            "Price is below rolling VWAP."
        )

    # --------------------------------------------------------
    # 27.9 LIQUIDITY PENALTY
    # --------------------------------------------------------

    if book[
        "spread_pct"
    ] > 0.03:

        score *= 0.75

        contradictions.append(
            "Bid-ask spread is relatively wide."
        )

    # --------------------------------------------------------
    # 27.10 FINAL CLASSIFICATION
    # --------------------------------------------------------

    if score >= 3.0:

        signal = "BULLISH_BIAS"

    elif score <= -3.0:

        signal = "BEARISH_BIAS"

    else:

        signal = "NEUTRAL_NO_CLEAR_EDGE"

    # --------------------------------------------------------
    # 27.11 EVIDENCE CONFIDENCE
    # --------------------------------------------------------

    confidence = min(
        100.0,
        max(
            0.0,
            50.0 +
            abs(score) / 7.0 * 50.0
        )
    )

    if signal == "NEUTRAL_NO_CLEAR_EDGE":

        confidence = min(
            confidence,
            55.0
        )

    return (
        signal,
        score,
        confidence,
        supporting_evidence,
        contradictions
    )


# ============================================================
# 28. FORMATTING
# ============================================================

def fmt_pct(value):
    return f"{value * 100:+.2f}%"


def build_report(
    tf15,
    tf1h,
    tf4h,
    book,
    signal,
    score,
    confidence,
    supporting_evidence,
    contradictions
):

    price = tf15[
        "price"
    ]

    regime = classify_regime(
        tf15
    )

    lines = [

        "<b>BTCUSDT — ADVANCED STATISTICAL RESEARCH</b>",

        f"<b>Primary timeframe:</b> {escape(PRIMARY_INTERVAL)}",

        f"<b>Price:</b> ${price:,.2f}",

        "",

        "<b>MARKET REGIME</b>",

        escape(regime),

        "",

        "<b>AGGREGATED EVIDENCE</b>",

        f"Signal bias: <b>{escape(signal)}</b>",

        f"Evidence score: <b>{score:+.2f}</b>",

        f"Evidence confidence: <b>{confidence:.0f}/100</b>",

        "",

        "<b>15M — LOCAL STRUCTURE</b>",

        f"1/5/20 candle return: "
        f"{fmt_pct(tf15['r1'])} / "
        f"{fmt_pct(tf15['r5'])} / "
        f"{fmt_pct(tf15['r20'])}",

        f"Price Z-score: {tf15['z_price']:+.2f}",

        f"Robust Z-score: {tf15['robust_z_price']:+.2f}",

        f"Return Z-score: {tf15['z_return']:+.2f}",

        f"RSI: {tf15['rsi']:.1f}",

        f"Trend slope: {tf15['slope']:+.3f}%/bar",

        f"Efficiency ratio: {tf15['efficiency']:.2f}",

        "",

        "<b>VOLATILITY</b>",

        f"Garman-Klass: {tf15['gk']:.3f}%",

        f"Parkinson: {tf15['parkinson']:.3f}%",

        f"Close-to-close: {tf15['close_vol']:.3f}%",

        f"ATR: {tf15['atr']:.3f}%",

        "",

        "<b>VOLUME / FLOW</b>",

        f"Relative volume: {tf15['relative_volume']:.2f}x",

        f"Taker-buy ratio: {tf15['taker_buy_ratio']:.3f}",

        f"Rolling VWAP: ${tf15['vwap']:,.2f}",

        "",

        "<b>MULTI-TIMEFRAME STRUCTURE</b>",

        f"1H return20: {fmt_pct(tf1h['r20'])}",

        f"1H slope: {tf1h['slope']:+.3f}%/bar",

        f"1H efficiency: {tf1h['efficiency']:.2f}",

        "",

        f"4H return20: {fmt_pct(tf4h['r20'])}",

        f"4H slope: {tf4h['slope']:+.3f}%/bar",

        f"4H efficiency: {tf4h['efficiency']:.2f}",

        "",

        "<b>ORDER BOOK</b>",

        f"Best bid: ${book['bid']:,.2f}",

        f"Best ask: ${book['ask']:,.2f}",

        f"Spread: {book['spread_pct']:.4f}%",

        f"Depth imbalance: {book['depth_imbalance']:+.3f}",

        f"Top-5 imbalance: {book['top5_imbalance']:+.3f}",

        "",

        "<b>SUPPORTING EVIDENCE</b>",
    ]

    for item in supporting_evidence[:8]:

        lines.append(
            "• " + escape(item)
        )

    lines += [
        "",
        "<b>CONTRADICTIONS / RISK FLAGS</b>"
    ]

    for item in contradictions[:6]:

        lines.append(
            "• " + escape(item)
        )

    if not contradictions:

        lines.append(
            "• No major contradiction detected by current rule set."
        )

    lines += [

        "",

        "<b>SCIENTIFIC INTERPRETATION</b>",

        "The score aggregates several independent market observations.",

        "It is NOT a probability of profit.",

        "It is NOT a guarantee of future direction.",

        "A single indicator should not be treated as sufficient evidence.",

        "Final execution remains a human decision.",

    ]

    text = "\n".join(
        lines
    )

    # Telegram sendMessage allows up to 4096 characters.
    return text[:4090]


# ============================================================
# 29. TELEGRAM
# ============================================================

def send_telegram(
    text: str
) -> bool:

    url = (
        f"{TELEGRAM_API_URL}"
        f"/bot{TELEGRAM_BOT_TOKEN}"
        f"/sendMessage"
    )

    payload = {

        "chat_id":
            TELEGRAM_CHAT_ID,

        "text":
            text,

        "parse_mode":
            "HTML",

        "disable_web_page_preview":
            True,
    }

    response = SESSION.post(
        url,
        json=payload,
        timeout=HTTP_TIMEOUT
    )

    response.raise_for_status()

    data = response.json()

    if not data.get("ok"):

        raise RuntimeError(
            f"Telegram API error: {data}"
        )

    return True


# ============================================================
# 30. ONE COMPLETE ANALYSIS CYCLE
# ============================================================

def run_cycle():

    log.info(
        "Fetching BTCUSDT market data..."
    )

    # Main timeframe.
    primary_data = fetch_klines(
        PRIMARY_INTERVAL
    )

    # Higher-timeframe context.
    one_hour_data = fetch_klines(
        "1h"
    )

    four_hour_data = fetch_klines(
        "4h"
    )

    # Current order book.
    bids, asks = fetch_order_book()

    # Statistical analysis.
    primary = analyze_timeframe(
        primary_data
    )

    one_hour = analyze_timeframe(
        one_hour_data
    )

    four_hour = analyze_timeframe(
        four_hour_data
    )

    book = orderbook_metrics(
        bids,
        asks
    )

    # Evidence engine.
    (
        signal,
        score,
        confidence,
        supporting_evidence,
        contradictions
    ) = score_signal(
        primary,
        one_hour,
        four_hour,
        book
    )

    # Human-readable research report.
    report = build_report(
        primary,
        one_hour,
        four_hour,
        book,
        signal,
        score,
        confidence,
        supporting_evidence,
        contradictions
    )

    # Send report to Telegram.
    send_telegram(
        report
    )

    log.info(
        "BTCUSDT | price=%.2f | signal=%s | "
        "score=%+.2f | confidence=%.0f | "
        "regime=%s | spread=%.4f%%",
        primary["price"],
        signal,
        score,
        confidence,
        classify_regime(primary),
        book["spread_pct"]
    )


# ============================================================
# 31. MAIN LOOP
# ============================================================

def main():

    require_config()

    log.info(
        "Starting BTCUSDT statistical research engine."
    )

    log.info(
        "Symbol=%s | primary_interval=%s | "
        "lookback=%s | candles=%s | polling=%ss",
        SYMBOL,
        PRIMARY_INTERVAL,
        LOOKBACK,
        CANDLE_LIMIT,
        POLL_SECONDS
    )

    while True:

        cycle_start = time.time()

        try:

            run_cycle()

        except requests.RequestException as exc:

            log.exception(
                "Network/API error: %s",
                exc
            )

        except Exception as exc:

            # One failed cycle must not kill Railway.
            log.exception(
                "Unexpected analysis error: %s",
                exc
            )

        elapsed = (
            time.time() -
            cycle_start
        )

        sleep_for = max(
            5,
            POLL_SECONDS -
            int(elapsed)
        )

        time.sleep(
            sleep_for
        )


# ============================================================
# 32. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
