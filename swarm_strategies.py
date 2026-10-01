"""
swarm_strategies.py
===================

Adaptive Multi-Strategy Swarm
-----------------------------

هذا الملف يمثل طبقة "Strategy Population".

الفكرة ليست:
    100 مؤشر → 100 تصويت → الأكثر أصواتًا يفوز.

بل:

    Strategy Population
            ↓
    Family Aggregation
            ↓
    Regime Awareness
            ↓
    Adaptive Fitness
            ↓
    Diversity Control
            ↓
    Consensus
            ↓
    Final Signal

مصادر التصميم:
----------------
- Eugene A. Durenard
  Professional Automated Trading
  Chapters 7, 9, 10, 12, 13, 14, 15

  خصوصًا:
      - Switching Strategies
      - Switching between Regimes
      - Strategy Neighborhoods
      - Choice of a Simple Individual
      - Additive Swarm
      - Maximizing Swarm
      - Global Performance Feedback

- Marcos López de Prado
  Advances in Financial Machine Learning

  مفاهيم مستخدمة هنا:
      - backtest overfitting awareness
      - multiple testing awareness
      - model/strategy diversification
      - meta-level performance control

ملاحظة مهمة:
-------------
هذا الملف لا يقوم بعمل backtest.

إنه:
    SIGNAL ENGINE

وليس:
    BACKTEST ENGINE

الـ Backtester يجب أن يختبر هذا السرب خارج العينة
ويحسب:
    Sharpe
    Sortino
    Max Drawdown
    Calmar
    turnover
    costs
    Deflated Sharpe
    PBO / multiple-testing diagnostics

قبل السماح للسرب بالتعلم من النتائج.

Input:
------
NumPy array بالشكل:

    [open, high, low, close, volume]

ترتيب الأعمدة:
    0 = Open
    1 = High
    2 = Low
    3 = Close
    4 = Volume


===========================================================
IMPORTANT DESIGN RULE
===========================================================

لا نريد أن تؤدي إضافة:

    ZScore(10)
    ZScore(11)
    ZScore(12)
    ...
    ZScore(100)

إلى جعل "Z-Score family" تسيطر على السرب.

لذلك:

    strategy weights
            ↓
    normalized INSIDE each family
            ↓
    family aggregation
            ↓
    family weights

وبالتالي 30 نسخة من Z-Score لا تتغلب
تلقائيًا على 5 استراتيجيات من عائلة أخرى.


===========================================================
NO LOOK-AHEAD
===========================================================

كل Strategy تستخدم البيانات حتى الشمعة الأخيرة فقط.

لا يتم استخدام:
    future return
    next candle
    future high
    future low

داخل analyze().

أما update() فيعمل فقط بعد تحقق النتيجة
ويُستخدم لتحديث Fitness للمستقبل.


===========================================================
"""

from __future__ import annotations

import logging
import math
import statistics

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


log = logging.getLogger("swarm_strategies")


# ============================================================================
# Constants
# ============================================================================

BUY = 1
SELL = -1
HOLD = 0

EPS = 1e-12


# ============================================================================
# Generic helpers
# ============================================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)

        if not np.isfinite(x):
            return default

        return x

    except (TypeError, ValueError):
        return default


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _sign(x: float) -> int:

    if x > 0:
        return BUY

    if x < 0:
        return SELL

    return HOLD


def _rolling_std(
    x: np.ndarray,
    period: int,
) -> float:

    if len(x) < period:
        return 0.0

    return float(
        np.std(
            x[-period:],
            ddof=0,
        )
    )


def _sma(
    x: np.ndarray,
    period: int,
) -> float:

    if len(x) < period:
        return float("nan")

    return float(
        np.mean(
            x[-period:]
        )
    )


def _ema(
    x: np.ndarray,
    period: int,
) -> float:

    if len(x) < period:
        return float("nan")

    alpha = 2.0 / (period + 1.0)

    value = float(
        np.mean(
            x[:period]
        )
    )

    for price in x[period:]:
        value = (
            alpha * float(price)
            + (1.0 - alpha) * value
        )

    return value


def _ema_series(
    x: np.ndarray,
    period: int,
) -> np.ndarray:

    result = np.full(
        len(x),
        np.nan,
        dtype=float,
    )

    if len(x) < period:
        return result

    alpha = 2.0 / (period + 1.0)

    result[period - 1] = np.mean(
        x[:period]
    )

    for i in range(period, len(x)):
        result[i] = (
            alpha * x[i]
            + (1.0 - alpha) * result[i - 1]
        )

    return result


def _true_range(
    candles: np.ndarray,
) -> np.ndarray:

    highs = candles[:, 1]
    lows = candles[:, 2]
    closes = candles[:, 3]

    tr = np.zeros(
        len(candles),
        dtype=float,
    )

    tr[0] = highs[0] - lows[0]

    for i in range(1, len(candles)):

        tr[i] = max(
            highs[i] - lows[i],
            abs(
                highs[i]
                - closes[i - 1]
            ),
            abs(
                lows[i]
                - closes[i - 1]
            ),
        )

    return tr


def _atr(
    candles: np.ndarray,
    period: int = 14,
) -> float:

    tr = _true_range(candles)

    if len(tr) < period:
        return float("nan")

    # Wilder-style recursive ATR.
    atr = float(
        np.mean(
            tr[:period]
        )
    )

    alpha = 1.0 / period

    for value in tr[period:]:
        atr = (
            (1.0 - alpha) * atr
            + alpha * value
        )

    return atr


def _returns(
    closes: np.ndarray,
) -> np.ndarray:

    if len(closes) < 2:
        return np.array([], dtype=float)

    return np.diff(closes) / np.maximum(
        closes[:-1],
        EPS,
    )


def _rsi_wilder(
    closes: np.ndarray,
    period: int = 14,
) -> float:

    if len(closes) < period + 1:
        return float("nan")

    delta = np.diff(closes)

    gains = np.maximum(
        delta,
        0.0,
    )

    losses = np.maximum(
        -delta,
        0.0,
    )

    avg_gain = np.mean(
        gains[:period]
    )

    avg_loss = np.mean(
        losses[:period]
    )

    alpha = 1.0 / period

    for i in range(period, len(gains)):

        avg_gain = (
            (1.0 - alpha) * avg_gain
            + alpha * gains[i]
        )

        avg_loss = (
            (1.0 - alpha) * avg_loss
            + alpha * losses[i]
        )

    if avg_loss <= EPS:

        if avg_gain > EPS:
            return 100.0

        return 50.0

    rs = avg_gain / avg_loss

    return 100.0 - (
        100.0
        / (1.0 + rs)
    )


def _macd(
    closes: np.ndarray,
    fast: int,
    slow: int,
    signal: int,
) -> Tuple[float, float, float]:

    if len(closes) < slow + signal:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
        )

    fast_series = _ema_series(
        closes,
        fast,
    )

    slow_series = _ema_series(
        closes,
        slow,
    )

    macd_series = (
        fast_series
        - slow_series
    )

    valid = macd_series[
        np.isfinite(macd_series)
    ]

    if len(valid) < signal:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
        )

    signal_series = _ema_series(
        valid,
        signal,
    )

    macd_value = float(
        macd_series[-1]
    )

    signal_value = float(
        signal_series[-1]
    )

    histogram = (
        macd_value
        - signal_value
    )

    return (
        macd_value,
        signal_value,
        histogram,
    )


def _stochastic(
    candles: np.ndarray,
    period: int = 14,
) -> Tuple[float, float]:

    if len(candles) < period + 1:
        return (
            float("nan"),
            float("nan"),
        )

    highs = candles[:, 1]
    lows = candles[:, 2]
    closes = candles[:, 3]

    k_values = []

    start = max(
        period - 1,
        len(candles) - 4,
    )

    for i in range(
        start,
        len(candles),
    ):

        window_high = np.max(
            highs[i - period + 1:i + 1]
        )

        window_low = np.min(
            lows[i - period + 1:i + 1]
        )

        denominator = (
            window_high
            - window_low
        )

        if denominator <= EPS:

            k = 50.0

        else:

            k = (
                (
                    closes[i]
                    - window_low
                )
                / denominator
            ) * 100.0

        k_values.append(k)

    if not k_values:
        return (
            float("nan"),
            float("nan"),
        )

    k = k_values[-1]

    d = float(
        np.mean(
            k_values[-3:]
        )
    )

    return k, d


def _cci(
    candles: np.ndarray,
    period: int = 20,
) -> float:

    if len(candles) < period:
        return float("nan")

    high = candles[:, 1]
    low = candles[:, 2]
    close = candles[:, 3]

    typical = (
        high
        + low
        + close
    ) / 3.0

    window = typical[-period:]

    mean = np.mean(window)

    mean_dev = np.mean(
        np.abs(
            window - mean
        )
    )

    if mean_dev <= EPS:
        return 0.0

    return float(
        (
            typical[-1]
            - mean
        )
        / (
            0.015
            * mean_dev
        )
    )


def _vwap(
    candles: np.ndarray,
    period: int = 30,
) -> float:

    if len(candles) < period:
        return float("nan")

    high = candles[-period:, 1]
    low = candles[-period:, 2]
    close = candles[-period:, 3]
    volume = candles[-period:, 4]

    typical = (
        high
        + low
        + close
    ) / 3.0

    volume_sum = np.sum(
        volume
    )

    if volume_sum <= EPS:
        return float("nan")

    return float(
        np.sum(
            typical * volume
        )
        / volume_sum
    )


def _linear_slope(
    x: np.ndarray,
    period: int,
) -> float:

    if len(x) < period:
        return float("nan")

    y = x[-period:]

    if np.std(y) <= EPS:
        return 0.0

    axis = np.arange(
        period,
        dtype=float,
    )

    slope = np.polyfit(
        axis,
        y,
        1,
    )[0]

    return float(slope)


def _volume_ratio(
    volume: np.ndarray,
    period: int = 20,
) -> float:

    if len(volume) < period + 1:
        return float("nan")

    baseline = np.mean(
        volume[-period - 1:-1]
    )

    if baseline <= EPS:
        return 1.0

    return float(
        volume[-1]
        / baseline
    )


def _robust_zscore(
    x: np.ndarray,
    period: int,
) -> float:

    if len(x) < period:
        return float("nan")

    window = x[-period:]

    median = np.median(
        window
    )

    mad = np.median(
        np.abs(
            window - median
        )
    )

    if mad <= EPS:
        return 0.0

    # 1.4826 converts MAD approximately to
    # Gaussian-equivalent standard deviation.
    robust_sigma = (
        1.4826 * mad
    )

    return float(
        (
            window[-1]
            - median
        )
        / robust_sigma
    )


def _percentile_rank(
    value: float,
    history: np.ndarray,
) -> float:

    if len(history) == 0:
        return 0.5

    return float(
        np.mean(
            history <= value
        )
    )


# ============================================================================
# Results
# ============================================================================

@dataclass
class StrategyResult:

    signal: int
    confidence: float
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:

        self.signal = int(
            np.clip(
                self.signal,
                -1,
                1,
            )
        )

        self.confidence = _clip01(
            self.confidence
        )

    @property
    def signed_score(self) -> float:

        return float(
            self.signal
            * self.confidence
        )


@dataclass
class SwarmResult:

    signal: int
    confidence: float

    score: float

    regime: str

    breadth: float

    active_strategies: int

    family_scores: Dict[str, float]

    strategy_scores: Dict[str, float]

    diagnostics: Dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================================
# Base Strategy
# ============================================================================

class BaseStrategy:
    """
    Base object.

    family:
        Strategy family.

    params:
        Parameter set.

    priority:
        Base prior weight.

    regime_bias:
        Optional compatibility:
            trend
            mean_reversion
            breakout
            neutral
    """

    def __init__(
        self,
        name: str,
        params: Dict[str, Any],
        family: str,
        priority: float = 1.0,
        regime_bias: str = "neutral",
    ):

        self.name = name
        self.params = dict(params)
        self.family = family
        self.priority = float(
            max(
                priority,
                0.01,
            )
        )
        self.regime_bias = (
            regime_bias
        )

    def analyze(
        self,
        candles: np.ndarray,
    ) -> StrategyResult:

        raise NotImplementedError

    def __repr__(self) -> str:

        return (
            f"{self.name}("
            f"family={self.family}, "
            f"params={self.params})"
        )


# ============================================================================
# Strategy 1: Standard Z-Score
# ============================================================================

class ZScoreStrategy(BaseStrategy):

    def __init__(
        self,
        period: int = 20,
        threshold: float = 2.0,
    ):

        super().__init__(
            name=(
                f"ZScore("
                f"{period},"
                f"{threshold})"
            ),
            params={
                "period": period,
                "threshold": threshold,
            },
            family="mean_reversion",
            priority=1.0,
            regime_bias="mean_reversion",
        )

    def analyze(
        self,
        candles: np.ndarray,
    ) -> StrategyResult:

        closes = candles[:, 3]

        period = int(
            self.params["period"]
        )

        threshold = float(
            self.params["threshold"]
        )

        if len(closes) < period:

            return StrategyResult(
                HOLD,
                0.0,
                {"reason": "insufficient_data"},
            )

        window = closes[-period:]

        mean = float(
            np.mean(window)
        )

        std = float(
            np.std(
                window,
                ddof=0,
            )
        )

        if std <= EPS:

            return StrategyResult(
                HOLD,
                0.0,
                {"reason": "zero_std"},
            )

        z = (
            float(closes[-1])
            - mean
        ) / std

        if z <= -threshold:

            confidence = min(
                (
                    abs(z)
                    - threshold
                    + 1.0
                ) / 2.0,
                1.0,
            )

            return StrategyResult(
                BUY,
                confidence,
                {
                    "z": z,
                    "threshold": threshold,
                },
            )

        if z >= threshold:

            confidence = min(
                (
                    abs(z)
                    - threshold
                    + 1.0
                ) / 2.0,
                1.0,
            )

            return StrategyResult(
                SELL,
                confidence,
                {
                    "z": z,
                    "threshold": threshold,
                },
            )

        return StrategyResult(
            HOLD,
            0.0,
            {"z": z},
        )


# ============================================================================
# Strategy 2: Robust Z-Score
# ============================================================================

class RobustZScoreStrategy(BaseStrategy):

    def __init__(
        self,
        period: int = 30,
        threshold: float = 2.5,
    ):

        super().__init__(
            name=(
                f"RobustZ("
                f"{period},"
                f"{threshold})"
            ),
            params={
                "period": period,
                "threshold": threshold,
            },
            family="robust_mean_reversion",
            priority=1.0,
            regime_bias="mean_reversion",
        )

    def analyze(
        self,
        candles: np.ndarray,
    ) -> StrategyResult:

        closes = candles[:, 3]

        period = int(
            self.params["period"]
        )

        threshold = float(
            self.params["threshold"]
        )

        z = _robust_zscore(
            closes,
            period,
        )

        if not np.isfinite(z):

            return StrategyResult(
                HOLD,
                0.0,
                {"reason": "insufficient_data"},
            )

        if z <= -threshold:

            strength = min(
                abs(z)
                / max(
                    threshold,
                    EPS,
                ),
                2.0,
            )

            return StrategyResult(
                BUY,
                _clip01(
                    strength / 2.0
                ),
                {"robust_z": z},
            )

        if z >= threshold:

            strength = min(
                abs(z)
                / max(
                    threshold,
                    EPS,
                ),
                2.0,
            )

            return StrategyResult(
                SELL,
                _clip01(
                    strength / 2.0
                ),
                {"robust_z": z},
            )

        return StrategyResult(
            HOLD,
            0.0,
            {"robust_z": z},
        )


# ============================================================================
# Strategy 3: RSI
# ============================================================================

class RSIStrategy(BaseStrategy):

    def __init__(
        self,
        period: int = 14,
        oversold: int = 30,
        overbought: int = 70,
    ):

        super().__init__(
            name=(
                f"RSI("
                f"{period},"
                f"{oversold},"
                f"{overbought})"
            ),
            params={
                "period": period,
                "oversold": oversold,
                "overbought": overbought,
            },
            family="mean_reversion",
            priority=1.0,
            regime_bias="mean_reversion",
        )

    def analyze(
        self,
        candles: np.ndarray,
    ) -> StrategyResult:

        closes = candles[:, 3]

        period = int(
            self.params["period"]
        )

        oversold = float(
            self.params["oversold"]
        )

        overbought = float(
            self.params["overbought"]
        )

        rsi = _rsi_wilder(
            closes,
            period,
        )

        if not np.isfinite(rsi):

            return StrategyResult(
                HOLD,
                0.0,
                {"reason": "insufficient_data"},
            )

        if rsi <= oversold:

            confidence = _clip01(
                (
                    oversold
                    - rsi
                )
                / max(
                    oversold,
                    1.0,
                )
            )

            return StrategyResult(
                BUY,
                confidence,
                {"rsi": rsi},
            )

        if rsi >= overbought:

            confidence = _clip01(
                (
                    rsi
                    - overbought
                )
                / max(
                    100.0 - overbought,
                    1.0,
                )
            )

            return StrategyResult(
                SELL,
                confidence,
                {"rsi": rsi},
            )

        return StrategyResult(
            HOLD,
            0.0,
            {"rsi": rsi},
        )


# ============================================================================
# Strategy 4: Moving Average Crossover
# ============================================================================

class MACrossoverStrategy(BaseStrategy):

    def __init__(
        self,
        fast: int = 10,
        slow: int = 30,
    ):

        if fast >= slow:
            raise ValueError(
                "fast must be < slow"
            )

        super().__init__(
            name=(
                f"EMA("
                f"{fast},"
                f"{slow})"
            ),
            params={
                "fast": fast,
                "slow": slow,
            },
            family="trend",
            priority=1.0,
            regime_bias="trend",
        )

    def analyze(
        self,
        candles: np.ndarray,
    ) -> StrategyResult:

        closes = candles[:, 3]

        fast = int(
            self.params["fast"]
        )

        slow = int(
            self.params["slow"]
        )

        if len(closes) < slow + 2:

            return StrategyResult(
                HOLD,
                0.0,
                {"reason": "insufficient_data"},
            )

        fast_series = _ema_series(
            closes,
            fast,
        )

        slow_series = _ema_series(
            closes,
            slow,
        )

        prev_fast = (
            fast_series[-2]
        )

        prev_slow = (
            slow_series[-2]
        )

        current_fast = (
            fast_series[-1]
        )

        current_slow = (
            slow_series[-1]
        )

        if not all(
            np.isfinite(
                [
                    prev_fast,
                    prev_slow,
                    current_fast,
                    current_slow,
                ]
            )
        ):

            return StrategyResult(
                HOLD,
                0.0,
                {"reason": "invalid_ma"},
            )

        # Actual crossing, not merely fast > slow.
        if (
            prev_fast
            <= prev_slow
            and current_fast
            > current_slow
        ):

            spread = abs(
                current_fast
                - current_slow
            )

            confidence = _clip01(
                spread
                / max(
                    float(
                        _atr(candles, 14)
                    ),
                    EPS,
                )
            )

            return StrategyResult(
                BUY,
                confidence,
                {"cross": "bullish"},
            )

        if (
            prev_fast
            >= prev_slow
            and current_fast
            < current_slow
        ):

            spread = abs(
                current_fast
                - current_slow
            )

            confidence = _clip01(
                spread
                / max(
                    float(
                        _atr(candles, 14)
                    ),
                    EPS,
                )
            )

            return StrategyResult(
                SELL,
                confidence,
                {"cross": "bearish"},
            )

        return StrategyResult(
            HOLD,
            0.0,
            {
                "fast": current_fast,
                "slow": current_slow,
            },
        )


# ============================================================================
# Strategy 5: Bollinger
# ============================================================================

class BollingerStrategy(BaseStrategy):

    def __init__(
        self,
        period: int = 20,
        num_std: float = 2.0,
    ):

        super().__init__(
            name=(
                f"BB("
                f"{period},"
                f"{num_std})"
            ),
            params={
                "period": period,
                "num_std": num_std,
            },
            family="mean_reversion",
            priority=1.0,
            regime_bias="mean_reversion",
        )

    def analyze(
        self,
        candles: np.ndarray,
    ) -> StrategyResult:

        closes = candles[:, 3]

        period = int(
            self.params["period"]
        )

        num_std = float(
            self.params["num_std"]
        )

        if len(closes) < period:

            return StrategyResult(
                HOLD,
                0.0,
                {"reason": "insufficient_data"},
            )

        window = closes[-period:]

        mean = float(
            np.mean(window)
        )

        std = float(
            np.std(
                window
            )
        )

        if std <= EPS:

            return StrategyResult(
                HOLD,
                0.0,
                {"reason": "zero_std"},
            )

        upper = (
            mean
            + num_std * std
        )

        lower = (
            mean
            - num_std * std
        )

        price = float(
            closes[-1]
        )

        width = (
            upper
            - lower
        )

        if price <= lower:

            position = (
                lower - price
            ) / max(
                width,
                EPS,
            )

            return StrategyResult(
                BUY,
                _clip01(
                    0.5
                    + position
                ),
                {
                    "band": "lower",
                    "width": width,
                },
            )

        if price >= upper:

            position = (
                price - upper
            ) / max(
                width,
                EPS,
            )

            return StrategyResult(
                SELL,
                _clip01(
                    0.5
                    + position
                ),
                {
                    "band": "upper",
                    "width": width,
                },
            )

        return StrategyResult(
            HOLD,
            0.0,
            {
                "price": price,
                "middle": mean,
                "width": width,
            },
        )


# ============================================================================
# Strategy 6: Donchian Breakout
# ============================================================================

class DonchianStrategy(BaseStrategy):

    def __init__(
        self,
        period: int = 20,
    ):

        super().__init__(
            name=f"Donchian({period})",
            params={
                "period": period,
            },
            family="breakout",
            priority=1.0,
            regime_bias="breakout",
        )

    def analyze(
        self,
        candles: np.ndarray,
    ) -> StrategyResult:

        period = int(
            self.params["period"]
        )

        if len(candles) < period + 1:

            return StrategyResult(
                HOLD,
                0.0,
                {"reason": "insufficient_data"},
            )

        highs = candles[:, 1]
        lows = candles[:, 2]
        close = float(
            candles[-1, 3]
        )

        # Exclude current candle from the
        # historical channel.
        upper = float(
            np.max(
                highs[-period - 1:-1]
            )
        )

        lower = float(
            np.min(
                lows[-period - 1:-1]
            )
        )

        if close > upper:

            distance = (
                close - upper
            ) / max(
                close,
                EPS,
            )

            return StrategyResult(
                BUY,
                _clip01(
                    distance * 100.0
                ),
                {
                    "breakout": "up",
                    "upper": upper,
                },
            )

        if close < lower:

            distance = (
                lower - close
            ) / max(
                close,
                EPS,
            )

            return StrategyResult(
                SELL,
                _clip01(
                    distance * 100.0
                ),
                {
                    "breakout": "down",
                    "lower": lower,
                },
            )

        return StrategyResult(
            HOLD,
            0.0,
            {
                "upper": upper,
                "lower": lower,
            },
        )


# ============================================================================
# Strategy 7: Momentum
# ============================================================================

class MomentumStrategy(BaseStrategy):

    def __init__(
        self,
        period: int = 20,
        threshold: float = 0.01,
    ):

        super().__init__(
            name=(
                f"Momentum("
                f"{period},"
                f"{threshold})"
            ),
            params={
                "period": period,
                "threshold": threshold,
            },
            family="momentum",
            priority=1.0,
            regime_bias="trend",
        )

    def analyze(
        self,
        candles: np.ndarray,
    ) -> StrategyResult:

        closes = candles[:, 3]

        period = int(
            self.params["period"]
        )

        threshold = float(
            self.params["threshold"]
        )

        if len(closes) < period + 1:

            return StrategyResult(
                HOLD,
                0.0,
                {"reason": "insufficient_data"},
            )

        change = (
            closes[-1]
            / max(
                closes[-period - 1],
                EPS,
            )
            - 1.0
        )

        if change >= threshold:

            return StrategyResult(
                BUY,
                _clip01(
                    change
                    / max(
                        threshold * 3.0,
                        EPS,
                    )
                ),
                {
                    "return": change,
                },
            )

        if change <= -threshold:

            return StrategyResult(
                SELL,
                _clip01(
                    abs(change)
                    / max(
                        threshold * 3.0,
                        EPS,
                    )
                ),
                {
                    "return": change,
                },
            )

        return StrategyResult(
            HOLD,
            0.0,
            {"return": change},
        )


# ============================================================================
# Strategy 8: MACD
# ============================================================================

class MACDStrategy(BaseStrategy):

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ):

        if fast >= slow:
            raise ValueError(
                "MACD fast must be < slow"
            )

        super().__init__(
            name=(
                f"MACD("
                f"{fast},"
                f"{slow},"
                f"{signal})"
            ),
            params={
                "fast": fast,
                "slow": slow,
                "signal": signal,
            },
            family="trend",
            priority=1.0,
            regime_bias="trend",
        )

    def analyze(
        self,
        candles: np.ndarray,
    ) -> StrategyResult:

        closes = candles[:, 3]

        fast = int(
            self.params["fast"]
        )

        slow = int(
            self.params["slow"]
        )

        signal_period = int(
            self.params["signal"]
        )

        macd, signal, histogram = _macd(
            closes,
            fast,
            slow,
            signal_period,
        )

        if not np.isfinite(
            histogram
        ):

            return StrategyResult(
                HOLD,
                0.0,
                {"reason": "insufficient_data"},
            )

        atr = _atr(
            candles,
            14,
        )

        if not np.isfinite(atr) or atr <= EPS:

            return StrategyResult(
                HOLD,
                0.0,
                {"histogram": histogram},
            )

        normalized = abs(
            histogram
        ) / atr

        if histogram > 0:

            return StrategyResult(
                BUY,
                _clip01(
                    normalized
                ),
                {
                    "macd": macd,
                    "signal": signal,
                    "histogram": histogram,
                },
            )

        if histogram < 0:

            return StrategyResult(
                SELL,
                _clip01(
                    normalized
                ),
                {
                    "macd": macd,
                    "signal": signal,
                    "histogram": histogram,
                },
            )

        return StrategyResult(
            HOLD,
            0.0,
            {
                "macd": macd,
                "signal": signal,
            },
        )


# ============================================================================
# Strategy 9: Stochastic
# ============================================================================

class StochasticStrategy(BaseStrategy):

    def __init__(
        self,
        period: int = 14,
        oversold: float = 20.0,
        overbought: float = 80.0,
    ):

        super().__init__(
            name=(
                f"Stoch("
                f"{period},"
                f"{oversold},"
                f"{overbought})"
            ),
            params={
                "period": period,
                "oversold": oversold,
                "overbought": overbought,
            },
            family="mean_reversion",
            priority=1.0,
            regime_bias="mean_reversion",
        )

    def analyze(
        self,
        candles: np.ndarray,
    ) -> StrategyResult:

        period = int(
            self.params["period"]
        )

        oversold = float(
            self.params["oversold"]
        )

        overbought = float(
            self.params["overbought"]
        )

        k, d = _stochastic(
            candles,
            period,
        )

        if not (
            np.isfinite(k)
            and np.isfinite(d)
        ):

            return StrategyResult(
                HOLD,
                0.0,
                {"reason": "insufficient_data"},
            )

        # Reversal from oversold.
        if (
            k < oversold
            and k > d
        ):

            confidence = _clip01(
                (
                    oversold - k
                )
                / max(
                    oversold,
                    EPS,
                )
            )

            return StrategyResult(
                BUY,
                confidence,
                {"k": k, "d": d},
            )

        # Reversal from overbought.
        if (
            k > overbought
            and k < d
        ):

            confidence = _clip01(
                (
                    k - overbought
                )
                / max(
                    100.0 - overbought,
                    EPS,
                )
            )

            return StrategyResult(
                SELL,
                confidence,
                {"k": k, "d": d},
            )

        return StrategyResult(
            HOLD,
            0.0,
            {"k": k, "d": d},
        )


# ============================================================================
# Strategy 10: CCI
# ============================================================================

class CCIStrategy(BaseStrategy):

    def __init__(
        self,
        period: int = 20,
        threshold: float = 100.0,
    ):

        super().__init__(
            name=(
                f"CCI("
                f"{period},"
                f"{threshold})"
            ),
            params={
                "period": period,
                "threshold": threshold,
            },
            family="mean_reversion",
            priority=1.0,
            regime_bias="mean_reversion",
        )

    def analyze(
        self,
        candles: np.ndarray,
    ) -> StrategyResult:

        period = int(
            self.params["period"]
        )

        threshold = float(
            self.params["threshold"]
        )

        cci = _cci(
            candles,
            period,
        )

        if not np.isfinite(cci):

            return StrategyResult(
                HOLD,
                0.0,
                {"reason": "insufficient_data"},
            )

        if cci <= -threshold:

            return StrategyResult(
                BUY,
                _clip01(
                    abs(cci)
                    / max(
                        threshold * 2.0,
                        EPS,
                    )
                ),
                {"cci": cci},
            )

        if cci >= threshold:

            return StrategyResult(
                SELL,
                _clip01(
                    abs(cci)
                    / max(
                        threshold * 2.0,
                        EPS,
                    )
                ),
                {"cci": cci},
            )

        return StrategyResult(
            HOLD,
            0.0,
            {"cci": cci},
        )


# ============================================================================
# Strategy 11: ATR Breakout
# ============================================================================

class ATRBreakoutStrategy(BaseStrategy):

    def __init__(
        self,
        period: int = 14,
        multiplier: float = 1.5,
    ):

        super().__init__(
            name=(
                f"ATRBreakout("
                f"{period},"
                f"{multiplier})"
            ),
            params={
                "period": period,
                "multiplier": multiplier,
            },
            family="volatility_breakout",
            priority=1.0,
            regime_bias="breakout",
        )

    def analyze(
        self,
        candles: np.ndarray,
    ) -> StrategyResult:

        period = int(
            self.params["period"]
        )

        multiplier = float(
            self.params["multiplier"]
        )

        if len(candles) < period + 2:

            return StrategyResult(
                HOLD,
                0.0,
                {"reason": "insufficient_data"},
            )

        atr = _atr(
            candles,
            period,
        )

        if not np.isfinite(atr) or atr <= EPS:

            return StrategyResult(
                HOLD,
                0.0,
                {"reason": "invalid_atr"},
            )

        previous_range_high = np.max(
            candles[
                -period - 1:-1,
                1
            ]
        )

        previous_range_low = np.min(
            candles[
                -period - 1:-1,
                2
            ]
        )

        price = candles[-1, 3]

        up_distance = (
            price
            - previous_range_high
        )

        down_distance = (
            previous_range_low
            - price
        )

        threshold = (
            multiplier * atr
        )

        if up_distance > threshold:

            return StrategyResult(
                BUY,
                _clip01(
                    up_distance
                    / max(
                        threshold * 2.0,
                        EPS,
                    )
                ),
                {
                    "atr": atr,
                    "distance": up_distance,
                },
            )

        if down_distance > threshold:

            return StrategyResult(
                SELL,
                _clip01(
                    down_distance
                    / max(
                        threshold * 2.0,
                        EPS,
                    )
                ),
                {
                    "atr": atr,
                    "distance": down_distance,
                },
            )

        return StrategyResult(
            HOLD,
            0.0,
            {"atr": atr},
        )


# ============================================================================
# Strategy 12: VWAP Mean Reversion
# ============================================================================

class VWAPStrategy(BaseStrategy):

    def __init__(
        self,
        period: int = 30,
        threshold_bps: float = 50.0,
    ):

        super().__init__(
            name=(
                f"VWAP("
                f"{period},"
                f"{threshold_bps})"
            ),
            params={
                "period": period,
                "threshold_bps": threshold_bps,
            },
            family="vwap_mean_reversion",
            priority=1.0,
            regime_bias="mean_reversion",
        )

    def analyze(
        self,
        candles: np.ndarray,
    ) -> StrategyResult:

        period = int(
            self.params["period"]
        )

        threshold_bps = float(
            self.params["threshold_bps"]
        )

        price = float(
            candles[-1, 3]
        )

        value = _vwap(
            candles,
            period,
        )

        if not np.isfinite(value):

            return StrategyResult(
                HOLD,
                0.0,
                {"reason": "insufficient_data"},
            )

        deviation_bps = (
            (
                price - value
            )
            / max(
                value,
                EPS,
            )
        ) * 10000.0

        if deviation_bps <= -threshold_bps:

            return StrategyResult(
                BUY,
                _clip01(
                    abs(deviation_bps)
                    / max(
                        threshold_bps * 3.0,
                        EPS,
                    )
                ),
                {
                    "vwap": value,
                    "deviation_bps": deviation_bps,
                },
            )

        if deviation_bps >= threshold_bps:

            return StrategyResult(
                SELL,
                _clip01(
                    abs(deviation_bps)
                    / max(
                        threshold_bps * 3.0,
                        EPS,
                    )
                ),
                {
                    "vwap": value,
                    "deviation_bps": deviation_bps,
                },
            )

        return StrategyResult(
            HOLD,
            0.0,
            {
                "vwap": value,
                "deviation_bps": deviation_bps,
            },
        )


# ============================================================================
# Strategy 13: Volume Pressure
# ============================================================================

class VolumePressureStrategy(BaseStrategy):

    def __init__(
        self,
        period: int = 20,
        threshold: float = 1.5,
    ):

        super().__init__(
            name=(
                f"VolumePressure("
                f"{period},"
                f"{threshold})"
            ),
            params={
                "period": period,
                "threshold": threshold,
            },
            family="volume",
            priority=1.0,
            regime_bias="neutral",
        )

    def analyze(
        self,
        candles: np.ndarray,
    ) -> StrategyResult:

        period = int(
            self.params["period"]
        )

        threshold = float(
            self.params["threshold"]
        )

        if len(candles) < period + 2:

            return StrategyResult(
                HOLD,
                0.0,
                {"reason": "insufficient_data"},
            )

        open_price = float(
            candles[-1, 0]
        )

        close_price = float(
            candles[-1, 3]
        )

        volume_ratio = _volume_ratio(
            candles[:, 4],
            period,
        )

        if not np.isfinite(
            volume_ratio
        ):

            return StrategyResult(
                HOLD,
                0.0,
                {"reason": "invalid_volume"},
            )

        body_return = (
            close_price
            - open_price
        ) / max(
            open_price,
            EPS,
        )

        if (
            volume_ratio >= threshold
            and body_return > 0
        ):

            confidence = _clip01(
                (
                    volume_ratio
                    / (
                        threshold * 2.0
                    )
                )
            )

            return StrategyResult(
                BUY,
                confidence,
                {
                    "volume_ratio": volume_ratio,
                    "body_return": body_return,
                },
            )

        if (
            volume_ratio >= threshold
            and body_return < 0
        ):

            confidence = _clip01(
                (
                    volume_ratio
                    / (
                        threshold * 2.0
                    )
                )
            )

            return StrategyResult(
                SELL,
                confidence,
                {
                    "volume_ratio": volume_ratio,
                    "body_return": body_return,
                },
            )

        return StrategyResult(
            HOLD,
            0.0,
            {
                "volume_ratio": volume_ratio,
            },
        )


# ============================================================================
# Regime
# ============================================================================

@dataclass
class RegimeState:

    regime: str

    trend_strength: float
    volatility_level: float

    ema_fast: float
    ema_slow: float
    atr: float

    diagnostics: Dict[str, Any] = field(
        default_factory=dict
    )


class RegimeDetector:

    """
    Lightweight rule-based regime detector.

    This is intentionally not HMM.

    HMM/GARCH can remain in their own files and feed
    an external regime state later.
    """

    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 50,
        atr_period: int = 14,
        history: int = 100,
    ):

        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self.history = history

    def detect(
        self,
        candles: np.ndarray,
    ) -> RegimeState:

        closes = candles[:, 3]

        fast = _ema(
            closes,
            self.fast_period,
        )

        slow = _ema(
            closes,
            self.slow_period,
        )

        atr = _atr(
            candles,
            self.atr_period,
        )

        price = float(
            closes[-1]
        )

        if not all(
            np.isfinite(
                [
                    fast,
                    slow,
                    atr,
                ]
            )
        ):

            return RegimeState(
                regime="unknown",
                trend_strength=0.0,
                volatility_level=0.0,
                ema_fast=fast,
                ema_slow=slow,
                atr=atr,
            )

        trend_strength = abs(
            fast - slow
        ) / max(
            atr,
            EPS,
        )

        returns = _returns(
            closes
        )

        if len(returns) >= self.history:

            vol = float(
                np.std(
                    returns[-self.history:]
                )
                * math.sqrt(1.0)
            )

        else:

            vol = float(
                np.std(
                    returns
                )
            ) if len(returns) else 0.0

        # Adaptive relative thresholds.
        if trend_strength >= 1.5:

            regime = (
                "trend_up"
                if fast > slow
                else "trend_down"
            )

        elif vol > 0:

            # Large trendless volatility.
            if trend_strength < 0.6:

                regime = "mean_reversion"

            else:

                regime = "transition"

        else:

            regime = "neutral"

        return RegimeState(
            regime=regime,
            trend_strength=float(
                trend_strength
            ),
            volatility_level=float(
                vol
            ),
            ema_fast=float(fast),
            ema_slow=float(slow),
            atr=float(atr),
            diagnostics={
                "price": price,
            },
        )


# ============================================================================
# Strategy Fitness
# ============================================================================

@dataclass
class FitnessState:

    observations: int = 0

    ewma_edge: float = 0.0

    ewma_win_rate: float = 0.5

    ewma_abs_return: float = 0.0

    last_signal: int = HOLD

    last_confidence: float = 0.0

    last_update_index: int = 0


class FitnessBook:

    """
    Online adaptive memory.

    It does NOT search the historical data for the best strategy.

    It waits for realized outcomes:

        signal_t
             ↓
        return_(t→t+k)
             ↓
        update()

    Then applies exponential forgetting.

    This is much safer than:
        "find strategy with highest backtest Sharpe
         and use it forever."

    A shrinkage term pulls every strategy toward neutral
    until enough observations exist.
    """

    def __init__(
        self,
        decay: float = 0.97,
        prior_strength: float = 25.0,
        edge_scale: float = 0.01,
        min_weight: float = 0.50,
        max_weight: float = 1.50,
    ):

        self.decay = float(
            np.clip(
                decay,
                0.5,
                0.9999,
            )
        )

        self.prior_strength = max(
            float(prior_strength),
            1.0,
        )

        self.edge_scale = max(
            float(edge_scale),
            EPS,
        )

        self.min_weight = float(
            min_weight
        )

        self.max_weight = float(
            max_weight
        )

        self.states: Dict[
            str,
            FitnessState
        ] = {}

    def state(
        self,
        name: str,
    ) -> FitnessState:

        if name not in self.states:

            self.states[name] = (
                FitnessState()
            )

        return self.states[name]

    def update(
        self,
        name: str,
        signal: int,
        realized_return: float,
        confidence: float = 1.0,
        strategy_volatility: float = 0.01,
    ) -> None:

        state = self.state(name)

        realized_return = _safe_float(
            realized_return
        )

        confidence = _clip01(
            confidence
        )

        strategy_volatility = max(
            abs(
                _safe_float(
                    strategy_volatility,
                    0.01,
                )
            ),
            self.edge_scale,
        )

        # Directional edge.
        edge = (
            float(signal)
            * realized_return
        )

        # Normalize by volatility so extremely volatile
        # strategies do not automatically look superior.
        normalized_edge = (
            edge
            / strategy_volatility
        )

        # Confidence weighting.
        normalized_edge *= (
            0.25
            + 0.75 * confidence
        )

        state.observations += 1

        state.ewma_edge = (
            self.decay
            * state.ewma_edge
            + (
                1.0 - self.decay
            )
            * normalized_edge
        )

        win = (
            1.0
            if edge > 0
            else 0.0
        )

        state.ewma_win_rate = (
            self.decay
            * state.ewma_win_rate
            + (
                1.0 - self.decay
            )
            * win
        )

        state.ewma_abs_return = (
            self.decay
            * state.ewma_abs_return
            + (
                1.0 - self.decay
            )
            * abs(realized_return)
        )

        state.last_signal = int(
            signal
        )

        state.last_confidence = (
            confidence
        )

    def weight(
        self,
        name: str,
    ) -> float:

        state = self.state(
            name
        )

        # Bayesian-like shrinkage:
        #
        # with low observation count,
        # weight remains close to 1.0.
        #
        # with more observations,
        # performance matters more.
        reliability = (
            state.observations
            / (
                state.observations
                + self.prior_strength
            )
        )

        adjusted_edge = (
            reliability
            * state.ewma_edge
        )

        raw = math.exp(
            np.clip(
                adjusted_edge
                / 5.0,
                -1.0,
                1.0,
            )
        )

        weight = (
            1.0
            + (
                raw - 1.0
            )
        )

        return float(
            np.clip(
                weight,
                self.min_weight,
                self.max_weight,
            )
        )

    def snapshot(
        self,
    ) -> Dict[str, Dict[str, float]]:

        result = {}

        for name, state in (
            self.states.items()
        ):

            result[name] = {
                "observations": float(
                    state.observations
                ),
                "ewma_edge": float(
                    state.ewma_edge
                ),
                "ewma_win_rate": float(
                    state.ewma_win_rate
                ),
                "weight": self.weight(
                    name
                ),
            }

        return result


# ============================================================================
# Swarm configuration
# ============================================================================

@dataclass
class SwarmConfig:

    minimum_data: int = 120

    minimum_confidence: float = 0.10

    decision_threshold: float = 0.20

    strong_threshold: float = 0.55

    temperature: float = 1.0

    trend_family_boost: float = 1.25

    mean_reversion_family_boost: float = 1.25

    breakout_family_boost: float = 1.15

    volume_family_weight: float = 0.85

    max_active_per_family: int = 20

    max_strategies: int = 250

    require_family_confirmation: int = 2

    learning_decay: float = 0.97

    learning_prior_strength: float = 25.0

    redundancy_penalty: float = 0.25

    history_length: int = 60


# ============================================================================
# Swarm Engine
# ============================================================================

class StrategySwarm:

    """
    Adaptive swarm controller.
    """

    def __init__(
        self,
        strategies: Sequence[BaseStrategy],
        config: Optional[SwarmConfig] = None,
        regime_detector: Optional[
            RegimeDetector
        ] = None,
    ):

        self.config = (
            config
            or SwarmConfig()
        )

        self.strategies = list(
            strategies
        )

        if not self.strategies:

            raise ValueError(
                "Swarm cannot be empty."
            )

        if len(
            self.strategies
        ) > self.config.max_strategies:

            self.strategies = (
                self.strategies[
                    :self.config.max_strategies
                ]
            )

        self.regime_detector = (
            regime_detector
            or RegimeDetector()
        )

        self.fitness = FitnessBook(
            decay=self.config.learning_decay,
            prior_strength=(
                self.config.learning_prior_strength
            ),
        )

        self.signal_history: Dict[
            str,
            Deque[float]
        ] = defaultdict(
            lambda: deque(
                maxlen=self.config.history_length
            )
        )

        self.family_history: Dict[
            str,
            Deque[float]
        ] = defaultdict(
            lambda: deque(
                maxlen=self.config.history_length
            )
        )

        self.last_results: Dict[
            str,
            StrategyResult
        ] = {}

        self.last_regime: Optional[
            RegimeState
        ] = None

        self.step_index = 0

    # ======================================================================
    # Validation
    # ======================================================================

    @staticmethod
    def validate_candles(
        candles: np.ndarray,
    ) -> np.ndarray:

        arr = np.asarray(
            candles,
            dtype=float,
        )

        if arr.ndim != 2:
            raise ValueError(
                "candles must be a 2D array."
            )

        if arr.shape[1] < 5:
            raise ValueError(
                "candles requires at least 5 columns: "
                "OHLCV"
            )

        arr = arr[:, :5]

        if len(arr) == 0:
            raise ValueError(
                "candles is empty."
            )

        if not np.all(
            np.isfinite(arr)
        ):
            raise ValueError(
                "candles contains NaN or Inf."
            )

        opens = arr[:, 0]
        highs = arr[:, 1]
        lows = arr[:, 2]
        closes = arr[:, 3]
        volumes = arr[:, 4]

        if np.any(
            opens <= 0
        ) or np.any(
            highs <= 0
        ) or np.any(
            lows <= 0
        ) or np.any(
            closes <= 0
        ):

            raise ValueError(
                "OHLC prices must be > 0."
            )

        if np.any(
            volumes < 0
        ):

            raise ValueError(
                "Volume cannot be negative."
            )

        if np.any(
            highs < lows
        ):

            raise ValueError(
                "High < Low detected."
            )

        if np.any(
            closes > highs
        ) or np.any(
            closes < lows
        ):

            raise ValueError(
                "Close outside High/Low range."
            )

        return arr

    # ======================================================================
    # Regime-dependent family weight
    # ======================================================================

    def _family_prior(
        self,
        family: str,
        regime: str,
    ) -> float:

        weight = 1.0

        if regime in {
            "trend_up",
            "trend_down",
        }:

            if family in {
                "trend",
                "momentum",
            }:

                weight *= (
                    self.config.trend_family_boost
                )

            if family in {
                "mean_reversion",
                "robust_mean_reversion",
                "vwap_mean_reversion",
            }:

                weight *= 0.75

            if family in {
                "breakout",
                "volatility_breakout",
            }:

                weight *= (
                    self.config.breakout_family_boost
                )

        elif regime == "mean_reversion":

            if family in {
                "mean_reversion",
                "robust_mean_reversion",
                "vwap_mean_reversion",
            }:

                weight *= (
                    self.config.mean_reversion_family_boost
                )

            if family in {
                "trend",
            }:

                weight *= 0.80

        elif regime == "transition":

            if family in {
                "volatility_breakout",
                "breakout",
            }:

                weight *= 1.10

            if family in {
                "mean_reversion",
                "robust_mean_reversion",
            }:

                weight *= 0.90

        if family == "volume":

            weight *= (
                self.config.volume_family_weight
            )

        return float(
            max(
                weight,
                0.01,
            )
        )

    # ======================================================================
    # Redundancy
    # ======================================================================

    def _redundancy_penalty(
        self,
        strategy_name: str,
    ) -> float:

        history = self.signal_history[
            strategy_name
        ]

        if len(history) < 15:
            return 1.0

        current = np.asarray(
            history,
            dtype=float,
        )

        correlations = []

        for (
            other_name,
            other_history,
        ) in self.signal_history.items():

            if (
                other_name
                == strategy_name
            ):
                continue

            if len(
                other_history
            ) != len(current):

                continue

            other = np.asarray(
                other_history,
                dtype=float,
            )

            if (
                np.std(current) <= EPS
                or np.std(other) <= EPS
            ):

                continue

            corr = np.corrcoef(
                current,
                other,
            )[0, 1]

            if np.isfinite(corr):

                correlations.append(
                    abs(
                        float(corr)
                    )
                )

        if not correlations:
            return 1.0

        redundancy = float(
            np.mean(
                correlations
            )
        )

        penalty = (
            1.0
            - (
                self.config.redundancy_penalty
                * redundancy
            )
        )

        return float(
            np.clip(
                penalty,
                0.50,
                1.00,
            )
        )

    # ======================================================================
    # Strategy evaluation
    # ======================================================================

    def _evaluate_strategy(
        self,
        strategy: BaseStrategy,
        candles: np.ndarray,
    ) -> StrategyResult:

        try:

            result = strategy.analyze(
                candles
            )

            if not isinstance(
                result,
                StrategyResult,
            ):

                raise TypeError(
                    (
                        f"{strategy.name} returned "
                        "invalid result type."
                    )
                )

            if result.signal == HOLD:

                return result

            if (
                result.confidence
                < self.config.minimum_confidence
            ):

                return StrategyResult(
                    HOLD,
                    0.0,
                    {
                        "suppressed": True,
                        "original_signal": (
                            result.signal
                        ),
                        **result.metadata,
                    },
                )

            return result

        except Exception as exc:

            log.exception(
                "[SWARM] Strategy failed: %s",
                strategy.name,
            )

            return StrategyResult(
                HOLD,
                0.0,
                {
                    "error": str(exc)
                },
            )

    # ======================================================================
    # Main analysis
    # ======================================================================

    def analyze(
        self,
        candles: np.ndarray,
    ) -> SwarmResult:

        candles = self.validate_candles(
            candles
        )

        if (
            len(candles)
            < self.config.minimum_data
        ):

            return SwarmResult(
                signal=HOLD,
                confidence=0.0,
                score=0.0,
                regime="insufficient_data",
                breadth=0.0,
                active_strategies=0,
                family_scores={},
                strategy_scores={},
                diagnostics={
                    "reason": "insufficient_data",
                    "required": (
                        self.config.minimum_data
                    ),
                    "available": len(candles),
                },
            )

        self.step_index += 1

        regime_state = (
            self.regime_detector.detect(
                candles
            )
        )

        self.last_regime = (
            regime_state
        )

        results: Dict[
            str,
            StrategyResult
        ] = {}

        # --------------------------------------------------------------
        # Evaluate every strategy.
        # --------------------------------------------------------------

        for strategy in self.strategies:

            result = (
                self._evaluate_strategy(
                    strategy,
                    candles,
                )
            )

            results[
                strategy.name
            ] = result

            # Store signal history as signed confidence.
            self.signal_history[
                strategy.name
            ].append(
                result.signed_score
            )

        self.last_results = results

        # --------------------------------------------------------------
        # Group by family.
        # --------------------------------------------------------------

        family_members: Dict[
            str,
            List[
                Tuple[
                    BaseStrategy,
                    StrategyResult,
                ]
            ]
        ] = defaultdict(list)

        strategy_by_name = {
            strategy.name: strategy
            for strategy in self.strategies
        }

        for (
            name,
            result,
        ) in results.items():

            if result.signal == HOLD:
                continue

            strategy = (
                strategy_by_name[name]
            )

            family_members[
                strategy.family
            ].append(
                (
                    strategy,
                    result,
                )
            )

        # --------------------------------------------------------------
        # Family-level normalized aggregation.
        # --------------------------------------------------------------

        family_scores: Dict[
            str,
            float
        ] = {}

        strategy_scores: Dict[
            str,
            float
        ] = {}

        family_effective_weight: Dict[
            str,
            float
        ] = {}

        for (
            family,
            members,
        ) in family_members.items():

            # Only keep top active members by current absolute score
            # so dozens of nearly identical variants don't dominate.
            members = sorted(
                members,
                key=lambda item: abs(
                    item[1].signed_score
                ),
                reverse=True,
            )[
                :self.config.max_active_per_family
            ]

            weighted_scores = []
            total_weight = 0.0

            for (
                strategy,
                result,
            ) in members:

                adaptive_weight = (
                    self.fitness.weight(
                        strategy.name
                    )
                )

                redundancy = (
                    self._redundancy_penalty(
                        strategy.name
                    )
                )

                weight = (
                    strategy.priority
                    * adaptive_weight
                    * redundancy
                )

                score = (
                    result.signed_score
                    * weight
                )

                weighted_scores.append(
                    score
                )

                total_weight += weight

                strategy_scores[
                    strategy.name
                ] = score

            if total_weight <= EPS:
                continue

            family_score = (
                sum(weighted_scores)
                / total_weight
            )

            # Family prior according to regime.
            family_prior = (
                self._family_prior(
                    family,
                    regime_state.regime,
                )
            )

            family_scores[
                family
            ] = (
                family_score
                * family_prior
            )

            family_effective_weight[
                family
            ] = family_prior

            self.family_history[
                family
            ].append(
                family_scores[
                    family
                ]
            )

        # --------------------------------------------------------------
        # Require independent family confirmation.
        # --------------------------------------------------------------

        nonzero_family_scores = [
            score
            for score in
            family_scores.values()
            if abs(score)
            >= self.config.decision_threshold
        ]

        bullish_families = sum(
            1
            for score in nonzero_family_scores
            if score > 0
        )

        bearish_families = sum(
            1
            for score in nonzero_family_scores
            if score < 0
        )

        # --------------------------------------------------------------
        # Family weighted average.
        # --------------------------------------------------------------

        if family_scores:

            numerator = 0.0
            denominator = 0.0

            for (
                family,
                score,
            ) in family_scores.items():

                prior = (
                    family_effective_weight[
                        family
                    ]
                )

                numerator += (
                    score * prior
                )

                denominator += abs(
                    prior
                )

            aggregate_score = (
                numerator
                / max(
                    denominator,
                    EPS,
                )
            )

        else:

            aggregate_score = 0.0

        # --------------------------------------------------------------
        # Consensus breadth.
        # --------------------------------------------------------------

        active_count = sum(
            1
            for result in results.values()
            if result.signal != HOLD
        )

        if active_count > 0:

            positive = sum(
                1
                for result in results.values()
                if (
                    result.signal == BUY
                    and result.confidence
                    >= self.config.minimum_confidence
                )
            )

            negative = sum(
                1
                for result in results.values()
                if (
                    result.signal == SELL
                    and result.confidence
                    >= self.config.minimum_confidence
                )
            )

            breadth = (
                abs(
                    positive
                    - negative
                )
                / active_count
            )

        else:

            breadth = 0.0

        # --------------------------------------------------------------
        # Final decision.
        # --------------------------------------------------------------

        final_signal = HOLD

        if (
            aggregate_score
            >= self.config.decision_threshold
            and bullish_families
            >= self.config.require_family_confirmation
        ):

            final_signal = BUY

        elif (
            aggregate_score
            <= -self.config.decision_threshold
            and bearish_families
            >= self.config.require_family_confirmation
        ):

            final_signal = SELL

        # --------------------------------------------------------------
        # Confidence is NOT simply the raw score.
        #
        # It incorporates:
        #   signal magnitude
        #   breadth
        #   family confirmation
        # --------------------------------------------------------------

        family_count = max(
            len(family_scores),
            1,
        )

        confirmation = min(
            max(
                bullish_families
                if final_signal == BUY
                else bearish_families
                if final_signal == SELL
                else 0,
                0,
            )
            / max(
                family_count,
                1,
            ),
            1.0,
        )

        magnitude = min(
            abs(
                aggregate_score
            ),
            1.0,
        )

        confidence = _clip01(
            0.45 * magnitude
            + 0.35 * breadth
            + 0.20 * confirmation
        )

        # --------------------------------------------------------------
        # Strong signal state.
        # --------------------------------------------------------------

        if abs(
            aggregate_score
        ) >= self.config.strong_threshold:

            confidence = max(
                confidence,
                0.60,
            )

        # --------------------------------------------------------------
        # Diagnostics.
        # --------------------------------------------------------------

        diagnostics = {

            "trend_strength": (
                regime_state.trend_strength
            ),

            "volatility_level": (
                regime_state.volatility_level
            ),

            "ema_fast": (
                regime_state.ema_fast
            ),

            "ema_slow": (
                regime_state.ema_slow
            ),

            "atr": (
                regime_state.atr
            ),

            "bullish_families": (
                bullish_families
            ),

            "bearish_families": (
                bearish_families
            ),

            "family_count": (
                len(family_scores)
            ),

            "strategy_count": (
                len(self.strategies)
            ),

            "active_strategy_count": (
                active_count
            ),

            "regime": (
                regime_state.regime
            ),
        }

        return SwarmResult(
            signal=final_signal,
            confidence=float(
                confidence
            ),
            score=float(
                aggregate_score
            ),
            regime=(
                regime_state.regime
            ),
            breadth=float(
                breadth
            ),
            active_strategies=(
                active_count
            ),
            family_scores={
                key: float(value)
                for key, value
                in family_scores.items()
            },
            strategy_scores={
                key: float(value)
                for key, value
                in strategy_scores.items()
            },
            diagnostics=diagnostics,
        )

    # ======================================================================
    # Online feedback
    # ======================================================================

    def update(
        self,
        realized_return: float,
        horizon: int = 1,
        confidence_floor: float = 0.0,
        strategy_volatility: float = 0.01,
    ) -> None:

        """
        Update all strategies from realized forward return.

        IMPORTANT:
        ----------
        This method must be called ONLY when the future return
        is actually known.

        Example:

            t:
                analyze()

            t + N:
                realized_return = P[t+N] / P[t] - 1

                update(
                    realized_return
                )

        Never call update() with the same current candle's return.
        That would destroy temporal integrity.
        """

        realized_return = _safe_float(
            realized_return
        )

        for strategy in self.strategies:

            result = self.last_results.get(
                strategy.name
            )

            if result is None:
                continue

            if result.signal == HOLD:
                continue

            if (
                result.confidence
                < confidence_floor
            ):
                continue

            self.fitness.update(
                name=strategy.name,
                signal=result.signal,
                realized_return=realized_return,
                confidence=result.confidence,
                strategy_volatility=(
                    strategy_volatility
                ),
            )

    # ======================================================================
    # Fitness snapshot
    # ======================================================================

    def fitness_snapshot(
        self,
    ) -> Dict[str, Dict[str, float]]:

        return self.fitness.snapshot()


# ============================================================================
# Factory
# ============================================================================

def create_swarm() -> List[BaseStrategy]:
    """
    Create a diversified strategy population.

    The exact number intentionally comes from multiple strategy families,
    not 50 copies of the same mathematical idea.
    """

    swarm: List[
        BaseStrategy
    ] = []

    # ----------------------------------------------------------------------
    # 1. Standard Z-score
    # ----------------------------------------------------------------------

    for period in [
        10,
        15,
        20,
        30,
        40,
        50,
    ]:

        for threshold in [
            1.5,
            1.8,
            2.0,
            2.2,
            2.5,
        ]:

            swarm.append(
                ZScoreStrategy(
                    period=period,
                    threshold=threshold,
                )
            )

    # ----------------------------------------------------------------------
    # 2. Robust Z-score
    # ----------------------------------------------------------------------

    for period in [
        20,
        30,
        40,
        60,
    ]:

        for threshold in [
            2.0,
            2.5,
            3.0,
        ]:

            swarm.append(
                RobustZScoreStrategy(
                    period=period,
                    threshold=threshold,
                )
            )

    # ----------------------------------------------------------------------
    # 3. RSI
    # ----------------------------------------------------------------------

    for period in [
        7,
        10,
        14,
        21,
        28,
    ]:

        for oversold, overbought in [
            (20, 80),
            (25, 75),
            (30, 70),
        ]:

            swarm.append(
                RSIStrategy(
                    period=period,
                    oversold=oversold,
                    overbought=overbought,
                )
            )

    # ----------------------------------------------------------------------
    # 4. EMA Trend
    # ----------------------------------------------------------------------

    for fast, slow in [
        (5, 20),
        (8, 21),
        (10, 30),
        (15, 40),
        (20, 50),
        (20, 100),
    ]:

        swarm.append(
            MACrossoverStrategy(
                fast=fast,
                slow=slow,
            )
        )

    # ----------------------------------------------------------------------
    # 5. Bollinger
    # ----------------------------------------------------------------------

    for period in [
        15,
        20,
        25,
        30,
    ]:

        for num_std in [
            1.5,
            2.0,
            2.5,
        ]:

            swarm.append(
                BollingerStrategy(
                    period=period,
                    num_std=num_std,
                )
            )

    # ----------------------------------------------------------------------
    # 6. Donchian
    # ----------------------------------------------------------------------

    for period in [
        10,
        20,
        30,
        40,
        55,
        80,
    ]:

        swarm.append(
            DonchianStrategy(
                period=period
            )
        )

    # ----------------------------------------------------------------------
    # 7. Momentum
    # ----------------------------------------------------------------------

    for period in [
        5,
        10,
        20,
        30,
        50,
    ]:

        for threshold in [
            0.005,
            0.01,
            0.02,
        ]:

            swarm.append(
                MomentumStrategy(
                    period=period,
                    threshold=threshold,
                )
            )

    # ----------------------------------------------------------------------
    # 8. MACD
    # ----------------------------------------------------------------------

    for fast, slow, signal in [
        (8, 21, 5),
        (12, 26, 9),
        (16, 34, 9),
        (19, 39, 9),
    ]:

        swarm.append(
            MACDStrategy(
                fast=fast,
                slow=slow,
                signal=signal,
            )
        )

    # ----------------------------------------------------------------------
    # 9. Stochastic
    # ----------------------------------------------------------------------

    for period in [
        9,
        14,
        21,
    ]:

        for oversold, overbought in [
            (20, 80),
            (25, 75),
        ]:

            swarm.append(
                StochasticStrategy(
                    period=period,
                    oversold=oversold,
                    overbought=overbought,
                )
            )

    # ----------------------------------------------------------------------
    # 10. CCI
    # ----------------------------------------------------------------------

    for period in [
        14,
        20,
        30,
        40,
    ]:

        for threshold in [
            100,
            150,
        ]:

            swarm.append(
                CCIStrategy(
                    period=period,
                    threshold=threshold,
                )
            )

    # ----------------------------------------------------------------------
    # 11. ATR Breakout
    # ----------------------------------------------------------------------

    for period in [
        10,
        14,
        20,
        30,
    ]:

        for multiplier in [
            1.0,
            1.5,
            2.0,
        ]:

            swarm.append(
                ATRBreakoutStrategy(
                    period=period,
                    multiplier=multiplier,
                )
            )

    # ----------------------------------------------------------------------
    # 12. VWAP
    # ----------------------------------------------------------------------

    for period in [
        20,
        30,
        50,
        80,
    ]:

        for threshold in [
            25,
            50,
            75,
            100,
        ]:

            swarm.append(
                VWAPStrategy(
                    period=period,
                    threshold_bps=threshold,
                )
            )

    # ----------------------------------------------------------------------
    # 13. Volume
    # ----------------------------------------------------------------------

    for period in [
        10,
        20,
        30,
    ]:

        for threshold in [
            1.25,
            1.50,
            2.00,
        ]:

            swarm.append(
                VolumePressureStrategy(
                    period=period,
                    threshold=threshold,
                )
            )

    return swarm


# ============================================================================
# Utility
# ============================================================================

def swarm_summary(
    strategies: Sequence[BaseStrategy],
) -> Dict[str, int]:

    summary: Dict[
        str,
        int
    ] = defaultdict(int)

    for strategy in strategies:

        summary[
            strategy.family
        ] += 1

    return dict(
        sorted(
            summary.items()
        )
    )


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s "
            "[%(levelname)s] "
            "%(message)s"
        ),
    )

    # ----------------------------------------------------------------------
    # Generate synthetic OHLCV data.
    # ----------------------------------------------------------------------

    rng = np.random.default_rng(
        42
    )

    n = 500

    returns = (
        0.0002
        + rng.normal(
            0.0,
            0.008,
            n,
        )
    )

    close = (
        100.0
        * np.cumprod(
            1.0 + returns
        )
    )

    open_price = np.concatenate(
        [
            [close[0]],
            close[:-1],
        ]
    )

    spread = (
        np.abs(
            rng.normal(
                0.0,
                0.002,
                n,
            )
        )
        * close
    )

    high = (
        np.maximum(
            open_price,
            close,
        )
        + spread
    )

    low = (
        np.minimum(
            open_price,
            close,
        )
        - spread
    )

    volume = rng.lognormal(
        mean=8.0,
        sigma=0.5,
        size=n,
    )

    candles = np.column_stack(
        [
            open_price,
            high,
            low,
            close,
            volume,
        ]
    )

    # ----------------------------------------------------------------------
    # Build swarm.
    # ----------------------------------------------------------------------

    swarm = create_swarm()

    print(
        "=" * 72
    )

    print(
        f"SWARM SIZE: {len(swarm)}"
    )

    print(
        "FAMILIES:"
    )

    for family, count in (
        swarm_summary(
            swarm
        ).items()
    ):

        print(
            f"  {family:<28} {count}"
        )

    print(
        "=" * 72
    )

    # ----------------------------------------------------------------------
    # Create engine.
    # ----------------------------------------------------------------------

    engine = StrategySwarm(
        strategies=swarm,
        config=SwarmConfig(
            minimum_data=120,
            minimum_confidence=0.10,
            decision_threshold=0.20,
            strong_threshold=0.55,
            max_active_per_family=12,
        ),
    )

    # ----------------------------------------------------------------------
    # Analyze latest market.
    # ----------------------------------------------------------------------

    result = engine.analyze(
        candles
    )

    print(
        "\nFINAL RESULT"
    )

    print(
        "signal:",
        result.signal,
    )

    print(
        "confidence:",
        round(
            result.confidence,
            4,
        ),
    )

    print(
        "score:",
        round(
            result.score,
            4,
        ),
    )

    print(
        "regime:",
        result.regime,
    )

    print(
        "breadth:",
        round(
            result.breadth,
            4,
        ),
    )

    print(
        "\nFAMILY SCORES"
    )

    for family, score in sorted(
        result.family_scores.items(),
        key=lambda item: abs(
            item[1]
        ),
        reverse=True,
    ):

        print(
            f"  {family:<28} "
            f"{score:+.4f}"
        )

    # ----------------------------------------------------------------------
    # Example feedback.
    #
    # This is ONLY an example.
    # In production the return must be measured after
    # the configured forward horizon.
    # ----------------------------------------------------------------------

    future_return = (
        close[-1]
        / close[-2]
        - 1.0
    )

    engine.update(
        realized_return=float(
            future_return
        ),
        strategy_volatility=0.01,
    )

    print(
        "\nFITNESS SNAPSHOT"
    )

    snapshot = (
        engine.fitness_snapshot()
    )

    for name, state in list(
        snapshot.items()
    )[:10]:

        print(
            f"  {name:<32} "
            f"obs={int(state['observations']):>4} "
            f"edge={state['ewma_edge']:+.4f} "
            f"weight={state['weight']:.3f}"
        )
