
"""
feature_engineer.py
===================

Advanced OHLCV feature engineering for MEGA-eva-AI.

Design goals
------------
- No look-ahead information in FEATURES.
- Correct forward TARGET alignment.
- Wilder-style RSI / ATR.
- Mostly scale-free features (returns, ratios, distances).
- OHLC-based volatility estimators.
- Trend, momentum, volatility, volume and candle-structure features.
- Optional time features when timestamps are supplied.
- Clean separation between training labels and live inference.

Input
-----
NumPy array:
    [open, high, low, close, volume]

or DataFrame containing:
    open, high, low, close, volume

Optional timestamp column:
    timestamp / open_time / close_time / time

Output
------
DataFrame containing feature columns and, when include_target=True:
    future_return
    target

IMPORTANT
---------
This module does not prove predictive power.
Use chronological / walk-forward / purged validation later.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd


log = logging.getLogger("feature_engineer")

EPS = 1e-12


# ============================================================
# DATA NORMALIZATION
# ============================================================

def _normalize_candles(
    candles: Union[np.ndarray, pd.DataFrame],
) -> pd.DataFrame:
    """Normalize OHLCV input to a standard DataFrame."""

    if isinstance(candles, pd.DataFrame):
        df = candles.copy()

        rename = {}
        for col in df.columns:
            rename[col] = str(col).strip().lower()

        df = df.rename(columns=rename)

        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]

        if missing:
            raise ValueError(
                f"Missing OHLCV columns: {missing}"
            )

        return df.reset_index(drop=True)

    arr = np.asarray(candles)

    if arr.ndim != 2:
        raise ValueError("candles must be a 2D array.")

    if arr.shape[1] < 5:
        raise ValueError(
            "candles must contain at least 5 columns: "
            "[open, high, low, close, volume]"
        )

    return pd.DataFrame(
        arr[:, :5],
        columns=["open", "high", "low", "close", "volume"],
    )


def _validate_ohlcv(df: pd.DataFrame) -> None:
    """Validate and normalize OHLCV numeric columns in-place."""

    numeric = ["open", "high", "low", "close", "volume"]

    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[numeric].isna().any().any():
        raise ValueError(
            "OHLCV contains missing or non-numeric values."
        )

    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(
            "OHLC prices must be strictly positive."
        )

    if (df["volume"] < 0).any():
        raise ValueError("Volume cannot be negative.")

    if (
        df["high"]
        < df[["open", "close"]].max(axis=1)
    ).any():
        raise ValueError(
            "Invalid OHLC data: high < max(open, close)."
        )

    if (
        df["low"]
        > df[["open", "close"]].min(axis=1)
    ).any():
        raise ValueError(
            "Invalid OHLC data: low > min(open, close)."
        )


def _extract_time(
    df: pd.DataFrame,
) -> Optional[pd.Series]:
    """Find and parse an optional timestamp column."""

    candidates = [
        "timestamp",
        "open_time",
        "close_time",
        "time",
        "datetime",
        "date",
    ]

    for name in candidates:
        if name in df.columns:
            ts = df[name]

            if pd.api.types.is_numeric_dtype(ts):
                # Binance timestamps are generally milliseconds.
                median = pd.to_numeric(
                    ts,
                    errors="coerce",
                ).median()

                unit = "ms" if median > 1e11 else "s"

                parsed = pd.to_datetime(
                    ts,
                    unit=unit,
                    errors="coerce",
                    utc=True,
                )
            else:
                parsed = pd.to_datetime(
                    ts,
                    errors="coerce",
                    utc=True,
                )

            return parsed

    return None


# ============================================================
# BASIC RETURNS
# ============================================================

def log_returns(
    close: pd.Series,
) -> pd.Series:
    """One-bar logarithmic return."""

    return np.log(
        close / close.shift(1)
    )


def pct_return(
    close: pd.Series,
    period: int,
) -> pd.Series:
    """Percentage return over `period` bars."""

    return close.pct_change(period)


# ============================================================
# WILDER RSI
# ============================================================

def compute_rsi(
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Wilder RSI.

    Wilder smoothing corresponds to alpha = 1 / period.
    """

    if period < 2:
        raise ValueError("RSI period must be >= 2.")

    delta = close.diff()

    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    avg_gain = gains.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = losses.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rsi = pd.Series(
        np.nan,
        index=close.index,
        dtype=float,
    )

    both_zero = (
        (avg_gain == 0)
        & (avg_loss == 0)
    )

    only_gain = (
        (avg_gain > 0)
        & (avg_loss == 0)
    )

    normal = avg_loss > 0

    rsi.loc[both_zero] = 50.0
    rsi.loc[only_gain] = 100.0

    rs = (
        avg_gain.loc[normal]
        / avg_loss.loc[normal]
    )

    rsi.loc[normal] = (
        100.0 - 100.0 / (1.0 + rs)
    )

    return rsi


# ============================================================
# MOVING AVERAGES
# ============================================================

def compute_sma(
    close: pd.Series,
    period: int,
) -> pd.Series:

    return close.rolling(
        period,
        min_periods=period,
    ).mean()


def compute_ema(
    close: pd.Series,
    period: int,
) -> pd.Series:

    return close.ewm(
        span=period,
        adjust=False,
        min_periods=period,
    ).mean()


# ============================================================
# TRUE RANGE / ATR
# ============================================================

def compute_true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:

    tr = compute_true_range(
        high,
        low,
        close,
    )

    return tr.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()


# ============================================================
# VOLATILITY ESTIMATORS
# ============================================================

def compute_realized_volatility(
    close: pd.Series,
    period: int = 20,
) -> pd.Series:

    return (
        log_returns(close)
        .rolling(
            period,
            min_periods=period,
        )
        .std()
    )


def compute_parkinson_volatility(
    high: pd.Series,
    low: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Parkinson high-low volatility estimator."""

    hl = np.log(high / low)

    variance = (
        hl.pow(2)
        .rolling(
            period,
            min_periods=period,
        )
        .mean()
        / (4.0 * np.log(2.0))
    )

    return np.sqrt(
        variance.clip(lower=0.0)
    )


def compute_garman_klass_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Garman-Klass OHLC volatility estimator."""

    log_hl = np.log(high / low)
    log_co = np.log(close / open_)

    variance = (
        0.5 * log_hl.pow(2)
        - (2.0 * np.log(2.0) - 1.0)
        * log_co.pow(2)
    )

    rolling_variance = (
        variance
        .rolling(
            period,
            min_periods=period,
        )
        .mean()
    )

    return np.sqrt(
        rolling_variance.clip(lower=0.0)
    )


# ============================================================
# Z-SCORE
# ============================================================

def compute_zscore(
    series: pd.Series,
    period: int = 20,
) -> pd.Series:

    mean = series.rolling(
        period,
        min_periods=period,
    ).mean()

    std = series.rolling(
        period,
        min_periods=period,
    ).std()

    return (
        (series - mean)
        / std.replace(0.0, np.nan)
    )


# ============================================================
# MOMENTUM
# ============================================================

def compute_momentum(
    close: pd.Series,
    period: int,
) -> pd.Series:

    return (
        close / close.shift(period)
    ) - 1.0


# ============================================================
# MACD
# ============================================================

def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
):
    ema_fast = compute_ema(
        close,
        fast,
    )

    ema_slow = compute_ema(
        close,
        slow,
    )

    macd = ema_fast - ema_slow

    signal_line = compute_ema(
        macd,
        signal,
    )

    histogram = (
        macd - signal_line
    )

    return (
        macd,
        signal_line,
        histogram,
    )


# ============================================================
# BOLLINGER BANDS
# ============================================================

def compute_bollinger(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
):
    mid = close.rolling(
        period,
        min_periods=period,
    ).mean()

    std = close.rolling(
        period,
        min_periods=period,
    ).std()

    upper = mid + num_std * std
    lower = mid - num_std * std

    bandwidth = (
        upper - lower
    ) / mid.replace(
        0.0,
        np.nan,
    )

    percent_b = (
        close - lower
    ) / (
        upper - lower
    ).replace(
        0.0,
        np.nan,
    )

    return (
        mid,
        upper,
        lower,
        bandwidth,
        percent_b,
    )


# ============================================================
# STOCHASTIC
# ============================================================

def compute_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    smooth: int = 3,
):
    lowest = low.rolling(
        period,
        min_periods=period,
    ).min()

    highest = high.rolling(
        period,
        min_periods=period,
    ).max()

    percent_k = (
        100.0
        * (close - lowest)
        / (highest - lowest).replace(
            0.0,
            np.nan,
        )
    )

    percent_d = percent_k.rolling(
        smooth,
        min_periods=smooth,
    ).mean()

    return percent_k, percent_d


# ============================================================
# ADX / DIRECTIONAL MOVEMENT
# ============================================================

def compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
):
    """
    Wilder-style +DM, -DM, DI and ADX.
    """

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move)
            & (up_move > 0),
            up_move,
            0.0,
        ),
        index=high.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move)
            & (down_move > 0),
            down_move,
            0.0,
        ),
        index=high.index,
    )

    tr = compute_true_range(
        high,
        low,
        close,
    )

    atr = tr.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    plus_smoothed = plus_dm.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    minus_smoothed = minus_dm.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    plus_di = (
        100.0 * plus_smoothed
        / atr.replace(0.0, np.nan)
    )

    minus_di = (
        100.0 * minus_smoothed
        / atr.replace(0.0, np.nan)
    )

    denominator = (
        plus_di + minus_di
    ).replace(
        0.0,
        np.nan,
    )

    dx = (
        100.0
        * (plus_di - minus_di).abs()
        / denominator
    )

    adx = dx.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    return (
        plus_di,
        minus_di,
        adx,
    )


# ============================================================
# OBV
# ============================================================

def compute_obv(
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:

    direction = np.sign(
        close.diff()
    ).fillna(0.0)

    return (
        direction * volume
    ).cumsum()


# ============================================================
# VWAP
# ============================================================

def compute_rolling_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 20,
) -> pd.Series:

    typical_price = (
        high + low + close
    ) / 3.0

    numerator = (
        typical_price * volume
    ).rolling(
        period,
        min_periods=period,
    ).sum()

    denominator = volume.rolling(
        period,
        min_periods=period,
    ).sum()

    return (
        numerator
        / denominator.replace(0.0, np.nan)
    )


# ============================================================
# TREND / RANGE
# ============================================================

def compute_efficiency_ratio(
    close: pd.Series,
    period: int = 10,
) -> pd.Series:
    """
    Kaufman-style efficiency ratio:

        directional displacement
        --------------------------------
        sum of absolute bar movements
    """

    direction = (
        close - close.shift(period)
    ).abs()

    volatility = (
        close.diff()
        .abs()
        .rolling(
            period,
            min_periods=period,
        )
        .sum()
    )

    return (
        direction
        / volatility.replace(
            0.0,
            np.nan,
        )
    )


def compute_rolling_slope(
    series: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Rolling OLS slope.

    Applied to log(price) so the result represents
    approximate directional drift rather than raw-price units.
    """

    x = np.arange(period, dtype=float)
    x_mean = x.mean()
    denom = np.sum(
        (x - x_mean) ** 2
    )

    def _slope(values: np.ndarray) -> float:
        if not np.isfinite(values).all():
            return np.nan

        y = np.log(
            np.maximum(values, EPS)
        )

        y_mean = y.mean()

        return float(
            np.sum(
                (x - x_mean)
                * (y - y_mean)
            ) / denom
        )

    return series.rolling(
        period,
        min_periods=period,
    ).apply(
        _slope,
        raw=True,
    )


# ============================================================
# CANDLE STRUCTURE
# ============================================================

def add_candle_features(
    features: pd.DataFrame,
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> None:

    candle_range = (
        high - low
    ).clip(lower=EPS)

    body = (
        close - open_
    )

    features["candle_return"] = (
        close / open_
    ) - 1.0

    features["body_pct"] = (
        body.abs() / close
    )

    features["body_signed_pct"] = (
        body / close
    )

    features["upper_wick_pct"] = (
        high
        - pd.concat([open_, close], axis=1).max(axis=1)
    ) / close

    features["lower_wick_pct"] = (
        pd.concat([open_, close], axis=1).min(axis=1)
        - low
    ) / close

    features["range_pct"] = (
        candle_range / close
    )

    features["close_position"] = (
        close - low
    ) / candle_range

    features["body_to_range"] = (
        body.abs()
        / candle_range
    )

    features["direction"] = np.sign(body)


# ============================================================
# TIME FEATURES
# ============================================================

def add_time_features(
    features: pd.DataFrame,
    timestamps: Optional[pd.Series],
) -> None:

    if timestamps is None:
        return

    ts = pd.to_datetime(
        timestamps,
        errors="coerce",
        utc=True,
    )

    if ts.isna().all():
        return

    minute_of_day = (
        ts.dt.hour * 60
        + ts.dt.minute
    )

    day_of_week = ts.dt.dayofweek

    # Cyclical encoding avoids a false discontinuity
    # between 23:59 and 00:00.
    features["tod_sin"] = np.sin(
        2.0 * np.pi
        * minute_of_day / 1440.0
    )

    features["tod_cos"] = np.cos(
        2.0 * np.pi
        * minute_of_day / 1440.0
    )

    features["dow_sin"] = np.sin(
        2.0 * np.pi
        * day_of_week / 7.0
    )

    features["dow_cos"] = np.cos(
        2.0 * np.pi
        * day_of_week / 7.0
    )


# ============================================================
# FEATURE BUILDER
# ============================================================

def build_features(
    candles: Union[np.ndarray, pd.DataFrame],
    *,
    horizon: int = 1,
    target_threshold: float = 0.0,
    include_target: bool = True,
    include_time_features: bool = True,
    return_feature_groups: bool = False,
) -> Union[
    pd.DataFrame,
    tuple[pd.DataFrame, dict[str, list[str]]],
]:
    """
    Build an advanced ML dataset.

    Parameters
    ----------
    candles:
        OHLCV ndarray or DataFrame.

    horizon:
        Number of future bars used for the target.

    target_threshold:
        Minimum future return required for target=1.

        Example:
            0.0    -> any positive future return
            0.001  -> at least +0.10%

    include_target:
        True for training/backtesting datasets.
        False for live inference.

    include_time_features:
        Adds cyclical calendar features when timestamps exist.

    return_feature_groups:
        Also returns a dict grouping feature names.

    Returns
    -------
    DataFrame
        Feature matrix plus targets if requested.

    Notes
    -----
    No feature uses values after the current row.
    Targets are shifted into the future.
    """

    if horizon < 1:
        raise ValueError(
            "horizon must be >= 1."
        )

    if target_threshold < 0:
        raise ValueError(
            "target_threshold must be >= 0."
        )

    df = _normalize_candles(candles)

    timestamps = _extract_time(df)

    _validate_ohlcv(df)

    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    v = df["volume"].astype(float)

    features = pd.DataFrame(
        index=df.index
    )

    groups: dict[str, list[str]] = {}

    def add(
        group: str,
        name: str,
        values: pd.Series,
    ) -> None:
        features[name] = values
        groups.setdefault(group, []).append(name)

    # ========================================================
    # RETURNS
    # ========================================================

    ret_1 = log_returns(c)

    add(
        "returns",
        "ret_1",
        ret_1,
    )

    for p in (2, 3, 5, 10, 20):
        add(
            "returns",
            f"ret_{p}",
            c.pct_change(p),
        )

    # Recent return acceleration.
    add(
        "returns",
        "ret_accel_1_5",
        c.pct_change(1)
        - c.pct_change(5) / 5.0,
    )

    # ========================================================
    # LAGGED RETURNS
    # ========================================================

    for lag in (1, 2, 3, 5, 10):
        add(
            "lags",
            f"ret_lag_{lag}",
            ret_1.shift(lag),
        )

    # ========================================================
    # VOLATILITY
    # ========================================================

    for p in (10, 20, 50):
        add(
            "volatility",
            f"realized_vol_{p}",
            compute_realized_volatility(
                c,
                p,
            ),
        )

        add(
            "volatility",
            f"parkinson_vol_{p}",
            compute_parkinson_volatility(
                h,
                l,
                p,
            ),
        )

        add(
            "volatility",
            f"gk_vol_{p}",
            compute_garman_klass_volatility(
                o,
                h,
                l,
                c,
                p,
            ),
        )

    # ========================================================
    # ATR / TRUE RANGE
    # ========================================================

    tr = compute_true_range(
        h,
        l,
        c,
    )

    add(
        "range_volatility",
        "true_range_pct",
        tr / c,
    )

    for p in (14, 20, 50):
        atr = compute_atr(
            h,
            l,
            c,
            p,
        )

        add(
            "range_volatility",
            f"atr_pct_{p}",
            atr / c,
        )

    # ========================================================
    # RSI
    # ========================================================

    for p in (7, 14, 21):
        rsi = compute_rsi(
            c,
            p,
        )

        add(
            "momentum",
            f"rsi_{p}",
            rsi,
        )

        add(
            "momentum",
            f"rsi_{p}_centered",
            (rsi - 50.0) / 50.0,
        )

    # ========================================================
    # MOVING AVERAGES
    # ========================================================

    sma_10 = compute_sma(c, 10)
    sma_20 = compute_sma(c, 20)
    sma_50 = compute_sma(c, 50)
    sma_100 = compute_sma(c, 100)

    ema_12 = compute_ema(c, 12)
    ema_26 = compute_ema(c, 26)

    # Use relative distances rather than raw MA prices.
    add(
        "trend",
        "dist_sma10",
        c / sma_10 - 1.0,
    )

    add(
        "trend",
        "dist_sma20",
        c / sma_20 - 1.0,
    )

    add(
        "trend",
        "dist_sma50",
        c / sma_50 - 1.0,
    )

    add(
        "trend",
        "dist_sma100",
        c / sma_100 - 1.0,
    )

    add(
        "trend",
        "sma10_sma50_ratio",
        sma_10 / sma_50 - 1.0,
    )

    add(
        "trend",
        "sma20_sma50_ratio",
        sma_20 / sma_50 - 1.0,
    )

    add(
        "trend",
        "ema12_ema26_ratio",
        ema_12 / ema_26 - 1.0,
    )

    # ========================================================
    # Z-SCORE
    # ========================================================

    add(
        "mean_reversion",
        "zscore_20",
        compute_zscore(c, 20),
    )

    add(
        "mean_reversion",
        "zscore_50",
        compute_zscore(c, 50),
    )

    # ========================================================
    # MOMENTUM
    # ========================================================

    for p in (3, 5, 10, 20):
        add(
            "momentum",
            f"momentum_{p}",
            compute_momentum(c, p),
        )

    # ========================================================
    # MACD
    # ========================================================

    macd, macd_signal, macd_hist = (
        compute_macd(c)
    )

    add(
        "trend",
        "macd_pct",
        macd / c,
    )

    add(
        "trend",
        "macd_signal_pct",
        macd_signal / c,
    )

    add(
        "trend",
        "macd_hist_pct",
        macd_hist / c,
    )

    add(
        "trend",
        "macd_hist_change",
        macd_hist.diff(),
    )

    # ========================================================
    # BOLLINGER
    # ========================================================

    (
        bb_mid,
        bb_upper,
        bb_lower,
        bb_bandwidth,
        bb_percent_b,
    ) = compute_bollinger(c)

    add(
        "mean_reversion",
        "bb_distance_mid",
        c / bb_mid - 1.0,
    )

    add(
        "mean_reversion",
        "bb_bandwidth",
        bb_bandwidth,
    )

    add(
        "mean_reversion",
        "bb_percent_b",
        bb_percent_b,
    )

    # ========================================================
    # STOCHASTIC
    # ========================================================

    stoch_k, stoch_d = (
        compute_stochastic(
            h,
            l,
            c,
        )
    )

    add(
        "momentum",
        "stoch_k",
        stoch_k,
    )

    add(
        "momentum",
        "stoch_d",
        stoch_d,
    )

    add(
        "momentum",
        "stoch_kd_diff",
        stoch_k - stoch_d,
    )

    # ========================================================
    # ADX / DI
    # ========================================================

    plus_di, minus_di, adx = (
        compute_adx(
            h,
            l,
            c,
        )
    )

    add(
        "trend_strength",
        "plus_di",
        plus_di,
    )

    add(
        "trend_strength",
        "minus_di",
        minus_di,
    )

    add(
        "trend_strength",
        "adx",
        adx,
    )

    add(
        "trend_strength",
        "di_spread",
        (plus_di - minus_di) / 100.0,
    )

    # ========================================================
    # VOLUME
    # ========================================================

    volume_mean_20 = v.rolling(
        20,
        min_periods=20,
    ).mean()

    volume_std_20 = v.rolling(
        20,
        min_periods=20,
    ).std()

    add(
        "volume",
        "volume_log",
        np.log1p(v),
    )

    add(
        "volume",
        "volume_ratio_20",
        v / volume_mean_20,
    )

    add(
        "volume",
        "volume_zscore_20",
        (
            v - volume_mean_20
        ) / volume_std_20.replace(
            0.0,
            np.nan,
        ),
    )

    add(
        "volume",
        "volume_change_1",
        v.pct_change(),
    )

    add(
        "volume",
        "volume_momentum_5",
        v / v.shift(5) - 1.0,
    )

    # ========================================================
    # OBV
    # ========================================================

    obv = compute_obv(
        c,
        v,
    )

    obv_mean = obv.rolling(
        20,
        min_periods=20,
    ).mean()

    obv_std = obv.rolling(
        20,
        min_periods=20,
    ).std()

    add(
        "volume",
        "obv_zscore_20",
        (obv - obv_mean)
        / obv_std.replace(
            0.0,
            np.nan,
        ),
    )

    # ========================================================
    # ROLLING VWAP
    # ========================================================

    rvwap = compute_rolling_vwap(
        h,
        l,
        c,
        v,
        20,
    )

    add(
        "volume_price",
        "price_vs_rvwap_20",
        c / rvwap - 1.0,
    )

    # ========================================================
    # CANDLE MICROSTRUCTURE
    # ========================================================

    add_candle_features(
        features,
        o,
        h,
        l,
        c,
    )

    for col in (
        "candle_return",
        "body_pct",
        "body_signed_pct",
        "upper_wick_pct",
        "lower_wick_pct",
        "range_pct",
        "close_position",
        "body_to_range",
        "direction",
    ):
        groups.setdefault(
            "candle_structure",
            [],
        ).append(col)

    # ========================================================
    # RANGE POSITION / DONCHIAN
    # ========================================================

    for p in (20, 50):
        rolling_high = h.rolling(
            p,
            min_periods=p,
        ).max()

        rolling_low = l.rolling(
            p,
            min_periods=p,
        ).min()

        add(
            "range_structure",
            f"donchian_position_{p}",
            (
                c - rolling_low
            ) / (
                rolling_high - rolling_low
            ).replace(
                0.0,
                np.nan,
            ),
        )

        add(
            "range_structure",
            f"breakout_high_distance_{p}",
            c / rolling_high - 1.0,
        )

        add(
            "range_structure",
            f"breakout_low_distance_{p}",
            c / rolling_low - 1.0,
        )

    # ========================================================
    # TREND EFFICIENCY / SLOPE
    # ========================================================

    add(
        "trend_strength",
        "efficiency_ratio_10",
        compute_efficiency_ratio(
            c,
            10,
        ),
    )

    add(
        "trend_strength",
        "efficiency_ratio_20",
        compute_efficiency_ratio(
            c,
            20,
        ),
    )

    add(
        "trend_slope",
        "log_price_slope_20",
        compute_rolling_slope(
            c,
            20,
        ),
    )

    add(
        "trend_slope",
        "log_price_slope_50",
        compute_rolling_slope(
            c,
            50,
        ),
    )

    # ========================================================
    # DISTRIBUTION SHAPE
    # ========================================================

    returns = log_returns(c)

    for p in (20, 50):
        add(
            "return_distribution",
            f"return_skew_{p}",
            returns.rolling(
                p,
                min_periods=p,
            ).skew(),
        )

        add(
            "return_distribution",
            f"return_kurtosis_{p}",
            returns.rolling(
                p,
                min_periods=p,
            ).kurt(),
        )

    # ========================================================
    # OPTIONAL TIME FEATURES
    # ========================================================

    if include_time_features:
        before = set(features.columns)

        add_time_features(
            features,
            timestamps,
        )

        new_time_cols = [
            col for col in features.columns
            if col not in before
        ]

        groups.setdefault(
            "time",
            []
        ).extend(new_time_cols)

    # ========================================================
    # TARGET
    # ========================================================

    if include_target:
        future_return = (
            c.shift(-horizon)
            / c
        ) - 1.0

        target = pd.Series(
            np.nan,
            index=features.index,
            dtype=float,
        )

        valid_target = future_return.notna()

        target.loc[valid_target] = (
            future_return.loc[valid_target]
            > target_threshold
        ).astype(int)

        features["future_return"] = (
            future_return
        )

        features["target"] = target

        groups["target"] = [
            "future_return",
            "target",
        ]

    # ========================================================
    # CLEANING
    # ========================================================

    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # For training, rows without a valid target must disappear.
    # This correctly removes the final `horizon` rows rather than
    # silently converting them into class 0.
    if include_target:
        features = features.dropna(
            subset=["target"]
        )

    # Remove rows with incomplete indicators.
    features = features.dropna(
        axis=0,
        how="any",
    ).reset_index(drop=True)

    # Explicit dtype for ML libraries.
    for col in features.columns:
        features[col] = pd.to_numeric(
            features[col],
            errors="coerce",
        )

    if len(features) == 0:
        raise ValueError(
            "No usable rows remain after feature construction. "
            "Increase the candle history."
        )

    feature_columns = [
        col
        for col in features.columns
        if col not in {
            "target",
            "future_return",
        }
    ]

    log.info(
        "Built %d rows × %d features",
        len(features),
        len(feature_columns),
    )

    if include_target:
        rate = float(
            features["target"].mean()
        )

        log.info(
            "Target positive rate: %.2f%%",
            rate * 100.0,
        )

    if return_feature_groups:
        # Keep only columns that actually survived cleaning.
        surviving = set(features.columns)

        clean_groups = {}

        for group, cols in groups.items():
            clean_groups[group] = [
                col for col in cols
                if col in surviving
            ]

        return features, clean_groups

    return features


# ============================================================
# LIVE FEATURE VECTOR
# ============================================================

def build_live_features(
    candles: Union[np.ndarray, pd.DataFrame],
    **kwargs,
) -> pd.DataFrame:
    """
    Convenience wrapper for live inference.

    It never creates a target and therefore keeps the newest
    candle available for prediction.
    """

    kwargs["include_target"] = False

    return build_features(
        candles,
        **kwargs,
    )


# ============================================================
# SELF TEST
# ============================================================

def _synthetic_ohlcv(
    n: int = 600,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV only for unit testing.

    This data must never be used as a trading dataset.
    """

    rng = np.random.default_rng(seed)

    returns = rng.normal(
        0.0001,
        0.008,
        n,
    )

    close = (
        50000.0
        * np.exp(np.cumsum(returns))
    )

    open_ = (
        close
        * np.exp(
            rng.normal(
                0,
                0.002,
                n,
            )
        )
    )

    high = np.maximum(
        open_,
        close,
    ) * (
        1.0
        + rng.uniform(
            0,
            0.004,
            n,
        )
    )

    low = np.minimum(
        open_,
        close,
    ) * (
        1.0
        - rng.uniform(
            0,
            0.004,
            n,
        )
    )

    volume = rng.lognormal(
        mean=10.0,
        sigma=0.5,
        size=n,
    )

    timestamp = pd.date_range(
        "2025-01-01",
        periods=n,
        freq="15min",
        tz="UTC",
    )

    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def self_test() -> None:
    """Basic correctness tests."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s "
            "[%(levelname)s] "
            "%(message)s"
        ),
    )

    candles = _synthetic_ohlcv()

    dataset, groups = build_features(
        candles,
        horizon=1,
        target_threshold=0.0,
        include_target=True,
        include_time_features=True,
        return_feature_groups=True,
    )

    live = build_live_features(
        candles,
        include_time_features=True,
    )

    print("\n" + "=" * 80)
    print("MEGA-eva-AI Feature Engineering Self-Test")
    print("=" * 80)

    print("Training dataset shape:", dataset.shape)
    print("Live dataset shape:    ", live.shape)

    print(
        "Feature count:",
        len(
            [
                c
                for c in dataset.columns
                if c not in {
                    "target",
                    "future_return",
                }
            ]
        ),
    )

    print(
        "Target distribution:"
    )

    print(
        dataset["target"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "Positive rate: %.2f%%"
        % (
            dataset["target"].mean()
            * 100.0
        )
    )

    # Target cannot exist in live mode.
    assert "target" not in live.columns
    assert "future_return" not in live.columns

    # Training target must contain no NaN.
    assert dataset["target"].notna().all()

    # No infinities.
    assert np.isfinite(
        dataset.to_numpy(dtype=float)
    ).all()

    # A simple leakage check:
    # Changing the final close must not alter historical features
    # before the final bar.
    changed = candles.copy()
    last = len(changed) - 1
    changed.loc[last, "close"] *= 1.50

    # Keep the modified bar OHLC-valid.
    changed.loc[last, "high"] = (
        max(
            changed.loc[last, "open"],
            changed.loc[last, "close"],
        ) * 1.002
    )
    changed.loc[last, "low"] = (
        min(
            changed.loc[last, "open"],
            changed.loc[last, "close"],
        ) * 0.998
    )

    base = build_live_features(
        candles,
        include_time_features=False,
    )

    altered = build_live_features(
        changed,
        include_time_features=False,
    )

    common = min(
        len(base),
        len(altered),
    )

    if common > 5:
        historical_base = base.iloc[:-1]
        historical_alt = altered.iloc[:-1]

        numeric_cols = [
            c for c in historical_base.columns
            if c in historical_alt.columns
        ]

        np.testing.assert_allclose(
            historical_base[numeric_cols].to_numpy(),
            historical_alt[numeric_cols].to_numpy(),
            rtol=1e-10,
            atol=1e-10,
        )

    print("\nFeature groups:")
    for group, names in groups.items():
        print(
            f"  {group:22s}: {len(names)}"
        )

    print("\nFirst five rows:")
    print(dataset.head().to_string())

    print("\nSelf-test: PASSED")


if __name__ == "__main__":
    self_test()
