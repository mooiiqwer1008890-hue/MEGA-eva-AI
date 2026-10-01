"""
hmm_regime.py
=============

Advanced Market Regime Detection Engine
using Gaussian Hidden Markov Models.

===========================================================
WHAT THIS MODULE DOES
===========================================================

يكتشف حالة السوق الكامنة (Market Regime) باستخدام HMM.

بدلاً من استخدام Log Return وحده، يستخدم النموذج عدة خصائص:

    1. Log Return
    2. Realized Volatility
    3. High-Low Range
    4. Volume Z-Score
    5. Momentum
    6. Return Z-Score

الحالات:

    n_states=2
        BEAR / BULL

    n_states=3
        BEAR / NEUTRAL / BULL

المخرجات المهمة:

    - current_regime
    - posterior probabilities
    - confidence
    - uncertainty
    - transition matrix
    - expected duration
    - regime persistence
    - state statistics
    - AIC / BIC
    - convergence status

===========================================================
IMPORTANT
===========================================================

HMM ليس نموذجاً مباشراً للتنبؤ بسعر BTC.

استخدامه الصحيح داخل نظام التداول:

    Market Data
          |
          v
    Feature Engineering
          |
          v
    HMM Regime Detection
          |
          +------> Regime Filter
          |
          +------> Risk Manager
          |
          +------> Position Sizing
          |
          v
    Signal Engine

ولا يجب أن يكون HMM وحده مسؤولاً عن Buy/Sell.

===========================================================
RESEARCH FOUNDATION
===========================================================

الفكرة الأساسية:

    Hamilton (1989)
    Markov Switching Models

والأدبيات الخاصة بـ HMM:

    Zucchini, MacDonald & Langrock
    Hidden Markov Models for Time Series

والتطبيقات المالية:

    Bhar & Hamori
    Hidden Markov Models:
    Applications to Financial Economics

وكذلك:

    Rossi & Gallo (2006)
    Volatility Estimation via Hidden Markov Models

===========================================================
RECOMMENDED
===========================================================

الفاصل:
    4H

عدد الشموع:
    400 - 1000

الحالات:
    3

مثال:

    detect_regimes(
        df,
        n_states=3
    )

===========================================================
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import requests

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


# =====================================================================
# LOGGER
# =====================================================================

log = logging.getLogger("hmm_regime")

warnings.filterwarnings("ignore")


# =====================================================================
# CONSTANTS
# =====================================================================

BINANCE_URL = "https://data-api.binance.vision/api/v3/klines"

DEFAULT_INTERVAL = "4h"
DEFAULT_LIMIT = 500

MIN_OBSERVATIONS = 150

DEFAULT_N_STATES = 3
DEFAULT_N_ITER = 2000
DEFAULT_ATTEMPTS = 5

EPS = 1e-12


# =====================================================================
# HELPERS
# =====================================================================

def _validate_series(series: pd.Series, name: str = "series") -> pd.Series:
    """
    تنظيف والتحقق من سلسلة رقمية.
    """

    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    result = pd.to_numeric(series, errors="coerce")

    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if result.empty:
        raise ValueError(f"{name} فارغة بعد التنظيف")

    return result.astype(float)


def _safe_zscore(series: pd.Series) -> pd.Series:
    """
    Z-score آمن.
    """

    std = float(series.std())

    if not np.isfinite(std) or std < EPS:
        return pd.Series(
            np.zeros(len(series)),
            index=series.index,
            dtype=float,
        )

    mean = float(series.mean())

    return (series - mean) / std


def _entropy(probabilities: np.ndarray) -> float:
    """
    Entropy normalized to [0,1].

    0:
        certainty عالية

    1:
        uncertainty عالية
    """

    p = np.asarray(probabilities, dtype=float)

    p = np.clip(p, EPS, 1.0)

    p = p / p.sum()

    if len(p) <= 1:
        return 0.0

    entropy = -np.sum(p * np.log(p))

    max_entropy = np.log(len(p))

    if max_entropy <= 0:
        return 0.0

    return float(entropy / max_entropy)


def _expected_duration(probability_stay: float) -> float:
    """
    Expected duration of a Markov state.

    E[D] = 1 / (1 - p_ii)

    إذا كان p_ii قريباً جداً من 1
    نعيد inf.
    """

    p = float(np.clip(probability_stay, 0.0, 1.0))

    if p >= 1.0 - EPS:
        return float("inf")

    if p <= 0:
        return 1.0

    return float(1.0 / (1.0 - p))


def _consecutive_count(states: np.ndarray, target_state: int) -> int:
    """
    عدد الشموع المتتالية التي بقيت فيها الحالة الحالية.
    """

    count = 0

    for state in reversed(states):
        if int(state) == int(target_state):
            count += 1
        else:
            break

    return count


# =====================================================================
# MARKET DATA
# =====================================================================

def fetch_ohlcv(
    symbol: str,
    interval: str = DEFAULT_INTERVAL,
    limit: int = DEFAULT_LIMIT,
    timeout: int = 20,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """
    جلب بيانات OHLCV من Binance.

    Parameters
    ----------
    symbol:
        مثال BTCUSDT

    interval:
        1h / 4h / 1d ...

    limit:
        عدد الشموع.

    drop_incomplete:
        حذف آخر شمعة إذا كانت لا تزال مفتوحة.

    Returns
    -------
    pd.DataFrame
    """

    symbol = str(symbol).upper().strip()

    if not symbol:
        raise ValueError("symbol فارغ")

    if limit < MIN_OBSERVATIONS:
        raise ValueError(
            f"limit يجب أن يكون >= {MIN_OBSERVATIONS}"
        )

    response = requests.get(
        BINANCE_URL,
        params={
            "symbol": symbol,
            "interval": interval,
            "limit": int(limit),
        },
        timeout=timeout,
    )

    response.raise_for_status()

    raw = response.json()

    if not isinstance(raw, list):
        raise ValueError("استجابة Binance غير صالحة")

    if len(raw) < MIN_OBSERVATIONS:
        raise ValueError(
            f"بيانات غير كافية: {len(raw)} شمعة"
        )

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]

    df = pd.DataFrame(
        raw,
        columns=columns,
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["timestamp"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True,
    )

    df = df[
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
        ]
    ].copy()

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if df.empty:
        raise ValueError("OHLCV فارغة بعد التنظيف")

    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------

    if (df["open"] <= 0).any():
        raise ValueError("وجد open <= 0")

    if (df["high"] <= 0).any():
        raise ValueError("وجد high <= 0")

    if (df["low"] <= 0).any():
        raise ValueError("وجد low <= 0")

    if (df["close"] <= 0).any():
        raise ValueError("وجد close <= 0")

    if (df["volume"] < 0).any():
        raise ValueError("وجد volume سالب")

    if (df["high"] < df["low"]).any():
        raise ValueError(
            "بيانات OHLC غير منطقية: high < low"
        )

    if (df["high"] < df["open"]).any():
        raise ValueError(
            "بيانات OHLC غير منطقية: high < open"
        )

    if (df["high"] < df["close"]).any():
        raise ValueError(
            "بيانات OHLC غير منطقية: high < close"
        )

    if (df["low"] > df["open"]).any():
        raise ValueError(
            "بيانات OHLC غير منطقية: low > open"
        )

    if (df["low"] > df["close"]).any():
        raise ValueError(
            "بيانات OHLC غير منطقية: low > close"
        )

    df = df.sort_values("timestamp")

    df = df.drop_duplicates(
        subset="timestamp",
        keep="last",
    )

    df = df.reset_index(drop=True)

    # --------------------------------------------------------------
    # Remove incomplete candle
    # --------------------------------------------------------------

    if drop_incomplete and len(df) >= 2:
        df = df.iloc[:-1].copy()

    if len(df) < MIN_OBSERVATIONS:
        raise ValueError(
            "عدد البيانات أصبح غير كافٍ بعد تنظيف الشموع"
        )

    return df


# =====================================================================
# BACKWARD COMPATIBILITY
# =====================================================================

def fetch_returns(
    symbol: str,
    interval: str = DEFAULT_INTERVAL,
    limit: int = DEFAULT_LIMIT,
) -> pd.Series:
    """
    وظيفة متوافقة مع النسخة القديمة.

    تعيد Log Returns فقط.
    """

    df = fetch_ohlcv(
        symbol=symbol,
        interval=interval,
        limit=limit,
    )

    returns = np.log(
        df["close"] / df["close"].shift(1)
    )

    returns = returns.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    returns.name = "log_return"

    return returns


# =====================================================================
# FEATURE ENGINEERING
# =====================================================================

def build_features(
    data: Union[pd.DataFrame, pd.Series],
    volatility_window: int = 24,
    momentum_window: int = 12,
    volume_window: int = 24,
) -> pd.DataFrame:
    """
    بناء خصائص HMM.

    إذا كان الإدخال DataFrame يحتوي OHLCV:
        يتم استخدام جميع الخصائص.

    إذا كان الإدخال Series:
        يتم استخدام خصائص العائد فقط.

    Features:

        return
        realized_vol
        range
        volume_z
        momentum
        return_z
    """

    if isinstance(data, pd.Series):

        returns = _validate_series(
            data,
            name="returns",
        )

        features = pd.DataFrame(
            index=returns.index
        )

        features["return"] = returns

        features["realized_vol"] = (
            returns
            .rolling(volatility_window)
            .std()
        )

        features["momentum"] = (
            returns
            .rolling(momentum_window)
            .sum()
        )

        historical_vol = (
            returns
            .rolling(volatility_window)
            .std()
            .shift(1)
        )

        features["return_z"] = (
            returns / historical_vol.clip(lower=EPS)
        )

    elif isinstance(data, pd.DataFrame):

        required = {
            "open",
            "high",
            "low",
            "close",
            "volume",
        }

        missing = required - set(data.columns)

        if missing:
            raise ValueError(
                f"OHLCV columns ناقصة: {sorted(missing)}"
            )

        df = data.copy()

        close = pd.to_numeric(
            df["close"],
            errors="coerce",
        )

        high = pd.to_numeric(
            df["high"],
            errors="coerce",
        )

        low = pd.to_numeric(
            df["low"],
            errors="coerce",
        )

        volume = pd.to_numeric(
            df["volume"],
            errors="coerce",
        )

        returns = np.log(
            close / close.shift(1)
        )

        features = pd.DataFrame(
            index=df.index
        )

        # ----------------------------------------------------------
        # 1. Return
        # ----------------------------------------------------------

        features["return"] = returns

        # ----------------------------------------------------------
        # 2. Realized volatility
        # ----------------------------------------------------------

        features["realized_vol"] = (
            returns
            .rolling(volatility_window)
            .std()
        )

        # ----------------------------------------------------------
        # 3. High-Low range
        # ----------------------------------------------------------

        features["range"] = np.log(
            high / low
        )

        # ----------------------------------------------------------
        # 4. Volume Z-score
        # ----------------------------------------------------------

        volume_mean = (
            volume
            .rolling(volume_window)
            .mean()
        )

        volume_std = (
            volume
            .rolling(volume_window)
            .std()
        )

        features["volume_z"] = (
            (volume - volume_mean)
            / volume_std.clip(lower=EPS)
        )

        # ----------------------------------------------------------
        # 5. Momentum
        # ----------------------------------------------------------

        features["momentum"] = np.log(
            close
            / close.shift(momentum_window)
        )

        # ----------------------------------------------------------
        # 6. Return Z-score
        # ----------------------------------------------------------

        previous_vol = (
            returns
            .rolling(volatility_window)
            .std()
            .shift(1)
        )

        features["return_z"] = (
            returns
            / previous_vol.clip(lower=EPS)
        )

    else:

        raise TypeError(
            "data يجب أن يكون pandas Series أو DataFrame"
        )

    # --------------------------------------------------------------
    # Numerical cleanup
    # --------------------------------------------------------------

    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    features = features.dropna()

    if features.empty:
        raise ValueError(
            "لا توجد features صالحة بعد الحساب"
        )

    # --------------------------------------------------------------
    # Conservative clipping for extreme observations
    #
    # This prevents one flash-move from dominating GaussianHMM.
    # --------------------------------------------------------------

    for column in features.columns:

        lower = features[column].quantile(0.01)
        upper = features[column].quantile(0.99)

        if np.isfinite(lower) and np.isfinite(upper):
            features[column] = features[column].clip(
                lower=lower,
                upper=upper,
            )

    return features


# =====================================================================
# LABEL STATES
# =====================================================================

def label_states(
    state_statistics: pd.DataFrame,
    n_states: int,
) -> Dict[int, str]:
    """
    تحويل أرقام الحالات العشوائية إلى أسماء اقتصادية.

    HMM لا يعرف مسبقاً أن state 0 = Bear.

    لذلك نقوم بترتيب الحالات حسب متوسط العائد.

    2 states:
        الأقل = Bear
        الأعلى = Bull

    3 states:
        الأقل = Bear
        الوسط = Neutral
        الأعلى = Bull
    """

    ordered_states = list(
        state_statistics
        .sort_values("mean_return")
        .index
    )

    labels: Dict[int, str] = {}

    if n_states == 2:

        labels[ordered_states[0]] = "BEAR"
        labels[ordered_states[1]] = "BULL"

    elif n_states == 3:

        labels[ordered_states[0]] = "BEAR"
        labels[ordered_states[1]] = "NEUTRAL"
        labels[ordered_states[2]] = "BULL"

    else:

        # Generalized labeling
        labels[ordered_states[0]] = "BEAR"
        labels[ordered_states[-1]] = "BULL"

        for state in ordered_states[1:-1]:
            labels[state] = f"NEUTRAL_{state}"

    return labels


# =====================================================================
# HMM FITTING
# =====================================================================

def detect_regimes(
    data: Union[pd.DataFrame, pd.Series],
    n_states: int = DEFAULT_N_STATES,
    n_iter: int = DEFAULT_N_ITER,
    attempts: int = DEFAULT_ATTEMPTS,
    random_state: int = 42,
    volatility_window: int = 24,
    momentum_window: int = 12,
    volume_window: int = 24,
) -> Dict[str, Any]:
    """
    تدريب Gaussian HMM واكتشاف الأنظمة.

    Parameters
    ----------
    data:
        DataFrame OHLCV أو Series returns.

    n_states:
        عدد الحالات.

    n_iter:
        الحد الأقصى لتكرارات EM.

        ملاحظة:
        n_iter الكبير لا "يضمن" convergence.

    attempts:
        عدد محاولات initialization.

    Returns
    -------
    dict
    """

    if n_states < 2:
        raise ValueError(
            "n_states يجب أن يكون >= 2"
        )

    if n_states > 5:
        raise ValueError(
            "n_states > 5 غير مستحسن لهذا النظام"
        )

    if n_iter < 100:
        raise ValueError(
            "n_iter يجب أن يكون >= 100"
        )

    if attempts < 1:
        raise ValueError(
            "attempts يجب أن يكون >= 1"
        )

    # --------------------------------------------------------------
    # Features
    # --------------------------------------------------------------

    features = build_features(
        data=data,
        volatility_window=volatility_window,
        momentum_window=momentum_window,
        volume_window=volume_window,
    )

    if len(features) < MIN_OBSERVATIONS:
        raise ValueError(
            f"عدد observations غير كافٍ: {len(features)}"
        )

    feature_names = list(features.columns)

    raw_X = features.values.astype(float)

    if not np.isfinite(raw_X).all():
        raise ValueError(
            "X يحتوي NaN أو Inf"
        )

    # --------------------------------------------------------------
    # Standardization
    # --------------------------------------------------------------

    scaler = StandardScaler()

    X = scaler.fit_transform(raw_X)

    if not np.isfinite(X).all():
        raise ValueError(
            "فشل StandardScaler: X غير صالح"
        )

    # --------------------------------------------------------------
    # Multiple random initializations
    # --------------------------------------------------------------

    best_model: Optional[GaussianHMM] = None
    best_score = -np.inf
    best_converged = False

    successful_models = 0

    for attempt in range(attempts):

        seed = int(random_state + attempt)

        try:

            model = GaussianHMM(
                n_components=n_states,
                covariance_type="diag",
                n_iter=n_iter,
                tol=1e-4,
                random_state=seed,
                min_covar=1e-4,
                verbose=False,
            )

            model.fit(X)

            score = float(
                model.score(X)
            )

            converged = bool(
                getattr(
                    model.monitor_,
                    "converged",
                    False,
                )
            )

            successful_models += 1

            log.info(
                "HMM attempt=%d score=%.4f converged=%s",
                attempt + 1,
                score,
                converged,
            )

            # Prefer higher likelihood.
            if score > best_score:

                best_score = score
                best_model = model
                best_converged = converged

        except Exception as exc:

            log.warning(
                "HMM attempt %d failed: %s",
                attempt + 1,
                exc,
            )

    if best_model is None:

        raise RuntimeError(
            "HMM فشل في جميع محاولات التدريب"
        )

    # --------------------------------------------------------------
    # Decode states
    # --------------------------------------------------------------

    hidden_states = best_model.predict(X)

    # True posterior probabilities
    posterior = best_model.predict_proba(X)

    if posterior.ndim != 2:
        raise RuntimeError(
            "predict_proba أعاد shape غير متوقع"
        )

    # --------------------------------------------------------------
    # State statistics in ORIGINAL feature space
    # --------------------------------------------------------------

    stats_rows = []

    for state in range(n_states):

        mask = hidden_states == state

        count = int(mask.sum())

        if count > 0:

            state_features = features.loc[mask]

            mean_return = float(
                state_features["return"].mean()
            )

            mean_vol = float(
                state_features["realized_vol"].mean()
            )

            mean_range = float(
                state_features["range"].mean()
            )

            mean_volume_z = float(
                state_features["volume_z"].mean()
            )

            mean_momentum = float(
                state_features["momentum"].mean()
            )

        else:

            mean_return = 0.0
            mean_vol = 0.0
            mean_range = 0.0
            mean_volume_z = 0.0
            mean_momentum = 0.0

        stats_rows.append(
            {
                "state": state,
                "count": count,
                "pct": count / len(hidden_states) * 100.0,
                "mean_return": mean_return,
                "mean_volatility": mean_vol,
                "mean_range": mean_range,
                "mean_volume_z": mean_volume_z,
                "mean_momentum": mean_momentum,
            }
        )

    state_statistics = (
        pd.DataFrame(stats_rows)
        .set_index("state")
    )

    labels = label_states(
        state_statistics,
        n_states=n_states,
    )

    state_statistics["label"] = [
        labels[state]
        for state in state_statistics.index
    ]

    # --------------------------------------------------------------
    # Current state
    # --------------------------------------------------------------

    current_state = int(
        hidden_states[-1]
    )

    current_probabilities = posterior[-1]

    current_probability = float(
        current_probabilities[current_state]
    )

    probability_map = {
        labels[i]: float(current_probabilities[i])
        for i in range(n_states)
    }

    # --------------------------------------------------------------
    # Uncertainty
    # --------------------------------------------------------------

    normalized_entropy = _entropy(
        current_probabilities
    )

    max_probability = float(
        np.max(current_probabilities)
    )

    confidence = max_probability

    # A conservative trust rule.
    trusted = bool(
        confidence >= 0.55
        and normalized_entropy <= 0.85
    )

    current_regime = labels[current_state]

    reported_regime = (
        current_regime
        if trusted
        else "UNKNOWN"
    )

    # --------------------------------------------------------------
    # State persistence
    # --------------------------------------------------------------

    persistence = _consecutive_count(
        hidden_states,
        current_state,
    )

    # --------------------------------------------------------------
    # Transition matrix
    # --------------------------------------------------------------

    transition_matrix = np.asarray(
        best_model.transmat_,
        dtype=float,
    )

    if transition_matrix.shape != (
        n_states,
        n_states,
    ):
        raise RuntimeError(
            "transition matrix shape غير صحيح"
        )

    # Ensure rows approximately sum to 1.
    transition_matrix = (
        transition_matrix
        / transition_matrix.sum(
            axis=1,
            keepdims=True,
        )
    )

    # Probability of remaining in current state.
    stay_probability = float(
        transition_matrix[
            current_state,
            current_state,
        ]
    )

    expected_duration = _expected_duration(
        stay_probability
    )

    # --------------------------------------------------------------
    # Next-state probabilities
    # --------------------------------------------------------------

    next_state_probability = (
        current_probabilities
        @ transition_matrix
    )

    next_state_probability = (
        next_state_probability
        / next_state_probability.sum()
    )

    next_regime_map = {
        labels[i]: float(
            next_state_probability[i]
        )
        for i in range(n_states)
    }

    # --------------------------------------------------------------
    # Regime change
    # --------------------------------------------------------------

    regime_changed = False

    if len(hidden_states) >= 2:
        regime_changed = bool(
            hidden_states[-1]
            != hidden_states[-2]
        )

    # --------------------------------------------------------------
    # Model quality metrics
    # --------------------------------------------------------------

    n_samples, n_features = X.shape

    # Gaussian diagonal HMM parameter count:
    #
    # transition probabilities:
    #     n_states * (n_states - 1)
    #
    # initial probabilities:
    #     n_states - 1
    #
    # means:
    #     n_states * n_features
    #
    # variances:
    #     n_states * n_features
    #
    n_parameters = (
        n_states * (n_states - 1)
        + (n_states - 1)
        + n_states * n_features
        + n_states * n_features
    )

    aic = (
        2.0 * n_parameters
        - 2.0 * best_score
    )

    bic = (
        n_parameters
        * np.log(n_samples)
        - 2.0 * best_score
    )

    # --------------------------------------------------------------
    # Model means/covariances
    # These are standardized-space parameters.
    # --------------------------------------------------------------

    means_scaled = np.asarray(
        best_model.means_,
        dtype=float,
    )

    variances_scaled = np.asarray(
        best_model.covars_,
        dtype=float,
    )

    # --------------------------------------------------------------
    # Logging
    # --------------------------------------------------------------

    log.info(
        "HMM completed | states=%d | "
        "score=%.4f | converged=%s",
        n_states,
        best_score,
        best_converged,
    )

    log.info(
        "Current regime=%s | state=%d | "
        "prob=%.2f%% | confidence=%.2f%%",
        reported_regime,
        current_state,
        current_probability * 100.0,
        confidence * 100.0,
    )

    return {
        # ----------------------------------------------------------
        # Core model
        # ----------------------------------------------------------

        "model": best_model,
        "scaler": scaler,
        "features": features,
        "feature_names": feature_names,

        # ----------------------------------------------------------
        # States
        # ----------------------------------------------------------

        "hidden_states": hidden_states,
        "current_state": current_state,
        "current_regime": current_regime,
        "reported_regime": reported_regime,

        # ----------------------------------------------------------
        # Probabilities
        # ----------------------------------------------------------

        "posterior_probabilities": posterior,
        "current_probabilities": current_probabilities,
        "regime_probabilities": probability_map,

        "current_probability": current_probability,
        "max_probability": max_probability,

        # ----------------------------------------------------------
        # Confidence / uncertainty
        # ----------------------------------------------------------

        "confidence": confidence,
        "entropy": normalized_entropy,
        "trusted": trusted,

        # ----------------------------------------------------------
        # Dynamics
        # ----------------------------------------------------------

        "transition_matrix": transition_matrix,
        "next_state_probability": next_state_probability,
        "next_regime_probabilities": next_regime_map,

        "stay_probability": stay_probability,
        "expected_duration": expected_duration,

        "state_persistence": persistence,
        "regime_changed": regime_changed,

        # ----------------------------------------------------------
        # State statistics
        # ----------------------------------------------------------

        "labels": labels,
        "state_statistics": state_statistics,

        # ----------------------------------------------------------
        # HMM parameters
        # ----------------------------------------------------------

        "means_scaled": means_scaled,
        "variances_scaled": variances_scaled,

        # ----------------------------------------------------------
        # Model quality
        # ----------------------------------------------------------

        "log_likelihood": best_score,
        "aic": float(aic),
        "bic": float(bic),
        "converged": best_converged,
        "successful_models": successful_models,
        "attempts": attempts,
        "n_states": n_states,
        "n_features": n_features,
        "n_observations": n_samples,
        "n_parameters": n_parameters,
        "n_iter": n_iter,
    }


# =====================================================================
# CURRENT REGIME
# =====================================================================

def get_current_regime(
    data: Union[pd.DataFrame, pd.Series],
    n_states: int = DEFAULT_N_STATES,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    إرجاع ملخص النظام الحالي.

    لا يعيد تدريب HMM أكثر من مرة داخل هذه العملية.
    """

    result = detect_regimes(
        data=data,
        n_states=n_states,
        **kwargs,
    )

    return {
        "current_state": result["current_state"],
        "current_regime": result["current_regime"],
        "reported_regime": result["reported_regime"],

        "regime_prob": result["current_probability"],
        "confidence": result["confidence"],
        "entropy": result["entropy"],
        "trusted": result["trusted"],

        "regime_probabilities": result[
            "regime_probabilities"
        ],

        "next_regime_probabilities": result[
            "next_regime_probabilities"
        ],

        "stay_probability": result[
            "stay_probability"
        ],

        "expected_duration": result[
            "expected_duration"
        ],

        "state_persistence": result[
            "state_persistence"
        ],

        "regime_changed": result[
            "regime_changed"
        ],

        "means": result[
            "state_statistics"
        ]["mean_return"].to_numpy(),

        "variances": result[
            "variances_scaled"
        ],

        "transition_matrix": result[
            "transition_matrix"
        ],

        "n_states": result["n_states"],

        "aic": result["aic"],
        "bic": result["bic"],
        "converged": result["converged"],
    }


# =====================================================================
# SIGNAL FILTER
# =====================================================================

def filter_signals_by_regime(
    signal: int,
    current_regime: str,
    confidence: float = 1.0,
    min_confidence: float = 0.55,
    trusted: Optional[bool] = None,
) -> int:
    """
    فلترة الإشارة بناءً على حالة السوق.

    signal:
        +1 = BUY / زيادة التعرض
         0 = HOLD
        -1 = SELL / تخفيض التعرض

    السياسة:

        UNKNOWN:
            لا دخول جديد.

        BEAR:
            BUY يتم حجبه.
            SELL يسمح به لتقليل التعرض.

        NEUTRAL:
            تمرير الإشارة.

        BULL:
            تمرير الإشارة.

    ملاحظة مهمة:
        SELL هنا يعني تخفيض التعرض في Spot.
        لا يعني فتح Short.
    """

    signal = int(np.sign(signal))

    current_regime = str(
        current_regime
    ).upper()

    confidence = float(
        np.clip(confidence, 0.0, 1.0)
    )

    if trusted is None:
        trusted = confidence >= min_confidence

    # --------------------------------------------------------------
    # Untrusted model
    # --------------------------------------------------------------

    if not trusted:
        log.info(
            "[HMM FILTER] blocked: model not trusted"
        )
        return 0

    if confidence < min_confidence:
        log.info(
            "[HMM FILTER] blocked: "
            "confidence %.2f%% < %.2f%%",
            confidence * 100,
            min_confidence * 100,
        )
        return 0

    # --------------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------------

    if current_regime == "UNKNOWN":
        if signal != 0:
            log.info(
                "[HMM FILTER] UNKNOWN -> signal blocked"
            )
        return 0

    # --------------------------------------------------------------
    # BEAR
    # --------------------------------------------------------------

    if current_regime == "BEAR":

        if signal == 1:

            log.info(
                "[HMM FILTER] BUY blocked "
                "(BEAR regime)"
            )

            return 0

        return signal

    # --------------------------------------------------------------
    # NEUTRAL
    # --------------------------------------------------------------

    if current_regime.startswith("NEUTRAL"):
        return signal

    # --------------------------------------------------------------
    # BULL
    # --------------------------------------------------------------

    if current_regime == "BULL":
        return signal

    # --------------------------------------------------------------
    # Unknown label
    # --------------------------------------------------------------

    log.warning(
        "[HMM FILTER] Unknown regime label: %s",
        current_regime,
    )

    return 0


# =====================================================================
# RISK MULTIPLIER
# =====================================================================

def regime_risk_multiplier(
    current_regime: str,
    confidence: float,
    min_multiplier: float = 0.0,
    max_multiplier: float = 1.0,
) -> float:
    """
    معامل إضافي لإدارة المخاطر.

    هذا ليس Position Size بحد ذاته.

    يمكن ضربه في base position size.

    مثال:

        final_size =
            base_size
            * garch_multiplier
            * regime_multiplier

    السياسة الافتراضية:

        BULL:
            1.00

        NEUTRAL:
            0.60

        BEAR:
            0.25

        UNKNOWN:
            0.00

    ثم يتم تعديل المعامل بناءً على confidence.
    """

    current_regime = str(
        current_regime
    ).upper()

    confidence = float(
        np.clip(confidence, 0.0, 1.0)
    )

    base = {
        "BULL": 1.00,
        "NEUTRAL": 0.60,
        "BEAR": 0.25,
        "UNKNOWN": 0.00,
    }.get(
        current_regime,
        0.0,
    )

    # Confidence gates exposure.
    #
    # confidence = 1.0
    #     full multiplier
    #
    # confidence = 0.5
    #     half multiplier

    multiplier = base * confidence

    return float(
        np.clip(
            multiplier,
            min_multiplier,
            max_multiplier,
        )
    )


# =====================================================================
# REGIME STATISTICS
# =====================================================================

def get_regime_stats(
    data: Union[pd.DataFrame, pd.Series],
    n_states: int = DEFAULT_N_STATES,
    **kwargs: Any,
) -> list:
    """
    إحصائيات تفصيلية لكل Regime.
    """

    result = detect_regimes(
        data=data,
        n_states=n_states,
        **kwargs,
    )

    state_statistics = (
        result["state_statistics"]
        .copy()
    )

    transition_matrix = result[
        "transition_matrix"
    ]

    hidden_states = result[
        "hidden_states"
    ]

    output = []

    for state in state_statistics.index:

        row = state_statistics.loc[state]

        stay_probability = float(
            transition_matrix[
                state,
                state,
            ]
        )

        duration = _expected_duration(
            stay_probability
        )

        persistence = _consecutive_count(
            hidden_states,
            int(state),
        )

        output.append(
            {
                "state": int(state),

                "label": str(
                    row["label"]
                ),

                "count": int(
                    row["count"]
                ),

                "pct": round(
                    float(row["pct"]),
                    2,
                ),

                "mean_return": float(
                    row["mean_return"]
                ),

                "mean_volatility": float(
                    row["mean_volatility"]
                ),

                "mean_range": float(
                    row["mean_range"]
                ),

                "mean_volume_z": float(
                    row["mean_volume_z"]
                ),

                "mean_momentum": float(
                    row["mean_momentum"]
                ),

                "stay_probability": round(
                    stay_probability,
                    6,
                ),

                "expected_duration": (
                    float(duration)
                    if np.isfinite(duration)
                    else float("inf")
                ),

                "current_persistence": int(
                    persistence
                ),
            }
        )

    return output


# =====================================================================
# SIMPLE SUMMARY
# =====================================================================

def summarize_regime(
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    ملخص صغير مناسب لإرساله إلى Telegram.
    """

    return {
        "regime": result[
            "reported_regime"
        ],

        "raw_regime": result[
            "current_regime"
        ],

        "confidence": round(
            float(result["confidence"]),
            4,
        ),

        "uncertainty": round(
            float(result["entropy"]),
            4,
        ),

        "trusted": bool(
            result["trusted"]
        ),

        "regime_probability": round(
            float(result[
                "current_probability"
            ]),
            4,
        ),

        "persistence": int(
            result[
                "state_persistence"
            ]
        ),

        "stay_probability": round(
            float(result[
                "stay_probability"
            ]),
            4,
        ),

        "expected_duration": (
            round(
                float(
                    result[
                        "expected_duration"
                    ]
                ),
                2,
            )
            if np.isfinite(
                result[
                    "expected_duration"
                ]
            )
            else None
        ),

        "regime_changed": bool(
            result[
                "regime_changed"
            ]
        ),

        "next_regime": max(
            result[
                "next_regime_probabilities"
            ],
            key=result[
                "next_regime_probabilities"
            ].get,
        ),

        "aic": round(
            float(result["aic"]),
            2,
        ),

        "bic": round(
            float(result["bic"]),
            2,
        ),

        "converged": bool(
            result["converged"]
        ),
    }


# =====================================================================
# TEST
# =====================================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s "
            "[%(levelname)s] "
            "%(message)s"
        ),
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "ADVANCED HMM MARKET REGIME ENGINE - 4H"
    )

    print(
        "=" * 78
        + "\n"
    )

    symbol = "BTCUSDT"

    try:

        # ----------------------------------------------------------
        # Fetch market data
        # ----------------------------------------------------------

        df = fetch_ohlcv(
            symbol=symbol,
            interval="4h",
            limit=500,
        )

        print(
            f"📊 Symbol: {symbol}"
        )

        print(
            f"📊 Candles: {len(df)}"
        )

        # ----------------------------------------------------------
        # Build and fit HMM
        # ----------------------------------------------------------

        result = detect_regimes(
            df,
            n_states=3,
            n_iter=2000,
            attempts=5,
            random_state=42,
        )

        # ----------------------------------------------------------
        # Summary
        # ----------------------------------------------------------

        summary = summarize_regime(
            result
        )

        print(
            "\n"
            + "-" * 78
        )

        print(
            "🎯 CURRENT MARKET REGIME"
        )

        print(
            "-" * 78
        )

        print(
            f"Regime           : "
            f"{summary['regime']}"
        )

        print(
            f"Raw Regime       : "
            f"{summary['raw_regime']}"
        )

        print(
            f"Confidence       : "
            f"{summary['confidence']:.2%}"
        )

        print(
            f"Uncertainty      : "
            f"{summary['uncertainty']:.2%}"
        )

        print(
            f"Trusted           : "
            f"{summary['trusted']}"
        )

        print(
            f"Probability       : "
            f"{summary['regime_probability']:.2%}"
        )

        print(
            f"Persistence       : "
            f"{summary['persistence']} candles"
        )

        print(
            f"Stay Probability  : "
            f"{summary['stay_probability']:.2%}"
        )

        print(
            f"Expected Duration : "
            f"{summary['expected_duration']}"
        )

        print(
            f"Regime Changed    : "
            f"{summary['regime_changed']}"
        )

        print(
            f"Next Regime       : "
            f"{summary['next_regime']}"
        )

        # ----------------------------------------------------------
        # Regime probabilities
        # ----------------------------------------------------------

        print(
            "\n"
            + "-" * 78
        )

        print(
            "📊 CURRENT REGIME PROBABILITIES"
        )

        print(
            "-" * 78
        )

        for regime, probability in (
            result[
                "regime_probabilities"
            ].items()
        ):

            print(
                f"{regime:<12}: "
                f"{probability:.2%}"
            )

        # ----------------------------------------------------------
        # Next regime probabilities
        # ----------------------------------------------------------

        print(
            "\n"
            + "-" * 78
        )

        print(
            "🔮 NEXT-STATE PROBABILITIES"
        )

        print(
            "-" * 78
        )

        for regime, probability in (
            result[
                "next_regime_probabilities"
            ].items()
        ):

            print(
                f"{regime:<12}: "
                f"{probability:.2%}"
            )

        # ----------------------------------------------------------
        # Transition matrix
        # ----------------------------------------------------------

        print(
            "\n"
            + "-" * 78
        )

        print(
            "🔄 TRANSITION MATRIX"
        )

        print(
            "-" * 78
        )

        matrix = result[
            "transition_matrix"
        ]

        labels = result["labels"]

        for i in range(
            result["n_states"]
        ):

            row_label = labels[i]

            values = " | ".join(
                [
                    f"{matrix[i, j]:.3f}"
                    for j in range(
                        result["n_states"]
                    )
                ]
            )

            print(
                f"{row_label:<10} -> "
                f"{values}"
            )

        # ----------------------------------------------------------
        # State statistics
        # ----------------------------------------------------------

        print(
            "\n"
            + "-" * 78
        )

        print(
            "📈 REGIME STATISTICS"
        )

        print(
            "-" * 78
        )

        stats = get_regime_stats(
            df,
            n_states=3,
            n_iter=2000,
            attempts=5,
            random_state=42,
        )

        for item in stats:

            print(
                f"\n"
                f"State      : "
                f"{item['state']}"
            )

            print(
                f"Label      : "
                f"{item['label']}"
            )

            print(
                f"Occurrences: "
                f"{item['count']}"
            )

            print(
                f"Share      : "
                f"{item['pct']:.2f}%"
            )

            print(
                f"Mean Return: "
                f"{item['mean_return']:.6f}"
            )

            print(
                f"Volatility : "
                f"{item['mean_volatility']:.6f}"
            )

            print(
                f"Range      : "
                f"{item['mean_range']:.6f}"
            )

            print(
                f"Volume Z   : "
                f"{item['mean_volume_z']:.4f}"
            )

            print(
                f"Momentum   : "
                f"{item['mean_momentum']:.6f}"
            )

            print(
                f"Stay Prob  : "
                f"{item['stay_probability']:.2%}"
            )

            print(
                f"Exp. Dur.  : "
                f"{item['expected_duration']}"
            )

        # ----------------------------------------------------------
        # Filter test
        # ----------------------------------------------------------

        print(
            "\n"
            + "-" * 78
        )

        print(
            "🛡 HMM SIGNAL FILTER TEST"
        )

        print(
            "-" * 78
        )

        for test_signal in [1, 0, -1]:

            filtered = filter_signals_by_regime(
                signal=test_signal,
                current_regime=(
                    result[
                        "reported_regime"
                    ]
                ),
                confidence=(
                    result[
                        "confidence"
                    ]
                ),
                min_confidence=0.55,
                trusted=result[
                    "trusted"
                ],
            )

            print(
                f"Signal "
                f"{test_signal:+d} "
                f"-> "
                f"{filtered:+d}"
            )

        # ----------------------------------------------------------
        # Risk multiplier
        # ----------------------------------------------------------

        multiplier = regime_risk_multiplier(
            current_regime=(
                result[
                    "reported_regime"
                ]
            ),
            confidence=(
                result[
                    "confidence"
                ]
            ),
        )

        print(
            "\n"
            f"🛡 Regime Risk Multiplier: "
            f"{multiplier:.4f}"
        )

        # ----------------------------------------------------------
        # Model metrics
        # ----------------------------------------------------------

        print(
            "\n"
            + "-" * 78
        )

        print(
            "📐 MODEL QUALITY"
        )

        print(
            "-" * 78
        )

        print(
            f"Converged       : "
            f"{result['converged']}"
        )

        print(
            f"Log-Likelihood  : "
            f"{result['log_likelihood']:.4f}"
        )

        print(
            f"AIC             : "
            f"{result['aic']:.2f}"
        )

        print(
            f"BIC             : "
            f"{result['bic']:.2f}"
        )

        print(
            f"Observations    : "
            f"{result['n_observations']}"
        )

        print(
            f"Features        : "
            f"{result['n_features']}"
        )

        print(
            f"Parameters      : "
            f"{result['n_parameters']}"
        )

        print(
            "\n"
            + "=" * 78
        )

        print(
            "HMM TEST COMPLETED"
        )

        print(
            "=" * 78
            + "\n"
        )

    except Exception as exc:

        log.exception(
            "HMM test failed: %s",
            exc,
        )

        print(
            "\n❌ HMM ERROR:"
        )

        print(
            str(exc)
        )
