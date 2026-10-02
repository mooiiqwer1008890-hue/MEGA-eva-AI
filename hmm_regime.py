"""
hmm_regime.py
=============
MEGA Market Regime Detection Engine

Gaussian Hidden Markov Model (HMM) for market-regime classification.

Design goals
------------
1. Keep backward compatibility with the current MEGA runtime.
2. Fix the historical Series-input KeyError on ``range`` / ``volume_z``.
3. Prefer real OHLCV features whenever a DataFrame is supplied.
4. Remain usable when the caller only supplies log returns.
5. Return a rich, explicit result for downstream risk/filter layers.
6. Treat low-confidence regimes as UNKNOWN for safety.

Important
---------
HMM is a regime/context model, not a standalone Buy/Sell predictor.
A high-probability regime is not proof that the next price move will have
any particular direction.

Research references used in the design:
- Hamilton (1989), Markov Switching Models.
- Zucchini, MacDonald & Langrock, Hidden Markov Models for Time Series.
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
# LOGGER / CONSTANTS
# =====================================================================

log = logging.getLogger("hmm_regime")
warnings.filterwarnings("ignore")

BINANCE_URL = "https://data-api.binance.vision/api/v3/klines"

DEFAULT_INTERVAL = "4h"
DEFAULT_LIMIT = 500
DEFAULT_N_STATES = 3
DEFAULT_N_ITER = 800
DEFAULT_ATTEMPTS = 5

MIN_OBSERVATIONS = 150
EPS = 1e-12

# Confidence / uncertainty policy.
DEFAULT_MIN_CONFIDENCE = 0.55
DEFAULT_MAX_ENTROPY = 0.85

# Quantile clipping reduces the influence of extreme observations without
# deleting rows from the sequence.
CLIP_LOW = 0.01
CLIP_HIGH = 0.99


# =====================================================================
# NUMERIC HELPERS
# =====================================================================

def _validate_series(series: pd.Series, name: str = "series") -> pd.Series:
    """Return a clean finite float Series."""
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    result = pd.to_numeric(series, errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan).dropna()

    if result.empty:
        raise ValueError(f"{name} فارغة بعد التنظيف")

    return result.astype(float)


def _safe_zscore(series: pd.Series) -> pd.Series:
    """Cross-sectional/time-series safe z-score."""
    std = float(series.std())
    if not np.isfinite(std) or std < EPS:
        return pd.Series(0.0, index=series.index, dtype=float)
    return (series - float(series.mean())) / std


def _entropy(probabilities: np.ndarray) -> float:
    """Normalized Shannon entropy in [0, 1]."""
    p = np.asarray(probabilities, dtype=float)
    p = np.clip(p, EPS, None)
    total = float(p.sum())
    if total <= EPS:
        return 1.0
    p = p / total

    if len(p) <= 1:
        return 0.0

    value = float(-np.sum(p * np.log(p)))
    maximum = float(np.log(len(p)))
    return float(np.clip(value / maximum, 0.0, 1.0)) if maximum > 0 else 0.0


def _expected_duration(probability_stay: float) -> float:
    """Expected number of observations before leaving a state."""
    p = float(np.clip(probability_stay, 0.0, 1.0))
    if p >= 1.0 - EPS:
        return float("inf")
    return float(1.0 / (1.0 - p))


def _consecutive_count(states: np.ndarray, target_state: int) -> int:
    """Number of consecutive observations in target_state at the end."""
    count = 0
    for state in reversed(states):
        if int(state) != int(target_state):
            break
        count += 1
    return count


def _validate_window(value: int, name: str) -> int:
    value = int(value)
    if value < 2:
        raise ValueError(f"{name} يجب أن يكون >= 2")
    return value


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
    """Fetch and validate Binance OHLCV candles."""
    symbol = str(symbol).upper().strip()
    if not symbol:
        raise ValueError("symbol فارغ")

    limit = int(limit)
    if limit < MIN_OBSERVATIONS:
        raise ValueError(f"limit يجب أن يكون >= {MIN_OBSERVATIONS}")

    response = requests.get(
        BINANCE_URL,
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=int(timeout),
    )
    response.raise_for_status()

    raw = response.json()
    if not isinstance(raw, list):
        raise ValueError("استجابة Binance غير صالحة")
    if len(raw) < MIN_OBSERVATIONS:
        raise ValueError(f"بيانات غير كافية: {len(raw)} شمعة")

    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(raw, columns=columns)

    numeric = ["open", "high", "low", "close", "volume", "quote_volume"]
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time_dt"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    df = df[
        [
            "timestamp", "open", "high", "low", "close", "volume",
            "quote_volume", "close_time_dt",
        ]
    ].copy()

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        raise ValueError("OHLCV فارغة بعد التنظيف")

    # Basic OHLCV sanity checks.
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices يجب أن تكون > 0")
    if (df["volume"] < 0).any():
        raise ValueError("volume لا يمكن أن يكون سالباً")
    if (df["high"] < df["low"]).any():
        raise ValueError("وجد high < low")
    if (df["high"] < df[["open", "close"]].max(axis=1)).any():
        raise ValueError("high أقل من open/close")
    if (df["low"] > df[["open", "close"]].min(axis=1)).any():
        raise ValueError("low أعلى من open/close")

    df = (
        df.sort_values("timestamp")
        .drop_duplicates(subset="timestamp", keep="last")
        .reset_index(drop=True)
    )

    # Binance normally returns the current open candle as the last row.
    # Remove it only when it is actually still open.
    if drop_incomplete and len(df) >= 2:
        now = pd.Timestamp.now(tz="UTC")
        if df.iloc[-1]["close_time_dt"] > now:
            df = df.iloc[:-1].copy()

    df = df.drop(columns=["close_time_dt"])

    if len(df) < MIN_OBSERVATIONS:
        raise ValueError("عدد البيانات أصبح غير كافٍ بعد تنظيف الشموع")

    return df


def fetch_returns(
    symbol: str,
    interval: str = DEFAULT_INTERVAL,
    limit: int = DEFAULT_LIMIT,
) -> pd.Series:
    """Backward-compatible helper returning log returns."""
    df = fetch_ohlcv(symbol, interval=interval, limit=limit)
    returns = np.log(df["close"] / df["close"].shift(1))
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
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
    Build the HMM observation matrix.

    DataFrame input (preferred)
        Uses real OHLCV-derived range and volume information.

    Series input (legacy / current quant_alert path)
        There is no high/low/volume information. To keep the historical
        API working, the function creates stable fallback features:

        range    = abs(log return)     [proxy only]
        volume_z = 0                   [neutral because volume is unknown]

    This prevents the old KeyError while making the limitation explicit.
    """
    volatility_window = _validate_window(volatility_window, "volatility_window")
    momentum_window = _validate_window(momentum_window, "momentum_window")
    volume_window = _validate_window(volume_window, "volume_window")

    if isinstance(data, pd.Series):
        returns = _validate_series(data, "returns")
        features = pd.DataFrame(index=returns.index)

        features["return"] = returns
        features["realized_vol"] = returns.rolling(volatility_window).std()

        # Returns-only proxy. It is NOT equivalent to true OHLC range.
        features["range"] = returns.abs()

        # Volume is unavailable in a Series-only input.
        features["volume_z"] = 0.0

        features["momentum"] = returns.rolling(momentum_window).sum()

        previous_vol = (
            returns.rolling(volatility_window).std().shift(1)
        )
        features["return_z"] = returns / previous_vol.clip(lower=EPS)

        feature_source = "returns_only"

    elif isinstance(data, pd.DataFrame):
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"OHLCV columns ناقصة: {sorted(missing)}")

        df = data.copy()
        close = pd.to_numeric(df["close"], errors="coerce")
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        volume = pd.to_numeric(df["volume"], errors="coerce")

        returns = np.log(close / close.shift(1))
        features = pd.DataFrame(index=df.index)

        features["return"] = returns
        features["realized_vol"] = returns.rolling(volatility_window).std()
        features["range"] = np.log(high / low).clip(lower=0.0)

        volume_mean = volume.rolling(volume_window).mean()
        volume_std = volume.rolling(volume_window).std()
        # Cross-window z-score is more informative than a raw volume level.
        features["volume_z"] = (
            (volume - volume_mean) / volume_std.clip(lower=EPS)
        )

        features["momentum"] = np.log(
            close / close.shift(momentum_window)
        )

        previous_vol = (
            returns.rolling(volatility_window).std().shift(1)
        )
        features["return_z"] = returns / previous_vol.clip(lower=EPS)

        feature_source = "ohlcv"

    else:
        raise TypeError("data يجب أن يكون pandas Series أو DataFrame")

    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    if features.empty:
        raise ValueError("لا توجد features صالحة بعد الحساب")

    # Protect the Gaussian likelihood from isolated extreme observations.
    # The rows themselves remain in the sequence.
    for column in features.columns:
        lower = float(features[column].quantile(CLIP_LOW))
        upper = float(features[column].quantile(CLIP_HIGH))
        if np.isfinite(lower) and np.isfinite(upper) and upper >= lower:
            features[column] = features[column].clip(lower=lower, upper=upper)

    features.attrs["feature_source"] = feature_source
    features.attrs["range_is_proxy"] = feature_source == "returns_only"
    features.attrs["volume_available"] = feature_source == "ohlcv"
    return features


# =====================================================================
# STATE LABELING
# =====================================================================

def label_states(state_statistics: pd.DataFrame, n_states: int) -> Dict[int, str]:
    """Map arbitrary HMM state IDs to economically readable labels."""
    if state_statistics.empty:
        raise ValueError("state_statistics فارغ")

    ordered = list(state_statistics.sort_values("mean_return").index)
    labels: Dict[int, str] = {}

    if n_states == 2:
        labels[ordered[0]] = "BEAR"
        labels[ordered[-1]] = "BULL"
    elif n_states == 3:
        labels[ordered[0]] = "BEAR"
        labels[ordered[1]] = "NEUTRAL"
        labels[ordered[2]] = "BULL"
    else:
        labels[ordered[0]] = "BEAR"
        labels[ordered[-1]] = "BULL"
        for state in ordered[1:-1]:
            labels[state] = f"NEUTRAL_{state}"

    return labels


def _canonicalize_probability_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    if total <= EPS:
        return np.full(len(values), 1.0 / max(len(values), 1))
    return values / total


# =====================================================================
# HMM ENGINE
# =====================================================================

def _fit_best_model(
    X: np.ndarray,
    n_states: int,
    n_iter: int,
    attempts: int,
    random_state: int,
) -> tuple[GaussianHMM, float, bool, int]:
    """Fit several initializations and choose the strongest valid model."""
    best_model: Optional[GaussianHMM] = None
    best_score = -np.inf
    best_converged = False
    successful = 0

    # Prefer converged models. Only if none converge do we fall back to the
    # best finite-likelihood model, preserving availability in difficult data.
    best_converged_model: Optional[GaussianHMM] = None
    best_converged_score = -np.inf

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
            score = float(model.score(X))
            if not np.isfinite(score):
                raise ValueError("non-finite log-likelihood")

            converged = bool(getattr(model.monitor_, "converged", False))
            successful += 1

            log.info(
                "HMM attempt=%d/%d score=%.4f converged=%s",
                attempt + 1,
                attempts,
                score,
                converged,
            )

            if score > best_score:
                best_model = model
                best_score = score
                best_converged = converged

            if converged and score > best_converged_score:
                best_converged_model = model
                best_converged_score = score

        except Exception as exc:
            log.warning("HMM attempt %d failed: %s", attempt + 1, exc)

    if best_converged_model is not None:
        return best_converged_model, best_converged_score, True, successful

    if best_model is not None:
        return best_model, best_score, best_converged, successful

    raise RuntimeError("HMM فشل في جميع محاولات التدريب")


def detect_regimes(
    data: Union[pd.DataFrame, pd.Series],
    n_states: int = DEFAULT_N_STATES,
    n_iter: int = DEFAULT_N_ITER,
    attempts: int = DEFAULT_ATTEMPTS,
    random_state: int = 42,
    volatility_window: int = 24,
    momentum_window: int = 12,
    volume_window: int = 24,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    max_entropy: float = DEFAULT_MAX_ENTROPY,
) -> Dict[str, Any]:
    """
    Fit the Gaussian HMM and return a complete regime analysis object.

    The result preserves the keys expected by the existing MEGA modules,
    while adding explicit diagnostics such as feature source and trust state.
    """
    n_states = int(n_states)
    n_iter = int(n_iter)
    attempts = int(attempts)

    if n_states < 2:
        raise ValueError("n_states يجب أن يكون >= 2")
    if n_states > 5:
        raise ValueError("n_states > 5 غير مستحسن لهذا النظام")
    if n_iter < 100:
        raise ValueError("n_iter يجب أن يكون >= 100")
    if attempts < 1:
        raise ValueError("attempts يجب أن يكون >= 1")

    min_confidence = float(np.clip(min_confidence, 0.0, 1.0))
    max_entropy = float(np.clip(max_entropy, 0.0, 1.0))

    features = build_features(
        data=data,
        volatility_window=volatility_window,
        momentum_window=momentum_window,
        volume_window=volume_window,
    )

    # HMM fitting requires enough effective observations for both transition
    # estimation and the rolling features.
    if len(features) < MIN_OBSERVATIONS:
        raise ValueError(
            f"عدد observations غير كافٍ: {len(features)} < {MIN_OBSERVATIONS}"
        )

    feature_names = list(features.columns)
    raw_X = features.to_numpy(dtype=float)
    if not np.isfinite(raw_X).all():
        raise ValueError("X يحتوي NaN أو Inf")

    scaler = StandardScaler()
    X = scaler.fit_transform(raw_X)
    if not np.isfinite(X).all():
        raise ValueError("فشل StandardScaler: X غير صالح")

    model, log_likelihood, converged, successful_models = _fit_best_model(
        X=X,
        n_states=n_states,
        n_iter=n_iter,
        attempts=attempts,
        random_state=random_state,
    )

    hidden_states = model.predict(X)
    posterior = np.asarray(model.predict_proba(X), dtype=float)
    if posterior.ndim != 2 or posterior.shape[1] != n_states:
        raise RuntimeError("predict_proba أعاد shape غير متوقع")

    posterior = np.vstack(
        [_canonicalize_probability_vector(row) for row in posterior]
    )

    # --------------------------------------------------------------
    # State statistics in original feature units
    # --------------------------------------------------------------
    rows = []
    for state in range(n_states):
        mask = hidden_states == state
        count = int(mask.sum())

        if count:
            state_features = features.iloc[np.flatnonzero(mask)]
            row = {
                "state": state,
                "count": count,
                "pct": count / len(hidden_states) * 100.0,
                "mean_return": float(state_features["return"].mean()),
                "mean_volatility": float(state_features["realized_vol"].mean()),
                "mean_range": float(state_features["range"].mean()),
                "mean_volume_z": float(state_features["volume_z"].mean()),
                "mean_momentum": float(state_features["momentum"].mean()),
            }
        else:
            row = {
                "state": state,
                "count": 0,
                "pct": 0.0,
                "mean_return": 0.0,
                "mean_volatility": 0.0,
                "mean_range": 0.0,
                "mean_volume_z": 0.0,
                "mean_momentum": 0.0,
            }
        rows.append(row)

    state_statistics = pd.DataFrame(rows).set_index("state")
    labels = label_states(state_statistics, n_states=n_states)
    state_statistics["label"] = [labels[int(i)] for i in state_statistics.index]

    # --------------------------------------------------------------
    # Current state / posterior
    # --------------------------------------------------------------
    current_state = int(hidden_states[-1])
    current_probabilities = _canonicalize_probability_vector(posterior[-1])
    current_probability = float(current_probabilities[current_state])
    max_probability = float(np.max(current_probabilities))
    entropy = _entropy(current_probabilities)

    raw_regime = labels[current_state]
    trusted = bool(
        converged
        and max_probability >= min_confidence
        and entropy <= max_entropy
    )
    reported_regime = raw_regime if trusted else "UNKNOWN"

    regime_probabilities = {
        labels[i]: float(current_probabilities[i])
        for i in range(n_states)
    }

    # --------------------------------------------------------------
    # State persistence / transitions
    # --------------------------------------------------------------
    persistence = _consecutive_count(hidden_states, current_state)

    transition_matrix = np.asarray(model.transmat_, dtype=float)
    if transition_matrix.shape != (n_states, n_states):
        raise RuntimeError("transition matrix shape غير صحيح")
    transition_matrix = np.clip(transition_matrix, 0.0, None)
    transition_matrix = transition_matrix / transition_matrix.sum(
        axis=1, keepdims=True
    )

    stay_probability = float(transition_matrix[current_state, current_state])
    expected_duration = _expected_duration(stay_probability)

    next_state_probability = _canonicalize_probability_vector(
        current_probabilities @ transition_matrix
    )
    next_regime_probabilities = {
        labels[i]: float(next_state_probability[i])
        for i in range(n_states)
    }

    regime_changed = bool(
        len(hidden_states) >= 2
        and int(hidden_states[-1]) != int(hidden_states[-2])
    )

    # --------------------------------------------------------------
    # Model information criteria
    # --------------------------------------------------------------
    n_samples, n_features = X.shape
    n_parameters = (
        n_states * (n_states - 1)
        + (n_states - 1)
        + 2 * n_states * n_features
    )

    aic = float(2.0 * n_parameters - 2.0 * log_likelihood)
    bic = float(n_parameters * np.log(n_samples) - 2.0 * log_likelihood)

    means_scaled = np.asarray(model.means_, dtype=float)
    variances_scaled = np.asarray(model.covars_, dtype=float)

    log.info(
        "HMM completed | source=%s | states=%d | score=%.4f | converged=%s | "
        "regime=%s | probability=%.2f%% | confidence=%.2f%%",
        features.attrs.get("feature_source", "unknown"),
        n_states,
        log_likelihood,
        converged,
        reported_regime,
        current_probability * 100.0,
        max_probability * 100.0,
    )

    return {
        # Core model
        "model": model,
        "scaler": scaler,
        "features": features,
        "feature_names": feature_names,
        "feature_source": features.attrs.get("feature_source", "unknown"),
        "range_is_proxy": bool(features.attrs.get("range_is_proxy", False)),
        "volume_available": bool(features.attrs.get("volume_available", False)),

        # States
        "hidden_states": hidden_states,
        "current_state": current_state,
        "current_regime": raw_regime,
        "raw_regime": raw_regime,
        "reported_regime": reported_regime,

        # Probabilities
        "posterior_probabilities": posterior,
        "current_probabilities": current_probabilities,
        "regime_probabilities": regime_probabilities,
        "current_probability": current_probability,
        "max_probability": max_probability,

        # Trust / uncertainty
        "confidence": max_probability,
        "entropy": entropy,
        "trusted": trusted,
        "min_confidence": min_confidence,
        "max_entropy": max_entropy,

        # Dynamics
        "transition_matrix": transition_matrix,
        "next_state_probability": next_state_probability,
        "next_regime_probabilities": next_regime_probabilities,
        "stay_probability": stay_probability,
        "expected_duration": expected_duration,
        "state_persistence": persistence,
        "regime_changed": regime_changed,

        # Statistics
        "labels": labels,
        "state_statistics": state_statistics,

        # HMM parameters
        "means_scaled": means_scaled,
        "variances_scaled": variances_scaled,

        # Model quality
        "log_likelihood": float(log_likelihood),
        "score_per_observation": float(log_likelihood / max(n_samples, 1)),
        "aic": aic,
        "bic": bic,
        "converged": converged,
        "successful_models": successful_models,
        "attempts": attempts,
        "n_states": n_states,
        "n_features": n_features,
        "n_observations": n_samples,
        "n_parameters": n_parameters,
        "n_iter": n_iter,
    }


# =====================================================================
# CURRENT REGIME API
# =====================================================================

def get_current_regime(
    data: Union[pd.DataFrame, pd.Series],
    n_states: int = DEFAULT_N_STATES,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Return the current regime in the compact shape expected by MEGA.

    Safety improvement:
    ``current_regime`` is the *reported* regime, so low-confidence or
    non-converged models become ``UNKNOWN`` and are not silently treated as
    confirmed market states by downstream filters.
    """
    result = detect_regimes(data=data, n_states=n_states, **kwargs)

    return {
        "current_state": result["current_state"],
        "current_regime": result["reported_regime"],
        "raw_regime": result["raw_regime"],
        "reported_regime": result["reported_regime"],
        "regime_prob": result["current_probability"],
        "confidence": result["confidence"],
        "entropy": result["entropy"],
        "trusted": result["trusted"],
        "regime_probabilities": result["regime_probabilities"],
        "next_regime_probabilities": result["next_regime_probabilities"],
        "stay_probability": result["stay_probability"],
        "expected_duration": result["expected_duration"],
        "state_persistence": result["state_persistence"],
        "regime_changed": result["regime_changed"],
        "means": result["state_statistics"]["mean_return"].to_numpy(),
        "variances": result["variances_scaled"],
        "transition_matrix": result["transition_matrix"],
        "n_states": result["n_states"],
        "aic": result["aic"],
        "bic": result["bic"],
        "converged": result["converged"],
        "feature_source": result["feature_source"],
        "range_is_proxy": result["range_is_proxy"],
        "volume_available": result["volume_available"],
        "score_per_observation": result["score_per_observation"],
    }


# =====================================================================
# SIGNAL FILTER
# =====================================================================

def filter_signals_by_regime(
    signal: int,
    current_regime: str,
    confidence: float = 1.0,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    trusted: Optional[bool] = None,
) -> int:
    """
    Filter a Spot signal by HMM regime.

    +1 = BUY / increase Spot exposure
     0 = HOLD
    -1 = SELL / reduce Spot exposure

    SELL is never interpreted as opening a short position.
    """
    signal = int(np.sign(signal))
    current_regime = str(current_regime).strip().upper()
    confidence = float(np.clip(confidence, 0.0, 1.0))
    min_confidence = float(np.clip(min_confidence, 0.0, 1.0))

    if trusted is None:
        trusted = confidence >= min_confidence

    if not trusted or confidence < min_confidence:
        log.info(
            "[HMM FILTER] blocked: trusted=%s confidence=%.2f%%",
            trusted,
            confidence * 100.0,
        )
        return 0

    if current_regime in {"UNKNOWN", "UNDEFINED", "NONE", ""}:
        return 0

    if current_regime == "BEAR" and signal == 1:
        log.info("[HMM FILTER] BUY blocked (BEAR regime)")
        return 0

    if current_regime == "BULL":
        return signal

    if current_regime.startswith("NEUTRAL"):
        return signal

    if current_regime == "BEAR":
        return signal

    log.warning("[HMM FILTER] Unknown regime label: %s", current_regime)
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
    """Return an exposure multiplier; this is not a complete position size."""
    current_regime = str(current_regime).strip().upper()
    confidence = float(np.clip(confidence, 0.0, 1.0))

    base = 0.0
    if current_regime == "BULL":
        base = 1.00
    elif current_regime.startswith("NEUTRAL"):
        base = 0.60
    elif current_regime == "BEAR":
        base = 0.25
    elif current_regime == "UNKNOWN":
        base = 0.00

    return float(np.clip(base * confidence, min_multiplier, max_multiplier))


# =====================================================================
# REGIME STATISTICS / TELEGRAM SUMMARY
# =====================================================================

def get_regime_stats(
    data: Union[pd.DataFrame, pd.Series],
    n_states: int = DEFAULT_N_STATES,
    **kwargs: Any,
) -> list:
    """Return compact per-regime statistics."""
    result = detect_regimes(data=data, n_states=n_states, **kwargs)
    matrix = result["transition_matrix"]
    hidden_states = result["hidden_states"]
    stats = result["state_statistics"]

    output = []
    for state in stats.index:
        state_int = int(state)
        stay = float(matrix[state_int, state_int])
        duration = _expected_duration(stay)
        output.append(
            {
                "state": state_int,
                "label": str(stats.loc[state, "label"]),
                "count": int(stats.loc[state, "count"]),
                "pct": round(float(stats.loc[state, "pct"]), 2),
                "mean_return": float(stats.loc[state, "mean_return"]),
                "mean_volatility": float(stats.loc[state, "mean_volatility"]),
                "mean_range": float(stats.loc[state, "mean_range"]),
                "mean_volume_z": float(stats.loc[state, "mean_volume_z"]),
                "mean_momentum": float(stats.loc[state, "mean_momentum"]),
                "stay_probability": round(stay, 6),
                "expected_duration": (
                    float(duration) if np.isfinite(duration) else float("inf")
                ),
                "current_persistence": _consecutive_count(hidden_states, state_int),
            }
        )
    return output


def summarize_regime(result: Dict[str, Any]) -> Dict[str, Any]:
    """Create a small Telegram-friendly summary from detect_regimes()."""
    next_map = result["next_regime_probabilities"]
    next_regime = max(next_map, key=next_map.get) if next_map else "UNKNOWN"
    duration = result["expected_duration"]

    return {
        "regime": result["reported_regime"],
        "raw_regime": result["raw_regime"],
        "confidence": round(float(result["confidence"]), 4),
        "uncertainty": round(float(result["entropy"]), 4),
        "trusted": bool(result["trusted"]),
        "regime_probability": round(float(result["current_probability"]), 4),
        "persistence": int(result["state_persistence"]),
        "stay_probability": round(float(result["stay_probability"]), 4),
        "expected_duration": round(float(duration), 2) if np.isfinite(duration) else None,
        "regime_changed": bool(result["regime_changed"]),
        "next_regime": next_regime,
        "aic": round(float(result["aic"]), 2),
        "bic": round(float(result["bic"]), 2),
        "converged": bool(result["converged"]),
        "feature_source": result["feature_source"],
        "range_is_proxy": bool(result["range_is_proxy"]),
        "volume_available": bool(result["volume_available"]),
    }


# =====================================================================
# SELF TEST
# =====================================================================

def _synthetic_ohlcv(rows: int = 260, seed: int = 7) -> pd.DataFrame:
    """Deterministic synthetic OHLCV for local smoke testing."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0002, 0.012, rows)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = np.r_[100.0, close[:-1]]
    spread = np.abs(rng.normal(0.0, 0.004, rows)) + 1e-4
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = np.exp(rng.normal(10.0, 0.4, rows))

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print("\n=== MEGA HMM SELF TEST ===")

    try:
        # 1) Preferred OHLCV path.
        df = _synthetic_ohlcv()
        result = detect_regimes(
            df,
            n_states=3,
            n_iter=300,
            attempts=3,
            random_state=42,
        )
        print("OHLCV result:", summarize_regime(result))

        # 2) Legacy Series path: this is the path that previously raised
        # KeyError('range'). It must now complete successfully.
        returns = np.log(df["close"] / df["close"].shift(1)).dropna()
        legacy = detect_regimes(
            returns,
            n_states=2,
            n_iter=300,
            attempts=3,
            random_state=42,
        )
        print("Series result:", summarize_regime(legacy))

        compact = get_current_regime(returns, n_states=2)
        print("Current regime:", compact)

        print("HMM SELF TEST PASSED")

    except Exception as exc:
        log.exception("HMM self test failed: %s", exc)
        raise
