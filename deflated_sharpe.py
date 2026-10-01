"""
deflated_sharpe.py
==================

Research-grade implementation of the Deflated Sharpe Ratio (DSR).

Primary references
------------------
1) Bailey, D. H. & Lopez de Prado, M. (2014),
   "The Deflated Sharpe Ratio: Correcting for Selection Bias,
   Backtest Overfitting, and Non-Normality",
   Journal of Portfolio Management, 40(5), 94-107.
   DOI: 10.3905/jpm.2014.40.5.094

2) Lopez de Prado's original reference implementation:
   DSR.py (2014), available from quantresearch.org.

3) Bailey, Borwein, Lopez de Prado & Zhu (2016/2017),
   "The Probability of Backtest Overfitting" (CSCV/PBO).

Purpose
-------
The DSR is a statistical diagnostic intended to answer:

    "After accounting for non-normal returns and selection among
     multiple trials, how much evidence is there that the observed
     Sharpe exceeds the Sharpe that could plausibly arise from luck?"

Important
---------
DSR is NOT a guarantee of future profitability.
It is also not a replacement for:
    - purged / embargoed cross-validation,
    - walk-forward testing,
    - transaction costs and slippage,
    - realistic execution assumptions,
    - multiple-testing accounting,
    - or out-of-sample evaluation.

The implementation keeps the canonical DSR mathematics separate from
optional engineering diagnostics so it can be composed with the later
backtest / CPCV modules in MEGA-eva-AI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
from scipy import stats


log = logging.getLogger("deflated_sharpe")

_EULER_GAMMA = 0.5772156649015328606
_EPS = 1e-12


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass(frozen=True)
class SharpeDistribution:
    """Estimated cross-trial distribution of Sharpe ratios."""

    mean: float
    std: float
    n: int


# ============================================================
# INPUT / NUMERICAL UTILITIES
# ============================================================

def _clean_returns(returns: Iterable[float]) -> np.ndarray:
    """Convert returns to a finite 1-D float array."""
    x = np.asarray(list(returns), dtype=float).reshape(-1)
    x = x[np.isfinite(x)]

    if x.size == 0:
        raise ValueError("returns contains no finite observations.")

    return x


def _clean_trial_sharpes(sharpes: Iterable[float]) -> np.ndarray:
    """Convert trial Sharpe ratios to a finite 1-D float array."""
    x = np.asarray(list(sharpes), dtype=float).reshape(-1)
    x = x[np.isfinite(x)]

    if x.size == 0:
        raise ValueError("trial_sharpes contains no finite values.")

    return x


def _validate_periods_per_year(periods_per_year: float) -> float:
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be > 0.")
    return float(periods_per_year)


# ============================================================
# SHARPE RATIO
# ============================================================

def excess_returns(
    returns: Iterable[float],
    periods_per_year: float = 252.0,
    rf: float = 0.0,
) -> np.ndarray:
    """
    Convert periodic returns into excess periodic returns.

    Parameters
    ----------
    returns:
        Periodic arithmetic returns, e.g. 0.001 = +0.1%.
    periods_per_year:
        Number of return observations per year.
    rf:
        Annualized risk-free rate expressed as a decimal.

    Notes
    -----
    For a small periodic risk-free rate, rf / periods_per_year is a
    conventional simple-period approximation. The DSR itself is then
    calculated from the unannualized periodic Sharpe ratio.
    """
    ppy = _validate_periods_per_year(periods_per_year)
    x = _clean_returns(returns)
    rf_period = float(rf) / ppy
    return x - rf_period


def sharpe_ratio(
    returns: Iterable[float],
    periods_per_year: float = 252.0,
    rf: float = 0.0,
    annualize: bool = True,
) -> float:
    """
    Sample Sharpe ratio.

    Returns the annualized Sharpe when annualize=True; otherwise the
    periodic Sharpe is returned. The non-annualized value is the quantity
    used internally by the canonical DSR formula.
    """
    x = excess_returns(
        returns,
        periods_per_year=periods_per_year,
        rf=rf,
    )

    if x.size < 2:
        return np.nan

    sd = float(np.std(x, ddof=1))
    if not np.isfinite(sd) or sd <= _EPS:
        return np.nan

    sr_periodic = float(np.mean(x) / sd)

    if annualize:
        return float(sr_periodic * np.sqrt(periods_per_year))

    return sr_periodic


def sharpe_standard_error(
    periodic_sharpe: float,
    sample_size: int,
    skew: float,
    kurtosis: float,
) -> float:
    """
    Standard error of the sample Sharpe under the Bailey/López-de-Prado
    non-normality correction.

    Formula:
        sqrt([1 - gamma3*SR + ((gamma4-1)/4)*SR^2] / (T-1))

    where gamma3 is skewness and gamma4 is Pearson kurtosis.
    """
    if sample_size <= 1:
        raise ValueError("sample_size must be > 1.")

    sr = float(periodic_sharpe)
    sk = float(skew)
    ku = float(kurtosis)

    inner = (
        1.0
        - sk * sr
        + ((ku - 1.0) / 4.0) * (sr ** 2)
    )

    # Numerical protection. A negative variance estimate is not meaningful.
    if inner < 0:
        log.warning(
            "Sharpe variance term became negative (%.6g); "
            "clamping to zero.",
            inner,
        )
        inner = 0.0

    return float(np.sqrt(inner / (sample_size - 1)))


# ============================================================
# CROSS-TRIAL SHARPE DISTRIBUTION
# ============================================================

def estimate_trial_sharpe_distribution(
    trial_sharpes: Iterable[float],
    ddof: int = 1,
) -> SharpeDistribution:
    """
    Estimate the cross-sectional mean and standard deviation of Sharpe
    ratios across the strategies / parameter trials actually tried.

    This is preferable to inventing an arbitrary variance.
    """
    x = _clean_trial_sharpes(trial_sharpes)

    if x.size < 2:
        raise ValueError(
            "At least two trial Sharpe ratios are required to estimate "
            "cross-trial standard deviation."
        )

    std = float(np.std(x, ddof=ddof))

    return SharpeDistribution(
        mean=float(np.mean(x)),
        std=std,
        n=int(x.size),
    )


# ============================================================
# EXPECTED MAXIMUM SHARPE
# ============================================================

def expected_max_sharpe(
    n_trials: int,
    mean_sharpe: float = 0.0,
    std_sharpe: float = 1.0,
) -> float:
    """
    Expected maximum Sharpe ratio among n_trials Gaussian trials.

    Bailey & López de Prado use the closed-form approximation:

        E[max(SR)] = mu + sigma * E[max(Z)]

    with

        E[max(Z)] ≈ (1-gamma) Phi^-1(1 - 1/N)
                       + gamma Phi^-1(1 - 1/(N*e))

    Parameters
    ----------
    n_trials:
        Number of independent trials represented by N.
    mean_sharpe:
        Mean of the cross-trial Sharpe distribution.
    std_sharpe:
        Standard deviation of the cross-trial Sharpe distribution.

    Notes
    -----
    The trial count should represent the number of effectively independent
    trials, not blindly every hyperparameter combination. If many trials
    are highly correlated, the raw count can overstate the effective search.
    """
    if int(n_trials) != n_trials or n_trials < 1:
        raise ValueError("n_trials must be a positive integer.")

    mu = float(mean_sharpe)
    sigma = float(std_sharpe)

    if not np.isfinite(mu) or not np.isfinite(sigma):
        raise ValueError("mean_sharpe and std_sharpe must be finite.")

    if sigma < 0:
        raise ValueError("std_sharpe cannot be negative.")

    if n_trials == 1:
        # For one trial, the expected maximum is simply its distribution mean.
        return mu

    n = float(n_trials)

    # Use survival-function probabilities away from exactly 1.0 to reduce
    # numerical problems in the inverse normal CDF for very large N.
    p1 = 1.0 - 1.0 / n
    p2 = 1.0 - 1.0 / (n * np.e)

    z1 = float(stats.norm.ppf(p1))
    z2 = float(stats.norm.ppf(p2))

    max_z = (
        (1.0 - _EULER_GAMMA) * z1
        + _EULER_GAMMA * z2
    )

    return float(mu + sigma * max_z)


# ============================================================
# PSR / DSR CORE
# ============================================================

def _moments(returns: np.ndarray) -> tuple[float, float]:
    """Pearson skewness and kurtosis of finite returns."""
    if returns.size < 3:
        raise ValueError("At least 3 observations are required for skewness.")

    skew = float(stats.skew(returns, bias=False))
    kurt = float(stats.kurtosis(returns, fisher=False, bias=False))

    # Constant or near-constant data can produce NaN moments.
    if not np.isfinite(skew):
        skew = 0.0
    if not np.isfinite(kurt):
        kurt = 3.0

    return skew, kurt


def probabilistic_sharpe_ratio(
    returns: Iterable[float],
    benchmark_sharpe: float = 0.0,
    rf: float = 0.0,
    periods_per_year: float = 252.0,
) -> float:
    """
    Probabilistic Sharpe Ratio (PSR).

    This is the DSR core with a user-specified benchmark Sharpe and without
    the multiple-testing benchmark estimation step.

    Returns the probability in [0, 1] that the strategy's true Sharpe
    exceeds benchmark_sharpe under the model assumptions.
    """
    x = excess_returns(
        returns,
        periods_per_year=periods_per_year,
        rf=rf,
    )

    t = x.size
    if t < 4:
        return np.nan

    sd = float(np.std(x, ddof=1))
    if not np.isfinite(sd) or sd <= _EPS:
        return np.nan

    sr = float(np.mean(x) / sd)
    skew, kurt = _moments(x)
    sigma_sr = sharpe_standard_error(
        sr,
        sample_size=t,
        skew=skew,
        kurtosis=kurt,
    )

    if sigma_sr <= _EPS:
        return float(1.0 if sr > benchmark_sharpe else 0.0)

    z = (sr - float(benchmark_sharpe)) / sigma_sr
    probability = float(stats.norm.cdf(z))

    return float(np.clip(probability, 0.0, 1.0))


def deflated_sharpe_ratio(
    returns: Iterable[float],
    n_trials: int = 1,
    trial_sharpes: Optional[Iterable[float]] = None,
    mean_sharpe: Optional[float] = None,
    std_sharpe: Optional[float] = None,
    periods_per_year: float = 252.0,
    rf: float = 0.0,
    min_observations: int = 20,
) -> dict:
    """
    Compute a research-grade Deflated Sharpe Ratio report.

    Trial-distribution inputs
    -------------------------
    Recommended:
        trial_sharpes=[SR(strategy_1), ..., SR(strategy_K)]

    Alternatively:
        mean_sharpe=<cross-trial mean>
        std_sharpe=<cross-trial std>

    If neither is supplied, the implementation falls back to a zero-
    dispersion benchmark. In that case the result is effectively a PSR-like
    test against zero rather than a fully multiple-testing-adjusted DSR.

    Important unit rule
    -------------------
    ``trial_sharpes``, ``mean_sharpe`` and ``std_sharpe`` must be expressed
    in the SAME Sharpe units as the periodic (non-annualized) Sharpe used
    internally by the DSR formula.
    """
    ppy = _validate_periods_per_year(periods_per_year)
    raw_returns = _clean_returns(returns)

    t = int(raw_returns.size)

    if t < int(min_observations):
        return {
            "status": "insufficient_data",
            "sharpe_periodic": np.nan,
            "sharpe_annualized": np.nan,
            "benchmark_sharpe": np.nan,
            "standard_error_sharpe": np.nan,
            "z_score": np.nan,
            "deflated_sharpe": np.nan,
            "probability": np.nan,
            "skew": np.nan,
            "kurtosis": np.nan,
            "T": t,
            "n_trials": int(n_trials),
            "trial_mean_sharpe": np.nan,
            "trial_std_sharpe": np.nan,
            "trial_count_observed": 0,
            "warning": f"Need at least {min_observations} finite return observations.",
        }

    x = excess_returns(
        raw_returns,
        periods_per_year=ppy,
        rf=rf,
    )

    sample_std = float(np.std(x, ddof=1))
    if not np.isfinite(sample_std) or sample_std <= _EPS:
        return {
            "status": "degenerate_returns",
            "sharpe_periodic": np.nan,
            "sharpe_annualized": np.nan,
            "benchmark_sharpe": np.nan,
            "standard_error_sharpe": np.nan,
            "z_score": np.nan,
            "deflated_sharpe": np.nan,
            "probability": np.nan,
            "skew": 0.0,
            "kurtosis": 3.0,
            "T": t,
            "n_trials": int(n_trials),
            "trial_mean_sharpe": np.nan,
            "trial_std_sharpe": np.nan,
            "trial_count_observed": 0,
            "warning": "Return volatility is zero or numerically degenerate.",
        }

    # Canonical DSR uses the NON-annualized sample Sharpe.
    sr_periodic = float(np.mean(x) / sample_std)
    sr_annualized = float(sr_periodic * np.sqrt(ppy))

    skew, kurt = _moments(x)

    sigma_sr = sharpe_standard_error(
        sr_periodic,
        sample_size=t,
        skew=skew,
        kurtosis=kurt,
    )

    trial_mean = None
    trial_std = None
    trial_count_observed = 0
    benchmark_source = "zero_sharpe_fallback"

    if trial_sharpes is not None:
        trial_dist = estimate_trial_sharpe_distribution(trial_sharpes)
        trial_mean = trial_dist.mean
        trial_std = trial_dist.std
        trial_count_observed = trial_dist.n
        benchmark_source = "trial_sharpes"

    if mean_sharpe is not None:
        trial_mean = float(mean_sharpe)
        benchmark_source = "explicit_mean"

    if std_sharpe is not None:
        trial_std = float(std_sharpe)
        benchmark_source = (
            "explicit_mean_and_std"
            if mean_sharpe is not None
            else "explicit_std"
        )

    if trial_mean is None:
        trial_mean = 0.0

    if trial_std is None:
        trial_std = 0.0

    if trial_std < 0 or not np.isfinite(trial_std):
        raise ValueError("std_sharpe must be finite and >= 0.")

    # A trial count larger than the number of supplied trial Sharpes is
    # permitted only when the caller deliberately supplies an effective
    # number of trials through n_trials. That is useful for large searches
    # whose full trial Sharpe vector is not retained.
    benchmark_sr = expected_max_sharpe(
        n_trials=int(n_trials),
        mean_sharpe=trial_mean,
        std_sharpe=trial_std,
    )

    if sigma_sr <= _EPS:
        z_score = np.inf if sr_periodic > benchmark_sr else -np.inf
        dsr = 1.0 if sr_periodic > benchmark_sr else 0.0
    else:
        z_score = (
            (sr_periodic - benchmark_sr)
            / sigma_sr
        )
        dsr = float(stats.norm.cdf(z_score))

    dsr = float(np.clip(dsr, 0.0, 1.0))

    warnings = []

    if benchmark_source == "zero_sharpe_fallback" and int(n_trials) > 1:
        warnings.append(
            "Multiple trials were specified, but no cross-trial Sharpe dispersion "
            "was supplied; the multiple-testing benchmark could not be estimated "
            "from observed trial statistics."
        )

    if int(n_trials) > 1 and trial_std == 0:
        warnings.append(
            "Cross-trial Sharpe std is zero. This is not a useful empirical estimate "
            "for a multi-trial search."
        )

    result = {
        "status": "ok",
        "sharpe_periodic": round(sr_periodic, 8),
        "sharpe_annualized": round(sr_annualized, 8),
        "benchmark_sharpe": round(benchmark_sr, 8),
        "standard_error_sharpe": round(sigma_sr, 8),
        "z_score": round(float(z_score), 8),
        "deflated_sharpe": round(dsr, 8),
        "probability": round(dsr, 8),
        "skew": round(skew, 8),
        "kurtosis": round(kurt, 8),
        "T": t,
        "n_trials": int(n_trials),
        "trial_mean_sharpe": round(float(trial_mean), 8),
        "trial_std_sharpe": round(float(trial_std), 8),
        "trial_count_observed": int(trial_count_observed),
        "benchmark_source": benchmark_source,
        "warnings": warnings,
    }

    log.info(
        "[DSR] SR(periodic)=%.4f | SR(annualized)=%.4f | "
        "benchmark=%.4f | SE=%.4f | DSR=%.4f | T=%d | K=%d",
        sr_periodic,
        sr_annualized,
        benchmark_sr,
        sigma_sr,
        dsr,
        t,
        int(n_trials),
    )

    return result


# ============================================================
# BACKTEST METRICS
# ============================================================

def equity_curve_from_returns(returns: Iterable[float]) -> np.ndarray:
    """Build a compounded equity curve starting from 1.0."""
    x = _clean_returns(returns)

    if np.any(x <= -1.0):
        raise ValueError("Returns <= -100% are not valid for compounding.")

    return np.cumprod(1.0 + x)


def max_drawdown(returns: Iterable[float]) -> float:
    """Return maximum drawdown as a negative decimal."""
    equity = equity_curve_from_returns(returns)
    peaks = np.maximum.accumulate(equity)
    dd = equity / np.maximum(peaks, _EPS) - 1.0
    return float(np.min(dd))


def profit_factor(returns: Iterable[float]) -> float:
    """Gross profits divided by absolute gross losses."""
    x = _clean_returns(returns)
    gains = float(np.sum(x[x > 0]))
    losses = float(-np.sum(x[x < 0]))

    if losses <= _EPS:
        return float("inf") if gains > 0 else np.nan

    return float(gains / losses)


def full_backtest_report(
    trades_df,
    n_trials: int = 1,
    trial_sharpes: Optional[Iterable[float]] = None,
    mean_sharpe: Optional[float] = None,
    std_sharpe: Optional[float] = None,
    periods_per_year: float = 252.0,
    rf: float = 0.0,
    return_column: str = "pnl_pct",
    min_observations: int = 20,
) -> dict:
    """
    Produce a backtest report around the DSR calculation.

    ``return_column`` is expected to contain per-observation percentage
    returns, e.g. +0.50 means +0.50% and is converted to +0.005.

    Warning:
        If these observations are individual trades rather than evenly spaced
        period returns, annualized Sharpe is only meaningful when
        ``periods_per_year`` is chosen to represent the actual sampling rate.
    """
    if trades_df is None or len(trades_df) == 0:
        return {"status": "no_data", "error": "No backtest observations."}

    if return_column not in trades_df.columns:
        return {
            "status": "missing_column",
            "error": f"Column '{return_column}' not found.",
        }

    raw_pct = np.asarray(trades_df[return_column], dtype=float)
    finite_mask = np.isfinite(raw_pct)
    raw_pct = raw_pct[finite_mask]

    if raw_pct.size == 0:
        return {
            "status": "no_finite_returns",
            "error": f"Column '{return_column}' contains no finite values.",
        }

    returns = raw_pct / 100.0

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    total_return = float(np.prod(1.0 + returns) - 1.0)
    win_rate = float(np.mean(returns > 0)) if returns.size else np.nan

    avg_win = float(np.mean(wins)) if wins.size else 0.0
    avg_loss = float(np.mean(losses)) if losses.size else 0.0

    dsr = deflated_sharpe_ratio(
        returns=returns,
        n_trials=n_trials,
        trial_sharpes=trial_sharpes,
        mean_sharpe=mean_sharpe,
        std_sharpe=std_sharpe,
        periods_per_year=periods_per_year,
        rf=rf,
        min_observations=min_observations,
    )

    report = {
        "status": "ok",
        "observations": int(returns.size),
        "total_compounded_return_pct": round(total_return * 100.0, 4),
        "win_rate_pct": round(win_rate * 100.0, 4),
        "avg_win_pct": round(avg_win * 100.0, 4),
        "avg_loss_pct": round(avg_loss * 100.0, 4),
        "profit_factor": round(profit_factor(returns), 6),
        "max_drawdown_pct": round(max_drawdown(returns) * 100.0, 4),
        **dsr,
    }

    return report


# ============================================================
# INTERPRETATION / REPORTING
# ============================================================

def interpret_dsr(
    dsr_probability: float,
) -> str:
    """
    Neutral statistical interpretation of the probability value.

    These are not trading recommendations or universal acceptance cutoffs.
    """
    p = float(dsr_probability)

    if not np.isfinite(p):
        return "غير صالح إحصائياً بسبب نقص/انهيار البيانات."

    if p >= 0.99:
        return "دليل إحصائي قوي جدًا ضمن افتراضات الاختبار."
    if p >= 0.95:
        return "دليل إحصائي قوي ضمن افتراضات الاختبار."
    if p >= 0.90:
        return "دليل إحصائي متوسط إلى قوي؛ يحتاج تحققًا خارج العينة."
    if p >= 0.75:
        return "دليل محدود؛ لا يكفي وحده لإثبات وجود مهارة."

    return "دليل ضعيف على تجاوز معيار الـSharpe المحدد."


# ============================================================
# SANITY TESTS
# ============================================================

def _run_self_tests() -> None:
    """Internal tests for mathematical and numerical invariants."""
    rng = np.random.default_rng(42)

    # 1) Constant positive returns must not yield an infinite Sharpe.
    constant = np.full(100, 0.001)
    sr = sharpe_ratio(constant, annualize=False)
    assert np.isnan(sr), "Constant returns should produce undefined Sharpe."

    # 2) Expected maximum should equal distribution mean for K=1.
    assert np.isclose(
        expected_max_sharpe(1, mean_sharpe=0.2, std_sharpe=0.5),
        0.2,
    )

    # 3) With larger dispersion and more trials, E[max] should not decrease.
    a = expected_max_sharpe(10, 0.0, 1.0)
    b = expected_max_sharpe(100, 0.0, 1.0)
    assert b > a

    # 4) Positive mean returns with sufficient observations produce finite DSR.
    good = rng.normal(0.001, 0.01, 1000)
    result = deflated_sharpe_ratio(
        good,
        n_trials=100,
        mean_sharpe=0.0,
        std_sharpe=0.5,
    )
    assert result["status"] == "ok"
    assert 0.0 <= result["deflated_sharpe"] <= 1.0

    # 5) Equity curve must compound, not sum.
    r = np.array([0.10, -0.10])
    eq = equity_curve_from_returns(r)
    assert np.isclose(eq[-1], 0.99)

    print("All self-tests passed.")


# ============================================================
# CLI TEST
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    _run_self_tests()

    rng = np.random.default_rng(123)

    print("\n" + "=" * 78)
    print("Deflated Sharpe Ratio — research test")
    print("=" * 78)

    # Example strategy returns: 2 years of daily observations.
    returns = rng.normal(
        loc=0.0008,
        scale=0.01,
        size=504,
    )

    # Cross-trial Sharpe sample. In the real pipeline this should come from
    # the actual strategy / hyperparameter search, ideally from the same
    # selection process being audited.
    trial_sharpes = rng.normal(
        loc=0.0,
        scale=0.50,
        size=100,
    )

    result = deflated_sharpe_ratio(
        returns=returns,
        n_trials=len(trial_sharpes),
        trial_sharpes=trial_sharpes,
        periods_per_year=365,  # example for daily crypto observations
    )

    print("\nDSR report:")
    for key, value in result.items():
        print(f"  {key}: {value}")

    if result.get("status") == "ok":
        print(
            "\nInterpretation:",
            interpret_dsr(result["deflated_sharpe"]),
        )
