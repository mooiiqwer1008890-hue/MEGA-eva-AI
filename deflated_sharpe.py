"""
deflated_sharpe.py
==================
Production-grade Probabilistic Sharpe Ratio (PSR) and
Deflated Sharpe Ratio (DSR).

Scientific basis
----------------
1) Bailey, D. H. & López de Prado, M. (2012).
   "The Sharpe Ratio Efficient Frontier", Journal of Risk, 15(2), 3-44.
2) Bailey, D. H. & López de Prado, M. (2014).
   "The Deflated Sharpe Ratio: Correcting for Selection Bias,
   Backtest Overfitting and Non-Normality", Journal of Portfolio Management,
   40(5), 94-107. DOI: 10.3905/jpm.2014.40.5.094.
3) Bailey, D. H., Borwein, J., López de Prado, M. & Zhu, Q. J. (2017).
   "The Probability of Backtest Overfitting", Journal of Computational Finance.

What this module does
---------------------
- Computes the per-period Sharpe ratio correctly.
- Computes PSR while accounting for sample size, skewness and kurtosis.
- Computes the DSR by replacing the PSR benchmark with the expected maximum
  Sharpe produced by multiple strategy trials.
- Uses the *cross-trial variance of Sharpe ratios* when available.
- Can estimate the effective number of independent trials from a matrix of
  strategy return series using the average-correlation construction described
  by Bailey & López de Prado (2014).
- Includes an exact-normal order-statistic refinement for small trial counts,
  while retaining the Bailey-López de Prado asymptotic formula for larger N.
- Computes Minimum Track Record Length (MinTRL).
- Returns detailed diagnostics without changing the simple float-return API.

Important statistical interpretation
------------------------------------
DSR is NOT the probability that "the strategy is real" in an absolute sense.
It is the PSR evaluated at the expected maximum Sharpe ratio implied by the
multiple-testing model. A high DSR means the observed Sharpe is unlikely to be
explained by the model's null + selection mechanism alone. It does not prove
future profitability, remove regime shifts, transaction costs, leakage, or
backtest overfitting outside the assumptions of the model.

Units
-----
All Sharpe ratios used in the core DSR/PSR equations are PER-PERIOD Sharpe
ratios. For daily data this means daily SR. Annualize only for presentation:
annualized_SR = per_period_SR * sqrt(periods_per_year).

Dependencies
------------
numpy
scipy
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Optional

import numpy as np
from scipy.integrate import quad
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import norm
from scipy.stats import skew as scipy_skew

log = logging.getLogger("deflated_sharpe")

# Euler-Mascheroni constant used by the Bailey-López de Prado approximation.
EULER_MASCHERONI = 0.5772156649015329

_EPS = np.finfo(float).eps


@dataclass(frozen=True)
class DSRResult:
    """Detailed DSR diagnostics."""

    observed_sr: float
    expected_max_sr: float
    dsr: float
    dsr_z: float
    benchmark_sr: float
    n_obs: int
    n_trials_raw: int
    n_trials_effective: float
    variance_sr: float
    sr_std_across_trials: float
    skew: float
    kurtosis: float
    psr: float
    min_track_record_length: Optional[float]
    annualized_observed_sr: Optional[float]
    annualized_expected_max_sr: Optional[float]
    variance_source: str
    trials_method: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Validation / preprocessing
# ---------------------------------------------------------------------------

def _clean_1d(values: Iterable[float], *, name: str = "values") -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError(f"{name} contains no finite observations")
    return arr


def _validate_positive_int(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    ivalue = int(value)
    if ivalue != value or ivalue < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return ivalue


def _validate_probability(value: float, name: str) -> float:
    value = float(value)
    if not (0.0 < value < 1.0):
        raise ValueError(f"{name} must be strictly between 0 and 1")
    return value


# ---------------------------------------------------------------------------
# Returns / moments
# ---------------------------------------------------------------------------

def sharpe_ratio(
    returns: Iterable[float],
    risk_free_rate: float = 0.0,
) -> float:
    """Per-period sample Sharpe ratio.

    Parameters
    ----------
    returns
        Strategy returns at a single, consistent sampling frequency.
    risk_free_rate
        Risk-free return at the SAME frequency as ``returns``.
    """
    r = _clean_1d(returns, name="returns")
    if r.size < 2:
        raise ValueError("At least 2 return observations are required")

    excess = r - float(risk_free_rate)
    sigma = np.std(excess, ddof=1)
    if not np.isfinite(sigma) or sigma <= _EPS:
        raise ValueError("Return volatility is zero/too small; Sharpe is undefined")
    return float(np.mean(excess) / sigma)


def return_moments(returns: Iterable[float]) -> tuple[float, float]:
    """Return sample skewness and RAW kurtosis (normal distribution => 3)."""
    r = _clean_1d(returns, name="returns")
    if r.size < 4:
        raise ValueError("At least 4 observations are recommended for skew/kurtosis")

    # bias=False gives finite-sample adjusted estimates. fisher=False is raw
    # kurtosis, exactly the convention required by the PSR/DSR equation.
    g3 = float(scipy_skew(r, bias=False))
    g4 = float(scipy_kurtosis(r, fisher=False, bias=False))

    if not np.isfinite(g3) or not np.isfinite(g4):
        raise ValueError("Skewness/kurtosis could not be estimated")
    return g3, g4


def sharpe_variance_factor(
    observed_sr: float,
    skew: float,
    kurtosis: float,
) -> float:
    """The higher-moment factor appearing in the PSR/DSR denominator."""
    sr = float(observed_sr)
    g3 = float(skew)
    g4 = float(kurtosis)

    # Bailey & López de Prado: 1 - g3*SR + ((g4-1)/4)*SR^2.
    factor = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * (sr * sr)

    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError(
            "PSR/DSR variance factor is non-positive. "
            "Check moments, sample size, and numerical stability."
        )
    return float(factor)


def sharpe_standard_error(
    observed_sr: float,
    skew: float,
    kurtosis: float,
    n_obs: int,
) -> float:
    """Asymptotic standard error of the Sharpe-ratio estimator."""
    t = _validate_positive_int(n_obs, "n_obs", minimum=2)
    factor = sharpe_variance_factor(observed_sr, skew, kurtosis)
    return float(math.sqrt(factor / (t - 1.0)))


# ---------------------------------------------------------------------------
# PSR
# ---------------------------------------------------------------------------

def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    n_obs: int = 252,
) -> float:
    """Probabilistic Sharpe Ratio.

    Returns P(true SR > benchmark_sr) under the PSR approximation.
    """
    t = _validate_positive_int(n_obs, "n_obs", minimum=2)
    sr = float(observed_sr)
    benchmark = float(benchmark_sr)
    se = sharpe_standard_error(sr, float(skew), float(kurtosis), t)
    z = (sr - benchmark) / se
    return float(norm.cdf(z))


# ---------------------------------------------------------------------------
# Multiple testing: expected maximum Sharpe
# ---------------------------------------------------------------------------

def _expected_max_standard_normal_asymptotic(n_trials: int) -> float:
    """Bailey-López de Prado large-N approximation for E[max(Z_1,...,Z_N)]."""
    n = _validate_positive_int(n_trials, "n_trials", minimum=2)

    # norm.isf(q) is numerically safer than norm.ppf(1-q) for very small q.
    z1 = norm.isf(1.0 / n)
    z2 = norm.isf(1.0 / (n * math.e))
    return float(
        (1.0 - EULER_MASCHERONI) * z1
        + EULER_MASCHERONI * z2
    )


def _expected_max_standard_normal_exact(n_trials: int) -> float:
    """Numerically exact E[max of N iid standard Normals].

    E[max Z] = N * integral x*phi(x)*Phi(x)^(N-1) dx.
    This is used only for small N where the asymptotic formula has its
    largest approximation error.
    """
    n = _validate_positive_int(n_trials, "n_trials", minimum=1)
    if n == 1:
        return 0.0

    def integrand(x: float) -> float:
        log_pdf = norm.logpdf(x)
        log_cdf = norm.logcdf(x)
        if not np.isfinite(log_cdf):
            return 0.0
        # Work in log space to avoid under/overflow when N is modest but x is
        # far in the tail.
        log_value = math.log(n) + math.log(max(abs(x), _EPS)) + log_pdf + (n - 1) * log_cdf
        value = math.exp(log_value)
        return value if x >= 0 else -value

    # quad over (-inf, inf) is robust for the modest N used in auto mode.
    result, error = quad(integrand, -np.inf, np.inf, epsabs=1e-10, epsrel=1e-10, limit=250)
    if not np.isfinite(result):
        raise FloatingPointError(f"Exact expected maximum integration failed: error={error}")
    return float(result)


def expected_max_sharpe(
    n_trials: float,
    variance_sr: float,
    mean_sr: float = 0.0,
    method: str = "auto",
    exact_small_n_threshold: int = 25,
) -> float:
    """Expected maximum Sharpe across multiple null trials.

    Parameters
    ----------
    n_trials
        Number of *independent/effective* trials.
    variance_sr
        Variance of estimated Sharpe ratios ACROSS trials.
        This is NOT the variance of the raw return series.
    mean_sr
        Mean of the cross-trial Sharpe distribution. For the DSR null,
        use 0 unless a different null is explicitly justified.
    method
        'auto', 'bailey' or 'exact_normal'.
    exact_small_n_threshold
        In auto mode, exact-normal integration is used for N <= threshold.
    """
    n = float(n_trials)
    if not np.isfinite(n) or n < 1.0:
        raise ValueError("n_trials must be finite and >= 1")
    var_sr = float(variance_sr)
    mu = float(mean_sr)
    if not np.isfinite(var_sr) or var_sr < 0.0:
        raise ValueError("variance_sr must be finite and >= 0")
    sigma = math.sqrt(var_sr)

    if n == 1 or sigma == 0.0:
        return mu

    if method not in {"auto", "bailey", "exact_normal"}:
        raise ValueError("method must be 'auto', 'bailey', or 'exact_normal'")

    is_integer_n = abs(n - round(n)) <= 1e-12
    if method == "exact_normal" or (method == "auto" and is_integer_n and n <= exact_small_n_threshold):
        if not is_integer_n:
            raise ValueError("exact_normal requires an integer n_trials")
        zmax = _expected_max_standard_normal_exact(int(round(n)))
    else:
        # Effective trial counts derived from dependence can be non-integer.
        # The Bailey-López de Prado EVT approximation extends naturally to
        # any positive real N and avoids an artificial rounding step.
        zmax = _expected_max_standard_normal_asymptotic(n)

    return float(mu + sigma * zmax)


# ---------------------------------------------------------------------------
# Cross-trial estimation / dependence
# ---------------------------------------------------------------------------

def cross_trial_variance(sharpe_estimates: Iterable[float]) -> float:
    """Sample variance of Sharpe estimates across strategy trials."""
    sr = _clean_1d(sharpe_estimates, name="sharpe_estimates")
    if sr.size < 2:
        raise ValueError("At least 2 trial Sharpe estimates are required")
    variance = float(np.var(sr, ddof=1))
    if not np.isfinite(variance) or variance < 0.0:
        raise ValueError("Invalid cross-trial Sharpe variance")
    return variance


def effective_trial_count_from_correlation(
    strategy_returns: np.ndarray,
    *,
    clip_to_raw_trials: bool = True,
) -> tuple[float, float]:
    """Estimate effective independent trial count from average correlation.

    Parameters
    ----------
    strategy_returns
        2-D array with shape (T, M): rows=time, columns=strategy trials.
    clip_to_raw_trials
        Negative average correlations can make the interpolation formula
        exceed M. Because this quantity is intended as the number of
        non-redundant trials in a finite search set, clipping to [1, M] is a
        conservative engineering choice.

    Returns
    -------
    (n_effective, average_correlation)

    Notes
    -----
    Bailey & López de Prado derive an implied independent-trial count from
    the average correlation. The method is an approximation and can become
    unstable when M is large relative to T, so the function raises a warning
    through the logger in that regime.
    """
    x = np.asarray(strategy_returns, dtype=float)
    if x.ndim != 2:
        raise ValueError("strategy_returns must be a 2-D array: (observations, trials)")

    t, m = x.shape
    if t < 3 or m < 2:
        raise ValueError("Need at least 3 observations and 2 strategy trials")

    # Rows with missing/non-finite values cannot be used consistently for a
    # correlation matrix. We remove any row containing a non-finite value.
    finite_rows = np.all(np.isfinite(x), axis=1)
    x = x[finite_rows]
    t = x.shape[0]
    if t < 3:
        raise ValueError("Too few complete observations after removing non-finite rows")

    if m > t:
        log.warning(
            "Number of strategy trials M=%d exceeds observations T=%d. "
            "The correlation matrix may be ill-conditioned; effective trial "
            "count is only a rough approximation.",
            m,
            t,
        )

    corr = np.corrcoef(x, rowvar=False)
    if not np.all(np.isfinite(corr)):
        raise ValueError("Correlation matrix contains non-finite values")

    off_diag = corr[np.triu_indices(m, k=1)]
    rho_bar = float(np.mean(off_diag))

    # Bailey & López de Prado's equicorrelation interpolation:
    #     N_eff ~= M / (1 + (M - 1) * rho_bar)
    denominator = 1.0 + (m - 1.0) * rho_bar
    if denominator <= 0.0:
        raise ValueError(
            "Estimated average correlation is at/near the positive-definiteness "
            "boundary; implied independent trial count is numerically unstable."
        )

    n_eff = float(m / denominator)
    if clip_to_raw_trials:
        n_eff = min(float(m), max(1.0, n_eff))
    else:
        n_eff = max(1.0, n_eff)

    return n_eff, rho_bar


def trial_sharpes_from_returns(
    strategy_returns: np.ndarray,
    risk_free_rate: float = 0.0,
) -> np.ndarray:
    """Compute one per-period Sharpe ratio per strategy column."""
    x = np.asarray(strategy_returns, dtype=float)
    if x.ndim != 2:
        raise ValueError("strategy_returns must be 2-D: (observations, trials)")

    out: list[float] = []
    for j in range(x.shape[1]):
        col = x[:, j]
        col = col[np.isfinite(col)]
        if col.size < 2:
            continue
        try:
            out.append(sharpe_ratio(col, risk_free_rate=risk_free_rate))
        except ValueError:
            # A zero-volatility strategy has no finite Sharpe estimate.
            continue
    if len(out) < 2:
        raise ValueError("Could not obtain at least 2 finite trial Sharpe estimates")
    return np.asarray(out, dtype=float)


# ---------------------------------------------------------------------------
# MinTRL
# ---------------------------------------------------------------------------

def minimum_track_record_length(
    observed_sr: float,
    benchmark_sr: float,
    skew: float,
    kurtosis: float,
    confidence: float = 0.95,
) -> float:
    """Minimum observations required for PSR >= confidence.

    This is the algebraic MinTRL implied by the PSR approximation. It is not a
    guarantee of future performance and does not substitute for out-of-sample
    validation.
    """
    confidence = _validate_probability(confidence, "confidence")
    sr = float(observed_sr)
    benchmark = float(benchmark_sr)

    gap = sr - benchmark
    if gap <= 0.0:
        return math.inf

    factor = sharpe_variance_factor(sr, float(skew), float(kurtosis))
    z = float(norm.ppf(confidence))
    return float(1.0 + factor * (z / gap) ** 2)


# ---------------------------------------------------------------------------
# Core DSR
# ---------------------------------------------------------------------------

def deflated_sharpe_ratio(
    observed_sr: float,
    n_trials: int,
    variance_sr: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    n_obs: int = 252,
    *,
    mean_sr: float = 0.0,
    method: str = "auto",
) -> float:
    """Compute the Deflated Sharpe Ratio as a probability in [0, 1].

    ``variance_sr`` is the variance of estimated Sharpe ratios across the
    strategy trials. Supplying ``1/n_trials`` is NOT a general DSR estimate and
    should be avoided unless independently justified.
    """
    n = _validate_positive_int(n_trials, "n_trials", minimum=1)
    t = _validate_positive_int(n_obs, "n_obs", minimum=2)

    sr = float(observed_sr)
    g3 = float(skew)
    g4 = float(kurtosis)
    factor = sharpe_variance_factor(sr, g3, g4)

    expected_max = expected_max_sharpe(
        n_trials=n,
        variance_sr=variance_sr,
        mean_sr=mean_sr,
        method=method,
    )

    # PSR/DSR z-statistic. Writing it directly avoids needless recomputation.
    denominator = math.sqrt(factor)
    z = (sr - expected_max) * math.sqrt(t - 1.0) / denominator
    dsr = float(norm.cdf(z))

    log.info(
        "[DSR] SR=%.6f | E[max SR]=%.6f | N=%s | Var(SR)=%.6g | T=%d | "
        "Skew=%.4f | Kurt=%.4f | DSR=%.6f",
        sr,
        expected_max,
        n,
        float(variance_sr),
        t,
        g3,
        g4,
        dsr,
    )
    return dsr


def deflated_sharpe_report(
    returns: Iterable[float],
    *,
    n_trials: Optional[int] = None,
    trial_sharpes: Optional[Iterable[float]] = None,
    trial_returns: Optional[np.ndarray] = None,
    variance_sr: Optional[float] = None,
    risk_free_rate: float = 0.0,
    confidence: float = 0.95,
    annualization_factor: Optional[float] = None,
    mean_sr: float = 0.0,
    method: str = "auto",
    use_effective_trials: bool = False,
) -> DSRResult:
    """High-level DSR calculation with a full diagnostic report.

    Preferred inputs, in order:
    1. ``trial_sharpes``: empirical SRs from all tested variants.
    2. ``trial_returns``: strategy return matrix from which SRs and dependence
       can be estimated.
    3. ``variance_sr``: explicitly supplied cross-trial variance.

    If none is supplied, an IID single-strategy variance proxy is used and the
    report labels the source as approximate. This is a fallback, not the ideal
    empirical DSR input.
    """
    r = _clean_1d(returns, name="returns")
    t = r.size
    sr = sharpe_ratio(r, risk_free_rate=risk_free_rate)
    g3, g4 = return_moments(r)

    raw_trials: int
    effective_trials: float
    variance_source: str
    trials_method: str

    trial_sr_array: Optional[np.ndarray] = None
    rho_bar: Optional[float] = None

    if trial_sharpes is not None:
        trial_sr_array = _clean_1d(trial_sharpes, name="trial_sharpes")
        raw_trials = int(trial_sr_array.size)
        if raw_trials < 2:
            raise ValueError("trial_sharpes must contain at least 2 values")
        var_sr = cross_trial_variance(trial_sr_array)
        variance_source = "empirical_cross_trial_variance"
        effective_trials = float(raw_trials)
        trials_method = "raw_trial_count"

    elif trial_returns is not None:
        x = np.asarray(trial_returns, dtype=float)
        if x.ndim != 2:
            raise ValueError("trial_returns must be 2-D: (observations, trials)")
        raw_trials = int(x.shape[1])
        trial_sr_array = trial_sharpes_from_returns(x, risk_free_rate=risk_free_rate)
        var_sr = cross_trial_variance(trial_sr_array)
        variance_source = "empirical_trial_returns"

        if use_effective_trials:
            effective_trials, rho_bar = effective_trial_count_from_correlation(x)
            trials_method = f"effective_from_average_correlation(rho={rho_bar:.6f})"
        else:
            effective_trials = float(raw_trials)
            trials_method = "raw_trial_count"

    elif variance_sr is not None:
        if n_trials is None:
            raise ValueError("n_trials is required when variance_sr is supplied")
        raw_trials = _validate_positive_int(n_trials, "n_trials", minimum=1)
        effective_trials = float(raw_trials)
        var_sr = float(variance_sr)
        variance_source = "user_supplied_cross_trial_variance"
        trials_method = "raw_trial_count"

    else:
        if n_trials is None:
            raise ValueError(
                "Provide trial_sharpes, trial_returns, or both n_trials and variance_sr"
            )
        raw_trials = _validate_positive_int(n_trials, "n_trials", minimum=1)

        # Fallback: asymptotic single-strategy SR-estimation variance.
        # This is NOT the same quantity as empirical cross-trial variance.
        factor = sharpe_variance_factor(sr, g3, g4)
        var_sr = float(factor / (t - 1.0))
        variance_source = "single_strategy_asymptotic_proxy_APPROXIMATE"
        effective_trials = float(raw_trials)
        trials_method = "raw_trial_count"
        log.warning(
            "DSR fallback: cross-trial Sharpe variance was not supplied. "
            "Using the single-strategy asymptotic variance as a proxy. "
            "For a strict DSR, supply trial_sharpes/trial_returns or variance_sr."
        )

    if n_trials is not None and trial_sharpes is not None:
        requested = _validate_positive_int(n_trials, "n_trials", minimum=1)
        if requested != raw_trials:
            log.warning(
                "Ignoring n_trials=%d because trial_sharpes contains %d actual trials.",
                requested,
                raw_trials,
            )

    expected_max = expected_max_sharpe(
        n_trials=effective_trials,
        variance_sr=var_sr,
        mean_sr=mean_sr,
        method=method,
    )

    factor = sharpe_variance_factor(sr, g3, g4)
    z = (sr - expected_max) * math.sqrt(t - 1.0) / math.sqrt(factor)
    dsr = float(norm.cdf(z))
    psr = probabilistic_sharpe_ratio(sr, benchmark_sr=expected_max, skew=g3, kurtosis=g4, n_obs=t)

    min_trl = minimum_track_record_length(
        observed_sr=sr,
        benchmark_sr=expected_max,
        skew=g3,
        kurtosis=g4,
        confidence=confidence,
    )

    annualized_sr: Optional[float] = None
    annualized_expected: Optional[float] = None
    if annualization_factor is not None:
        af = float(annualization_factor)
        if not np.isfinite(af) or af <= 0.0:
            raise ValueError("annualization_factor must be > 0")
        root_af = math.sqrt(af)
        annualized_sr = sr * root_af
        annualized_expected = expected_max * root_af

    return DSRResult(
        observed_sr=float(sr),
        expected_max_sr=float(expected_max),
        dsr=float(dsr),
        dsr_z=float(z),
        benchmark_sr=float(expected_max),
        n_obs=int(t),
        n_trials_raw=int(raw_trials),
        n_trials_effective=float(effective_trials),
        variance_sr=float(var_sr),
        sr_std_across_trials=float(math.sqrt(max(var_sr, 0.0))),
        skew=float(g3),
        kurtosis=float(g4),
        psr=float(psr),
        min_track_record_length=float(min_trl) if np.isfinite(min_trl) else None,
        annualized_observed_sr=annualized_sr,
        annualized_expected_max_sr=annualized_expected,
        variance_source=variance_source,
        trials_method=trials_method,
    )


def deflated_sharpe_ratio_from_returns(
    returns: Iterable[float],
    n_trials: Optional[int] = None,
    variance_sr: Optional[float] = None,
    *,
    trial_sharpes: Optional[Iterable[float]] = None,
    trial_returns: Optional[np.ndarray] = None,
    risk_free_rate: float = 0.0,
    mean_sr: float = 0.0,
    method: str = "auto",
    use_effective_trials: bool = False,
) -> float:
    """Compatibility wrapper: return only the DSR probability."""
    report = deflated_sharpe_report(
        returns,
        n_trials=n_trials,
        trial_sharpes=trial_sharpes,
        trial_returns=trial_returns,
        variance_sr=variance_sr,
        risk_free_rate=risk_free_rate,
        mean_sr=mean_sr,
        method=method,
        use_effective_trials=use_effective_trials,
    )
    return report.dsr


# ---------------------------------------------------------------------------
# Self-test / demonstration
# ---------------------------------------------------------------------------

def _run_self_test() -> None:
    """Internal numerical sanity checks."""
    rng = np.random.default_rng(42)

    # Pure-noise strategies: the best SR should suffer a multiple-testing penalty.
    t = 252
    m = 100
    returns = rng.normal(0.0005, 0.01, size=t)
    trials = rng.normal(0.0, 0.01, size=(t, m))
    trial_srs = trial_sharpes_from_returns(trials)

    # Formula sanity: for a single trial, E[max] under a zero-mean null is zero.
    assert abs(expected_max_sharpe(1, 0.25)) < 1e-15

    # Exact normal expected max should be close to Monte Carlo for small N.
    n_small = 10
    exact = _expected_max_standard_normal_exact(n_small)
    sim = np.max(rng.normal(size=(200_000, n_small)), axis=1).mean()
    assert abs(exact - sim) < 0.02, (exact, sim)

    # PSR at its own benchmark should be 0.5.
    psr = probabilistic_sharpe_ratio(
        observed_sr=0.5,
        benchmark_sr=0.5,
        skew=0.0,
        kurtosis=3.0,
        n_obs=252,
    )
    assert abs(psr - 0.5) < 1e-12

    report = deflated_sharpe_report(
        returns,
        trial_sharpes=trial_srs,
        confidence=0.95,
        annualization_factor=252,
    )
    assert 0.0 <= report.dsr <= 1.0
    assert report.n_trials_raw == m
    assert report.variance_source == "empirical_cross_trial_variance"

    # Effective-trial estimate must remain positive and no larger than M under
    # the conservative clipping convention.
    n_eff, rho = effective_trial_count_from_correlation(trials)
    assert 1.0 <= n_eff <= m
    assert -1.0 <= rho <= 1.0

    # MinTRL is infinite when the observed SR is not above the benchmark.
    assert math.isinf(minimum_track_record_length(0.5, 0.5, 0.0, 3.0))

    print("DSR self-test: PASS")
    print(f"Observed SR       : {report.observed_sr:.6f}")
    print(f"Expected Max SR   : {report.expected_max_sr:.6f}")
    print(f"DSR               : {report.dsr:.6f}")
    print(f"PSR@DSR benchmark : {report.psr:.6f}")
    print(f"Trials            : {report.n_trials_raw}")
    print(f"Variance(SR)      : {report.variance_sr:.8f}")
    print(f"MinTRL            : {report.min_track_record_length}")
    print(f"Effective trials  : {n_eff:.4f} (rho_bar={rho:.4f})")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _run_self_test()
