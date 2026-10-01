"""
backtest_zscore_pro.py
======================
Research-grade mean-reversion backtester for BTC/USDT spot.

Core design:
    1) Look-ahead-safe Binance OHLCV acquisition with pagination.
    2) Log-price rolling Z-score (standard or robust/MAD).
    3) Volatility-scaled Triple-Barrier labeling.
    4) Realistic entry timing: signal at bar close -> entry at next bar open.
    5) Transaction costs + slippage.
    6) Conservative same-bar TP/SL ambiguity handling.
    7) Non-overlapping single-position portfolio simulation by default.
    8) Portfolio-level daily equity curve and risk statistics.
    9) Probabilistic Sharpe Ratio (PSR).
   10) Deflated Sharpe Ratio (classic Bailey-López de Prado formulation).
   11) Purged K-Fold CV using label intervals + embargo.
   12) Anchored Walk-Forward CV for genuine temporal OOS testing.
   13) Optional Combinatorial Purged CV / PBO-style search diagnostic.
   14) Parameter-search accounting: all tested configurations are counted.

Scientific references:
    - Bailey & López de Prado (2012/2013), The Sharpe Ratio Efficient Frontier.
    - Bailey & López de Prado (2014), The Deflated Sharpe Ratio.
    - López de Prado (2018), Advances in Financial Machine Learning,
      Ch. 3 (Triple Barrier), Ch. 7 (Purged K-Fold / Embargo), Ch. 11-12.
    - Bailey et al. (2017), The Probability of Backtest Overfitting.
    - Arian, Norouzi Mobarekeh & Seco (2024),
      Backtest overfitting in the machine learning era.
    - López de Prado & Porcu (2026),
      The Deflated Sharpe Ratio: A Unified Framework for Search-Adjusted
      Performance Inference. This file deliberately uses the well-defined
      classic DSR location benchmark rather than inventing an unverified
      implementation of the newer location-scale/exact variants.

Important:
    This is a research/backtesting engine, not a guarantee of live profitability.
    Results depend critically on data quality, fee schedule, slippage, search
    count, and the choice of validation protocol.
"""

from __future__ import annotations

import itertools
import logging
import math
import time
from dataclasses import dataclass, asdict
from datetime import timedelta
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from scipy.stats import kurtosis, norm, skew


# ============================================================================
# CONFIGURATION
# ============================================================================

SYMBOL = "BTCUSDT"
INTERVAL = "15m"
TOTAL_BARS = 5000

# Signal / volatility model
Z_PERIOD = 40
Z_THRESHOLD = 2.0
ZSCORE_METHOD = "standard"       # "standard" or "robust"
VOL_METHOD = "ewma"              # "ewma" or "close"
VOL_WINDOW = 48
VOL_EWMA_SPAN = 48

# Triple barrier
PT_MULT = 2.0
SL_MULT = 1.0
MAX_HOLDING = 20                 # bars after entry
SAME_BAR_POLICY = "worst"        # "worst" or "best"; long-only engine

# Execution assumptions (CHANGE THESE to your actual Binance tier)
FEE_BPS_PER_SIDE = 10.0
SLIPPAGE_BPS_PER_SIDE = 5.0

# Portfolio behaviour
ALLOW_OVERLAP = False             # False = one position at a time
MIN_SIGNAL_GAP = 1                # bars between accepted signals

# Statistical inference / search accounting
N_TRIALS = 100                    # MUST represent real research/search count
MIN_OBS_FOR_STATS = 30

# Validation
WALK_FORWARD_SPLITS = 5
WALK_FORWARD_MIN_TRAIN_RATIO = 0.50
PURGED_KFOLD_SPLITS = 5
EMBARGO_BARS = MAX_HOLDING

# Optional CPCV / PBO-style diagnostic. Expensive, so disabled by default.
RUN_CPCV_PBO = False
CPCV_GROUPS = 6
CPCV_TEST_GROUPS = 2

# Parameter search used by walk-forward/CPCV.
# Every tested configuration contributes to multiple-testing/search burden.
SEARCH_Z_THRESHOLDS = (1.5, 2.0, 2.5)
SEARCH_PT_MULTS = (1.5, 2.0, 2.5)
SEARCH_SL_MULTS = (0.75, 1.0, 1.5)

# Output
SAVE_CSV = True
TRADE_CSV = "backtest_zscore_trades.csv"
EQUITY_CSV = "backtest_zscore_equity.csv"
WF_CSV = "backtest_zscore_walk_forward.csv"

REQUEST_LIMIT = 1000
REQUEST_TIMEOUT = 15
MAX_RETRIES = 4
API_URL = "https://data-api.binance.vision/api/v3/klines"

log = logging.getLogger("backtest_zscore_pro")


# ============================================================================
# DATA TYPES
# ============================================================================

@dataclass(frozen=True)
class StrategyParams:
    z_threshold: float
    pt_mult: float
    sl_mult: float


@dataclass
class Trade:
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    volatility: float
    pt_level: float
    sl_level: float
    label: int
    exit_reason: str
    holding_bars: int
    gross_return: float
    net_return: float
    mfe: float
    mae: float


# ============================================================================
# BINANCE DATA
# ============================================================================

INTERVAL_MS = {
    "1s": 1_000,
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
}


def interval_ms(interval: str) -> int:
    if interval not in INTERVAL_MS:
        raise ValueError(f"Unsupported interval: {interval}")
    return INTERVAL_MS[interval]


def _request_json(session: requests.Session, params: dict) -> list:
    last_error: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(API_URL, params=params, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            payload = r.json()
            if not isinstance(payload, list):
                raise RuntimeError(f"Unexpected Binance response: {payload}")
            return payload
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            sleep_s = 0.75 * (2 ** attempt)
            log.warning("Binance request failed (%s); retrying in %.2fs", exc, sleep_s)
            time.sleep(sleep_s)
    raise RuntimeError(f"Binance data request failed after retries: {last_error}")


def fetch_data(symbol: str, interval: str, total_bars: int) -> pd.DataFrame:
    """Fetch exactly up to total_bars closed candles using paginated requests."""
    if total_bars < 100:
        raise ValueError("total_bars should be >= 100 for a useful research sample")

    step = interval_ms(interval)
    rows: List[list] = []
    end_time: Optional[int] = None

    with requests.Session() as session:
        session.headers.update({"User-Agent": "research-backtester/1.0"})

        while len(rows) < total_bars:
            remaining = total_bars - len(rows)
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": min(REQUEST_LIMIT, remaining),
            }
            if end_time is not None:
                params["endTime"] = end_time

            batch = _request_json(session, params)
            if not batch:
                break

            # API returns chronological order.
            if rows:
                # We paginated backwards using endTime; prepend this batch.
                rows = batch + rows
            else:
                rows = batch

            if len(batch) == 0:
                break

            oldest_open = int(batch[0][0])
            end_time = oldest_open - 1

            if len(batch) < min(REQUEST_LIMIT, remaining):
                break

            time.sleep(0.05)

    rows = rows[-total_bars:]
    if not rows:
        raise RuntimeError("Binance returned no candles")

    cols = [
        "time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(rows, columns=cols)

    numeric_cols = [
        "open", "high", "low", "close", "volume",
        "quote_volume", "taker_buy_base", "taker_buy_quote",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trades"] = pd.to_numeric(df["trades"], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    df = df.sort_values("time").drop_duplicates("time").set_index("time")

    # Never use a candle that is still forming.
    now = pd.Timestamp.now(tz="UTC")
    df = df[df["close_time"] <= now]

    if len(df) < 100:
        raise RuntimeError(f"Too few closed candles after cleaning: {len(df)}")

    return df


# ============================================================================
# FEATURES
# ============================================================================

def rolling_zscore(series: pd.Series, period: int, method: str = "standard") -> pd.Series:
    """Rolling z-score of log-price, avoiding raw-price scale dependence."""
    x = np.log(series.replace(0, np.nan))

    if method == "standard":
        mean = x.rolling(period, min_periods=period).mean()
        std = x.rolling(period, min_periods=period).std(ddof=1)
        return (x - mean) / std.replace(0, np.nan)

    if method == "robust":
        med = x.rolling(period, min_periods=period).median()
        mad = x.rolling(period, min_periods=period).apply(
            lambda a: np.median(np.abs(a - np.median(a))), raw=True
        )
        robust_std = 1.4826 * mad
        return (x - med) / robust_std.replace(0, np.nan)

    raise ValueError("method must be 'standard' or 'robust'")


def garman_klass_volatility(df: pd.DataFrame, window: int) -> pd.Series:
    """Rolling annualization-free Garman-Klass volatility estimate."""
    o = np.log(df["open"])
    h = np.log(df["high"])
    l = np.log(df["low"])
    c = np.log(df["close"])

    var = 0.5 * (h - l) ** 2 - (2.0 * np.log(2.0) - 1.0) * (c - o) ** 2
    var = var.clip(lower=0.0)
    return np.sqrt(var.rolling(window, min_periods=window).mean())


def realized_volatility(df: pd.DataFrame, method: str, window: int, span: int) -> pd.Series:
    log_ret = np.log(df["close"]).diff()

    if method == "close":
        return log_ret.rolling(window, min_periods=window).std(ddof=1)

    if method == "ewma":
        # EWM volatility reacts faster to regime changes than a simple rolling SD.
        return log_ret.ewm(span=span, adjust=False, min_periods=span).std(bias=False)

    if method == "gk":
        return garman_klass_volatility(df, window)

    raise ValueError("VOL_METHOD must be 'close', 'ewma', or 'gk'")


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_close"] = np.log(out["close"])
    out["zscore"] = rolling_zscore(out["close"], Z_PERIOD, ZSCORE_METHOD)
    out["volatility"] = realized_volatility(
        out, VOL_METHOD, VOL_WINDOW, VOL_EWMA_SPAN
    )
    out["next_open"] = out["open"].shift(-1)
    return out


# ============================================================================
# TRIPLE-BARRIER SIMULATOR
# ============================================================================

def _effective_entry_price(raw: float, side: int) -> float:
    slip = SLIPPAGE_BPS_PER_SIDE / 10_000.0
    fee = FEE_BPS_PER_SIDE / 10_000.0
    if side == 1:
        return raw * (1.0 + slip) * (1.0 + fee)
    return raw * (1.0 - slip) * (1.0 - fee)


def _effective_exit_price(raw: float, side: int) -> float:
    slip = SLIPPAGE_BPS_PER_SIDE / 10_000.0
    fee = FEE_BPS_PER_SIDE / 10_000.0
    if side == 1:
        return raw * (1.0 - slip) * (1.0 - fee)
    return raw * (1.0 + slip) * (1.0 + fee)


def net_return(entry: float, exit: float, side: int = 1) -> float:
    e = _effective_entry_price(entry, side)
    x = _effective_exit_price(exit, side)
    if side == 1:
        return x / e - 1.0
    return e / x - 1.0


def _barrier_levels(entry: float, vol: float, pt_mult: float, sl_mult: float, side: int) -> Tuple[float, float]:
    # Work in log-return space: barriers scale with current volatility.
    pt_log = max(pt_mult * vol, 1e-8)
    sl_log = max(sl_mult * vol, 1e-8)
    if side == 1:
        return entry * math.exp(pt_log), entry * math.exp(-sl_log)
    return entry * math.exp(-pt_log), entry * math.exp(sl_log)


def signal_indices(df: pd.DataFrame, params: StrategyParams, side: int = 1) -> np.ndarray:
    if side == 1:
        mask = df["zscore"] <= -abs(params.z_threshold)
    else:
        mask = df["zscore"] >= abs(params.z_threshold)
    return np.flatnonzero(mask.fillna(False).to_numpy())


def simulate_strategy(
    df: pd.DataFrame,
    params: StrategyParams,
    side: int = 1,
    start_time: Optional[pd.Timestamp] = None,
    end_time: Optional[pd.Timestamp] = None,
    allow_overlap: bool = ALLOW_OVERLAP,
) -> List[Trade]:
    """
    Simulate entries generated at signal-bar close and executed at next-bar open.

    A trade is included only when both entry and exit are inside the requested
    evaluation window. This is essential for clean train/test accounting.
    """
    if side not in (1, -1):
        raise ValueError("Only long (+1) and short (-1) are supported")

    idx = df.index
    positions = signal_indices(df, params, side)
    trades: List[Trade] = []
    next_allowed_entry = -1

    for sig_pos in positions:
        entry_pos = sig_pos + 1
        if entry_pos >= len(df):
            continue

        if not allow_overlap and entry_pos <= next_allowed_entry:
            continue

        signal_time = idx[sig_pos]
        entry_time = idx[entry_pos]

        if start_time is not None and entry_time < start_time:
            continue
        if end_time is not None and entry_time > end_time:
            continue

        vol = float(df["volatility"].iloc[sig_pos])
        if not np.isfinite(vol) or vol <= 0:
            continue

        entry_price = float(df["open"].iloc[entry_pos])
        pt_level, sl_level = _barrier_levels(
            entry_price, vol, params.pt_mult, params.sl_mult, side
        )

        last_pos = min(entry_pos + MAX_HOLDING, len(df) - 1)
        if last_pos <= entry_pos:
            continue

        # A test/training split must never use an exit after its boundary.
        if end_time is not None:
            eligible = np.flatnonzero(idx[entry_pos:last_pos + 1] <= end_time)
            if len(eligible) == 0:
                continue
            last_pos = entry_pos + int(eligible[-1])
            if last_pos <= entry_pos:
                continue

        side_high = df["high"].to_numpy()
        side_low = df["low"].to_numpy()
        closes = df["close"].to_numpy()

        exit_pos: Optional[int] = None
        exit_reason = "vertical"
        label = 0

        for pos in range(entry_pos, last_pos + 1):
            hi = float(side_high[pos])
            lo = float(side_low[pos])

            if side == 1:
                hit_pt = hi >= pt_level
                hit_sl = lo <= sl_level
            else:
                hit_pt = lo <= pt_level
                hit_sl = hi >= sl_level

            if hit_pt and hit_sl:
                # OHLC does not identify intrabar order. For scientific
                # backtesting, default to the adverse ordering rather than
                # silently assuming the profitable ordering.
                if SAME_BAR_POLICY == "worst":
                    exit_reason = "sl_same_bar_ambiguous"
                    label = -1
                    exit_pos = pos
                else:
                    exit_reason = "tp_same_bar_ambiguous"
                    label = 1
                    exit_pos = pos
                break

            if hit_pt:
                exit_reason = "pt"
                label = 1
                exit_pos = pos
                break

            if hit_sl:
                exit_reason = "sl"
                label = -1
                exit_pos = pos
                break

        if exit_pos is None:
            exit_pos = last_pos
            label = 0
            exit_reason = "vertical"

        exit_time = idx[exit_pos]
        if end_time is not None and exit_time > end_time:
            continue

        exit_price = pt_level if exit_reason == "pt" else sl_level if exit_reason in {
            "sl", "sl_same_bar_ambiguous"
        } else float(closes[exit_pos])

        # If an assumed barrier is crossed inside an OHLC candle, we use the
        # barrier itself, not the candle close. That avoids overstating returns.
        if exit_reason == "pt":
            exit_price = pt_level
        elif exit_reason in {"sl", "sl_same_bar_ambiguous"}:
            exit_price = sl_level
        else:
            exit_price = float(closes[exit_pos])

        gross = (exit_price / entry_price - 1.0) if side == 1 else (entry_price / exit_price - 1.0)
        net = net_return(entry_price, exit_price, side)

        path_high = df["high"].iloc[entry_pos:exit_pos + 1].to_numpy(dtype=float)
        path_low = df["low"].iloc[entry_pos:exit_pos + 1].to_numpy(dtype=float)
        if side == 1:
            mfe = float(np.max(path_high / entry_price - 1.0))
            mae = float(np.min(path_low / entry_price - 1.0))
        else:
            mfe = float(np.max(entry_price / path_low - 1.0))
            mae = float(np.min(entry_price / path_high - 1.0))

        trade = Trade(
            signal_time=signal_time,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            volatility=vol,
            pt_level=pt_level,
            sl_level=sl_level,
            label=label,
            exit_reason=exit_reason,
            holding_bars=exit_pos - entry_pos + 1,
            gross_return=float(gross),
            net_return=float(net),
            mfe=mfe,
            mae=mae,
        )
        trades.append(trade)

        if not allow_overlap:
            next_allowed_entry = exit_pos + MIN_SIGNAL_GAP

    return trades


def trades_to_frame(trades: Sequence[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=[f.name for f in Trade.__dataclass_fields__.values()])
    return pd.DataFrame([asdict(t) for t in trades]).sort_values("entry_time").reset_index(drop=True)


# ============================================================================
# EQUITY CURVE
# ============================================================================

def build_equity_curve(
    df: pd.DataFrame,
    trades: pd.DataFrame,
    side: int = 1,
) -> pd.Series:
    """
    Mark-to-market equity curve.

    Entry costs are charged from entry; exit costs are charged at actual exit.
    Between trades the portfolio remains flat.
    """
    equity = pd.Series(1.0, index=df.index, dtype=float)
    if trades.empty:
        return equity

    current_equity = 1.0
    last_pos = 0
    idx = df.index

    for _, tr in trades.sort_values("entry_time").iterrows():
        entry_time = pd.Timestamp(tr["entry_time"])
        exit_time = pd.Timestamp(tr["exit_time"])

        entry_loc = idx.get_indexer([entry_time])[0]
        exit_loc = idx.get_indexer([exit_time])[0]
        if entry_loc < 0 or exit_loc < 0 or exit_loc < entry_loc:
            continue

        entry_raw = float(tr["entry_price"])
        entry_eff = _effective_entry_price(entry_raw, side)

        # Flat before entry.
        equity.iloc[last_pos:entry_loc] = current_equity

        closes = df["close"].iloc[entry_loc:exit_loc + 1].to_numpy(dtype=float)
        if side == 1:
            marked = current_equity * (closes / entry_eff)
        else:
            marked = current_equity * (entry_eff / closes)

        equity.iloc[entry_loc:exit_loc + 1] = marked
        current_equity = current_equity * (1.0 + float(tr["net_return"]))
        # Make the exact exit point agree with realized net return.
        equity.iloc[exit_loc] = current_equity
        last_pos = exit_loc + 1

    equity.iloc[last_pos:] = current_equity
    return equity


# ============================================================================
# SHARPE / DSR / RISK METRICS
# ============================================================================

def sharpe_ratio(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 2:
        return np.nan
    sd = np.std(returns, ddof=1)
    return float(np.mean(returns) / sd) if sd > 0 else np.nan


def lag1_autocorrelation(returns: np.ndarray) -> float:
    """First-order autocorrelation used by the 2026 Sharpe-inference framework."""
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return 0.0
    x0 = x[:-1] - np.mean(x[:-1])
    x1 = x[1:] - np.mean(x[1:])
    denom = float(np.sqrt(np.sum(x0 * x0) * np.sum(x1 * x1)))
    if denom <= 0:
        return 0.0
    return float(np.clip(np.sum(x0 * x1) / denom, -0.99, 0.99))


def sharpe_sampling_variance(
    benchmark_sr: float,
    returns: np.ndarray,
    include_autocorrelation: bool = True,
) -> float:
    """
    Approximate Var[SR-hat | SR=benchmark_sr] from the 2026 Sharpe-inference
    framework, including non-normality and an AR(1)-style serial-correlation term.

    When include_autocorrelation=False this reduces to the familiar IID form.
    """
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    t = len(x)
    if t < MIN_OBS_FOR_STATS:
        return np.nan

    g3 = float(skew(x, bias=False))
    k4 = float(kurtosis(x, fisher=False, bias=False))  # Pearson kurtosis.
    rho = lag1_autocorrelation(x) if include_autocorrelation else 0.0

    # López de Prado, Lipton & Zoonekynd (2026), Sharpe variance approximation.
    a = (1.0 + rho) / (1.0 - rho)
    b = (1.0 + rho + rho * rho) / (1.0 - rho * rho)
    c = (1.0 + rho * rho) / (1.0 - rho * rho)

    variance = (
        a
        - b * g3 * benchmark_sr
        + c * ((k4 - 1.0) / 4.0) * benchmark_sr * benchmark_sr
    ) / t
    return float(max(variance, 1e-15))


def probabilistic_sharpe_ratio(
    returns: np.ndarray,
    benchmark_sr: float = 0.0,
    include_autocorrelation: bool = True,
) -> float:
    """
    PSR using the 2026 non-IID/non-Normal Sharpe sampling-variance approximation.
    """
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < MIN_OBS_FOR_STATS:
        return np.nan
    sr = sharpe_ratio(x)
    if not np.isfinite(sr):
        return np.nan

    var = sharpe_sampling_variance(
        benchmark_sr, x, include_autocorrelation=include_autocorrelation
    )
    if not np.isfinite(var) or var <= 0:
        return np.nan
    z = (sr - benchmark_sr) / math.sqrt(var)
    return float(norm.cdf(z))


def expected_max_sharpe(trials: int, sigma_sr: float = 1.0) -> float:
    """Expected maximum Sharpe under the classic DSR search-null approximation."""
    if trials < 2:
        return 0.0
    gamma = 0.5772156649015329
    a = norm.ppf(1.0 - 1.0 / trials)
    b = norm.ppf(1.0 - 1.0 / (trials * math.e))
    return float(sigma_sr * ((1.0 - gamma) * a + gamma * b))


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    benchmark_sr: float = 0.0,
    include_autocorrelation: bool = True,
) -> Dict[str, float]:
    """
    Search-adjusted DSR with a modern non-IID sampling-error correction.

    This is the location-only DSR family (DSR-L): the search raises the null
    benchmark from benchmark_sr to the expected maximum under the null. The
    sampling standard error is then evaluated at that search-adjusted null using
    the 2026 Sharpe-variance approximation for non-Normal/serially-correlated
    returns. This is intentionally NOT the 2026 location-scale/full-search DSR.
    """
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    t = len(x)
    if t < MIN_OBS_FOR_STATS or n_trials < 2:
        return {"dsr": np.nan, "sr": np.nan, "sr_star": np.nan, "sr_se": np.nan}

    sr = sharpe_ratio(x)
    if not np.isfinite(sr):
        return {"dsr": np.nan, "sr": np.nan, "sr_star": np.nan, "sr_se": np.nan}

    # First obtain the standard error under the unselected null, then apply the
    # False Strategy Theorem location adjustment.
    var0 = sharpe_sampling_variance(
        benchmark_sr, x, include_autocorrelation=include_autocorrelation
    )
    sigma0 = math.sqrt(var0) if np.isfinite(var0) else np.nan
    if not np.isfinite(sigma0):
        return {"dsr": np.nan, "sr": float(sr), "sr_star": np.nan, "sr_se": np.nan}

    sr_star = max(benchmark_sr, expected_max_sharpe(n_trials, sigma_sr=sigma0))

    # Recompute the sampling error at the search-adjusted null, as required by
    # the PSR -> DSR construction.
    var_star = sharpe_sampling_variance(
        sr_star, x, include_autocorrelation=include_autocorrelation
    )
    se_star = math.sqrt(var_star) if np.isfinite(var_star) else np.nan
    if not np.isfinite(se_star) or se_star <= 0:
        return {"dsr": np.nan, "sr": float(sr), "sr_star": float(sr_star), "sr_se": np.nan}

    z = (sr - sr_star) / se_star
    return {
        "dsr": float(norm.cdf(z)),
        "sr": float(sr),
        "sr_star": float(sr_star),
        "sr_se": float(se_star),
    }


def max_drawdown(equity: pd.Series) -> float:
    eq = equity.replace([np.inf, -np.inf], np.nan).dropna()
    if eq.empty:
        return np.nan
    dd = eq / eq.cummax() - 1.0
    return float(dd.min())


def cagr(equity: pd.Series) -> float:
    eq = equity.dropna()
    if len(eq) < 2 or eq.iloc[0] <= 0 or eq.iloc[-1] <= 0:
        return np.nan
    days = (eq.index[-1] - eq.index[0]).total_seconds() / 86_400.0
    if days <= 0:
        return np.nan
    return float((eq.iloc[-1] / eq.iloc[0]) ** (365.0 / days) - 1.0)


def sortino_ratio(daily_returns: pd.Series) -> float:
    x = daily_returns.dropna().to_numpy(dtype=float)
    if len(x) < 2:
        return np.nan
    downside = x[x < 0]
    if len(downside) == 0:
        return np.inf if np.mean(x) > 0 else np.nan
    downside_dev = math.sqrt(np.mean(downside ** 2))
    return float(np.mean(x) / downside_dev) if downside_dev > 0 else np.nan


def profit_factor(trade_returns: np.ndarray) -> float:
    wins = trade_returns[trade_returns > 0].sum()
    losses = -trade_returns[trade_returns < 0].sum()
    if losses <= 0:
        return np.inf if wins > 0 else np.nan
    return float(wins / losses)


def compute_metrics(df: pd.DataFrame, trades: pd.DataFrame, equity: pd.Series) -> Dict[str, float]:
    trade_rets = trades["net_return"].to_numpy(dtype=float) if not trades.empty else np.array([])

    daily_equity = equity.resample("1D").last().ffill()
    daily_returns = daily_equity.pct_change().dropna()

    daily_sr = sharpe_ratio(daily_returns.to_numpy())
    dsr = deflated_sharpe_ratio(daily_returns.to_numpy(), N_TRIALS)
    psr = probabilistic_sharpe_ratio(daily_returns.to_numpy())

    mdd = max_drawdown(equity)
    annual_sr = daily_sr * math.sqrt(365.0) if np.isfinite(daily_sr) else np.nan
    cagr_value = cagr(equity)

    labels = trades["label"].to_numpy(dtype=int) if not trades.empty else np.array([])
    hit_rate = float(np.mean(trade_rets > 0)) if len(trade_rets) else np.nan
    tp_rate = float(np.mean(labels == 1)) if len(labels) else np.nan
    sl_rate = float(np.mean(labels == -1)) if len(labels) else np.nan
    vertical_rate = float(np.mean(labels == 0)) if len(labels) else np.nan

    avg_hold = float(trades["holding_bars"].mean()) if not trades.empty else np.nan
    avg_mfe = float(trades["mfe"].mean()) if not trades.empty else np.nan
    avg_mae = float(trades["mae"].mean()) if not trades.empty else np.nan

    return {
        "n_bars": float(len(df)),
        "n_trades": float(len(trades)),
        "net_total_return": float(equity.iloc[-1] - 1.0) if len(equity) else np.nan,
        "cagr": cagr_value,
        "daily_sharpe": daily_sr,
        "annualized_sharpe": annual_sr,
        "sortino_daily": sortino_ratio(daily_returns),
        "profit_factor": profit_factor(trade_rets) if len(trade_rets) else np.nan,
        "win_rate": hit_rate,
        "tp_label_rate": tp_rate,
        "sl_label_rate": sl_rate,
        "vertical_label_rate": vertical_rate,
        "mean_trade_return": float(np.mean(trade_rets)) if len(trade_rets) else np.nan,
        "median_trade_return": float(np.median(trade_rets)) if len(trade_rets) else np.nan,
        "trade_return_std": float(np.std(trade_rets, ddof=1)) if len(trade_rets) > 1 else np.nan,
        "avg_holding_bars": avg_hold,
        "avg_mfe": avg_mfe,
        "avg_mae": avg_mae,
        "max_drawdown": mdd,
        "calmar": cagr_value / abs(mdd) if np.isfinite(cagr_value) and np.isfinite(mdd) and mdd < 0 else np.nan,
        "PSR": psr,
        "DSR": dsr["dsr"],
        "DSR_sr": dsr["sr"],
        "DSR_sr_star": dsr["sr_star"],
        "DSR_sr_se": dsr["sr_se"],
    }


# ============================================================================
# PURGED CROSS-VALIDATION
# ============================================================================

def _time_bounds_from_events(events: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    return events["entry_time"].min(), events["exit_time"].max()


def purged_kfold_splits(
    events: pd.DataFrame,
    n_splits: int = PURGED_KFOLD_SPLITS,
    embargo_bars: int = EMBARGO_BARS,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Purged K-Fold over event intervals, with a post-test embargo."""
    n = len(events)
    if n < n_splits:
        return

    order = np.arange(n)
    folds = np.array_split(order, n_splits)

    time_index = pd.DatetimeIndex(events["entry_time"])
    exit_times = pd.DatetimeIndex(events["exit_time"])

    for test_idx in folds:
        if len(test_idx) == 0:
            continue

        test_start = time_index[test_idx].min()
        test_end = exit_times[test_idx].max()

        # Embargo is a time period immediately after the test interval.
        if len(time_index) > 1:
            median_step = pd.Series(time_index.sort_values()).diff().median()
        else:
            median_step = pd.Timedelta(minutes=15)
        if pd.isna(median_step):
            median_step = pd.Timedelta(minutes=15)
        embargo_end = test_end + embargo_bars * median_step

        overlap = (time_index <= test_end) & (exit_times >= test_start)
        embargo = (time_index > test_end) & (time_index <= embargo_end)
        train_mask = ~(overlap | embargo)
        train_mask[test_idx] = False

        train_idx = np.flatnonzero(train_mask)
        yield train_idx, np.asarray(test_idx, dtype=int)


def evaluate_purged_kfold(
    df: pd.DataFrame,
    params: StrategyParams,
    n_splits: int = PURGED_KFOLD_SPLITS,
) -> pd.DataFrame:
    """Evaluate a fixed strategy on purged folds by event intervals."""
    all_trades = trades_to_frame(simulate_strategy(df, params))
    if all_trades.empty:
        return pd.DataFrame()

    rows = []
    for fold, (train_idx, test_idx) in enumerate(
        purged_kfold_splits(all_trades, n_splits), start=1
    ):
        test_events = all_trades.iloc[test_idx]
        if test_events.empty:
            continue

        start = test_events["entry_time"].min()
        end = test_events["exit_time"].max()
        fold_trades = simulate_strategy(
            df, params, start_time=start, end_time=end, allow_overlap=ALLOW_OVERLAP
        )
        tdf = trades_to_frame(fold_trades)
        equity = build_equity_curve(df.loc[start:end], tdf)
        metrics = compute_metrics(df.loc[start:end], tdf, equity)
        rows.append({"fold": fold, **metrics})

    return pd.DataFrame(rows)


# ============================================================================
# WALK-FORWARD
# ============================================================================

def make_walk_forward_splits(
    n_bars: int,
    n_splits: int = WALK_FORWARD_SPLITS,
    min_train_ratio: float = WALK_FORWARD_MIN_TRAIN_RATIO,
) -> Iterator[Tuple[int, int, int]]:
    """Anchored expanding-window splits: train is strictly before test."""
    if not (0 < min_train_ratio < 1):
        raise ValueError("min_train_ratio must be between 0 and 1")

    min_train = max(100, int(n_bars * min_train_ratio))
    remaining = n_bars - min_train
    if remaining < n_splits:
        raise ValueError("Not enough observations for requested walk-forward splits")

    test_size = max(1, remaining // n_splits)
    for i in range(n_splits):
        train_end = min_train + i * test_size
        test_start = train_end
        test_end = min(n_bars, test_start + test_size)
        if test_end > test_start:
            yield 0, train_end, test_end


def strategy_grid() -> List[StrategyParams]:
    return [
        StrategyParams(z, pt, sl)
        for z, pt, sl in itertools.product(
            SEARCH_Z_THRESHOLDS,
            SEARCH_PT_MULTS,
            SEARCH_SL_MULTS,
        )
    ]


def _score_for_selection(metrics: Dict[str, float]) -> float:
    """Train-fold selection score: Sharpe with a penalty for deep drawdown."""
    sr = metrics.get("daily_sharpe", np.nan)
    mdd = metrics.get("max_drawdown", np.nan)
    if not np.isfinite(sr):
        return -np.inf
    penalty = 0.25 * abs(mdd) if np.isfinite(mdd) else 0.25
    return float(sr - penalty)


def evaluate_window(
    df: pd.DataFrame,
    params: StrategyParams,
    start: int,
    end: int,
) -> Dict[str, float]:
    start_time = df.index[start]
    end_time = df.index[end - 1]
    trades = trades_to_frame(
        simulate_strategy(
            df,
            params,
            start_time=start_time,
            end_time=end_time,
            allow_overlap=ALLOW_OVERLAP,
        )
    )
    window = df.iloc[start:end]
    equity = build_equity_curve(window, trades)
    return compute_metrics(window, trades, equity)


def walk_forward_search(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select parameters only from the past, then test on the next unseen block.

    This is the primary honest OOS test in this file.
    """
    grid = strategy_grid()
    rows = []
    splits = list(make_walk_forward_splits(len(df)))

    log.info("Walk-forward: %d splits x %d candidate strategies", len(splits), len(grid))

    for fold, (_, train_end, test_end) in enumerate(splits, start=1):
        best_params: Optional[StrategyParams] = None
        best_score = -np.inf
        best_train: Optional[Dict[str, float]] = None

        for params in grid:
            metrics = evaluate_window(df, params, 0, train_end)
            score = _score_for_selection(metrics)
            if score > best_score:
                best_score = score
                best_params = params
                best_train = metrics

        assert best_params is not None
        test_metrics = evaluate_window(df, best_params, train_end, test_end)

        rows.append({
            "fold": fold,
            "train_end": str(df.index[train_end - 1]),
            "test_start": str(df.index[train_end]),
            "test_end": str(df.index[test_end - 1]),
            "selected_z": best_params.z_threshold,
            "selected_pt": best_params.pt_mult,
            "selected_sl": best_params.sl_mult,
            "selection_score": best_score,
            "train_sharpe": best_train["daily_sharpe"],
            "train_mdd": best_train["max_drawdown"],
            "test_sharpe": test_metrics["daily_sharpe"],
            "test_return": test_metrics["net_total_return"],
            "test_mdd": test_metrics["max_drawdown"],
            "test_trades": test_metrics["n_trades"],
            "test_psr": test_metrics["PSR"],
            "test_dsr": test_metrics["DSR"],
        })

    return pd.DataFrame(rows)


# ============================================================================
# COMBINATORIAL PURGED CV / PBO-STYLE DIAGNOSTIC
# ============================================================================

def combinatorial_group_splits(
    n_events: int,
    n_groups: int,
    n_test_groups: int,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    if n_groups < 2 or n_test_groups < 1 or n_test_groups >= n_groups:
        raise ValueError("Invalid CPCV group parameters")

    groups = np.array_split(np.arange(n_events), n_groups)
    for test_group_ids in itertools.combinations(range(n_groups), n_test_groups):
        test_idx = np.concatenate([groups[g] for g in test_group_ids])
        train_idx = np.concatenate([
            groups[g] for g in range(n_groups) if g not in test_group_ids
        ])
        yield np.sort(train_idx), np.sort(test_idx)


def estimate_pbo_style(df: pd.DataFrame) -> Dict[str, float]:
    """
    A practical CSCV/CPCV-style selection diagnostic.

    For each combinatorial split:
      - evaluate all parameter configurations on the training partition;
      - select the best training Sharpe;
      - evaluate that selected configuration in the test partition;
      - record whether its test performance falls below the median of candidates.

    This is a PBO-style research diagnostic, not a claim of an exact theorem
    unless all assumptions of the underlying CSCV construction are satisfied.
    """
    grid = strategy_grid()
    events_reference = trades_to_frame(simulate_strategy(df, grid[0]))
    if events_reference.empty:
        return {"pbo_style": np.nan, "splits": 0, "trials": len(grid)}

    event_groups = list(
        combinatorial_group_splits(len(events_reference), CPCV_GROUPS, CPCV_TEST_GROUPS)
    )
    overfit_count = 0
    used = 0

    # Map event-time ranges; all strategies share the same signal timestamps only
    # approximately. We therefore use the reference event partition as a split
    # clock and run each candidate on the resulting time window.
    for train_event_idx, test_event_idx in event_groups:
        ref_train = events_reference.iloc[train_event_idx]
        ref_test = events_reference.iloc[test_event_idx]
        if ref_train.empty or ref_test.empty:
            continue

        train_start = ref_train["entry_time"].min()
        train_end = ref_train["exit_time"].max()
        test_start = ref_test["entry_time"].min()
        test_end = ref_test["exit_time"].max()

        train_scores = []
        test_scores = []
        for params in grid:
            train_metrics = evaluate_window_by_time(df, params, train_start, train_end)
            test_metrics = evaluate_window_by_time(df, params, test_start, test_end)
            train_scores.append(train_metrics.get("daily_sharpe", np.nan))
            test_scores.append(test_metrics.get("daily_sharpe", np.nan))

        train_scores_np = np.asarray(train_scores, dtype=float)
        test_scores_np = np.asarray(test_scores, dtype=float)
        valid = np.isfinite(train_scores_np) & np.isfinite(test_scores_np)
        if valid.sum() < 3:
            continue

        winner = np.flatnonzero(valid)[np.argmax(train_scores_np[valid])]
        median_test = float(np.median(test_scores_np[valid]))
        winner_test = float(test_scores_np[winner])

        overfit_count += int(winner_test < median_test)
        used += 1

    return {
        "pbo_style": float(overfit_count / used) if used else np.nan,
        "splits": used,
        "trials": len(grid),
    }


def evaluate_window_by_time(
    df: pd.DataFrame,
    params: StrategyParams,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> Dict[str, float]:
    trades = trades_to_frame(
        simulate_strategy(
            df,
            params,
            start_time=start_time,
            end_time=end_time,
            allow_overlap=ALLOW_OVERLAP,
        )
    )
    if trades.empty:
        return {"daily_sharpe": np.nan, "n_trades": 0}

    window = df.loc[start_time:end_time]
    equity = build_equity_curve(window, trades)
    return compute_metrics(window, trades, equity)


# ============================================================================
# BUY & HOLD BENCHMARK
# ============================================================================

def benchmark_metrics(df: pd.DataFrame) -> Dict[str, float]:
    bh = df["close"].iloc[-1] / df["close"].iloc[0] - 1.0
    daily_close = df["close"].resample("1D").last().dropna()
    daily_ret = daily_close.pct_change().dropna()
    sr = sharpe_ratio(daily_ret.to_numpy())
    return {
        "buy_hold_total_return": float(bh),
        "buy_hold_daily_sharpe": float(sr),
        "buy_hold_annualized_sharpe": float(sr * math.sqrt(365.0)) if np.isfinite(sr) else np.nan,
    }


# ============================================================================
# REPORTING
# ============================================================================

def log_metrics(title: str, metrics: Dict[str, float]) -> None:
    log.info("\n%s", "=" * 72)
    log.info("%s", title)
    log.info("%s", "=" * 72)
    for key, value in metrics.items():
        if isinstance(value, float):
            log.info("%-28s % .8f", key, value)
        else:
            log.info("%-28s %s", key, value)


def validate_configuration() -> None:
    if Z_PERIOD < 10:
        raise ValueError("Z_PERIOD too small for a meaningful mean-reversion estimate")
    if MAX_HOLDING < 2:
        raise ValueError("MAX_HOLDING must be >= 2")
    if N_TRIALS < 2:
        raise ValueError("N_TRIALS must be >= 2")
    if FEE_BPS_PER_SIDE < 0 or SLIPPAGE_BPS_PER_SIDE < 0:
        raise ValueError("Fees/slippage cannot be negative")
    if SAME_BAR_POLICY not in {"worst", "best"}:
        raise ValueError("SAME_BAR_POLICY must be 'worst' or 'best'")


def save_outputs(trades: pd.DataFrame, equity: pd.Series, wf: pd.DataFrame) -> None:
    if not SAVE_CSV:
        return
    if not trades.empty:
        trades.to_csv(TRADE_CSV, index=False)
    equity.rename("equity").to_csv(EQUITY_CSV, header=True)
    if not wf.empty:
        wf.to_csv(WF_CSV, index=False)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    validate_configuration()

    log.info("=" * 72)
    log.info("Z-Score Research Backtest PRO")
    log.info("%s | %s | bars=%d", SYMBOL, INTERVAL, TOTAL_BARS)
    log.info("=" * 72)

    prices = fetch_data(SYMBOL, INTERVAL, TOTAL_BARS)
    data = prepare_features(prices)

    log.info("Closed candles loaded: %d", len(data))
    log.info("Period: %s -> %s", data.index[0], data.index[-1])

    params = StrategyParams(Z_THRESHOLD, PT_MULT, SL_MULT)
    trades = trades_to_frame(simulate_strategy(data, params))
    equity = build_equity_curve(data, trades)
    metrics = compute_metrics(data, trades, equity)

    log_metrics("BASELINE STRATEGY", metrics)
    log_metrics("BUY & HOLD BENCHMARK", benchmark_metrics(data))

    # Fixed-parameter Purged K-Fold diagnostics.
    if not trades.empty:
        pkf = evaluate_purged_kfold(data, params)
        if not pkf.empty:
            log_metrics(
                "PURGED K-FOLD MEAN",
                pkf.mean(numeric_only=True).to_dict(),
            )

    # Walk-forward is the principal model-selection/OOS test.
    wf = walk_forward_search(data)
    if not wf.empty:
        log.info("\nWalk-forward OOS results:")
        log.info("%s", wf.to_string(index=False))

    if RUN_CPCV_PBO:
        pbo = estimate_pbo_style(data)
        log_metrics("CPCV / PBO-STYLE SEARCH DIAGNOSTIC", pbo)
    else:
        log.info("CPCV/PBO diagnostic disabled (RUN_CPCV_PBO=False)")

    save_outputs(trades, equity, wf)

    log.info("\nResearch interpretation:")
    log.info(
        "Do NOT treat DSR/PSR as 'win probability'. They are inference measures "
        "and depend on sample size, return distribution, and search count."
    )
    log.info(
        "N_TRIALS=%d must reflect the actual number of materially different "
        "research attempts, not a convenient round number.", N_TRIALS
    )
    log.info(
        "For deployment decisions, prioritize unseen walk-forward performance, "
        "cost sensitivity, drawdown behaviour, and stability across regimes."
    )


if __name__ == "__main__":
    main()
