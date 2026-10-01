"""
triple_barrier.py
=================

Production-oriented Triple-Barrier Labeling.

Based on:
- Marcos López de Prado
  Advances in Financial Machine Learning, Chapter 3
- mlfinpy / mlfinlab implementations of:
  - Triple Barrier Method
  - Vertical Barriers
  - Meta-Labeling
- Practical extensions:
  - OHLC-aware barrier detection
  - Forward-safe volatility targets
  - Side-aware labeling
  - Gap handling
  - Ambiguous intrabar barrier handling
  - Log returns
  - Event metadata
  - Rare-label filtering

Main labels:

    +1 = Profit-Taking barrier hit first
    -1 = Stop-Loss barrier hit first
     0 = Vertical/time barrier reached first

Important:
The default behavior follows the standard Triple-Barrier
interpretation where the vertical barrier produces label 0.

For a primary model side prediction, the output also includes:

    meta_label = 1  -> trade was favorable
    meta_label = 0  -> trade was not favorable

This file is a LABELING module.
It is not itself a trading strategy or execution engine.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd


log = logging.getLogger("triple_barrier")


# ============================================================
# TYPES
# ============================================================

PriceInput = Union[pd.Series, pd.DataFrame]


# ============================================================
# VALIDATION / DATA PREPARATION
# ============================================================

def _prepare_market_data(
    prices: PriceInput,
) -> tuple[pd.DataFrame, bool]:
    """
    Normalize input price data.

    Accepted:

    1) pd.Series
       -> interpreted as Close

    2) pd.DataFrame
       -> expected at minimum:
          close

       Optional:
          open
          high
          low

    Returns
    -------
    market : pd.DataFrame
    has_ohlc : bool
    """

    if isinstance(prices, pd.Series):
        close = prices.copy()

        if not isinstance(close.index, pd.DatetimeIndex):
            raise TypeError(
                "prices.index must be a pandas DatetimeIndex."
            )

        close = close.sort_index()

        if close.index.has_duplicates:
            raise ValueError(
                "prices.index contains duplicate timestamps."
            )

        close = pd.to_numeric(close, errors="coerce")

        if close.isna().any():
            raise ValueError(
                "prices contains NaN values."
            )

        if (close <= 0).any():
            raise ValueError(
                "Close prices must be positive."
            )

        return close.rename("close").to_frame(), False

    if not isinstance(prices, pd.DataFrame):
        raise TypeError(
            "prices must be pd.Series or pd.DataFrame."
        )

    df = prices.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "prices.index must be a pandas DatetimeIndex."
        )

    df = df.sort_index()

    if df.index.has_duplicates:
        raise ValueError(
            "prices.index contains duplicate timestamps."
        )

    # Normalize column names
    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]

    if "close" not in df.columns:
        raise ValueError(
            "DataFrame must contain a 'close' column."
        )

    # Convert numeric
    for column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if df["close"].isna().any():
        raise ValueError(
            "close contains NaN values."
        )

    if (df["close"] <= 0).any():
        raise ValueError(
            "Close prices must be positive."
        )

    has_ohlc = (
        "high" in df.columns
        and "low" in df.columns
    )

    if has_ohlc:

        if df[["high", "low"]].isna().any().any():
            raise ValueError(
                "high/low contains NaN values."
            )

        if (df["high"] < df["low"]).any():
            raise ValueError(
                "Found rows where high < low."
            )

        if (df["high"] <= 0).any():
            raise ValueError(
                "High prices must be positive."
            )

        if (df["low"] <= 0).any():
            raise ValueError(
                "Low prices must be positive."
            )

    if "open" in df.columns:

        if df["open"].isna().any():
            raise ValueError(
                "open contains NaN values."
            )

        if (df["open"] <= 0).any():
            raise ValueError(
                "Open prices must be positive."
            )

    return df, has_ohlc


# ============================================================
# VOLATILITY TARGET
# ============================================================

def estimate_volatility(
    prices: PriceInput,
    method: str = "ewm",
    window: int = 50,
    min_periods: int = 20,
) -> pd.Series:
    """
    Estimate forward-safe volatility target.

    Parameters
    ----------
    prices:
        Close Series or OHLC DataFrame.

    method:
        "ewm"
            Exponentially weighted standard deviation
            of log returns.

        "rolling"
            Rolling standard deviation of log returns.

        "garman_klass"
            OHLC volatility estimator.

    window:
        Lookback period.

    min_periods:
        Minimum observations required.

    Returns
    -------
    pd.Series
        Volatility target aligned with prices.index.
    """

    market, has_ohlc = _prepare_market_data(prices)

    if window < 2:
        raise ValueError(
            "window must be >= 2."
        )

    if min_periods < 2:
        raise ValueError(
            "min_periods must be >= 2."
        )

    close = market["close"]

    log_returns = np.log(close).diff()

    method = method.lower().strip()

    # --------------------------------------------------------
    # EWM volatility
    # --------------------------------------------------------

    if method == "ewm":

        volatility = (
            log_returns
            .ewm(
                span=window,
                min_periods=min_periods,
                adjust=False,
            )
            .std(bias=False)
        )

    # --------------------------------------------------------
    # Rolling volatility
    # --------------------------------------------------------

    elif method == "rolling":

        volatility = (
            log_returns
            .rolling(
                window=window,
                min_periods=min_periods,
            )
            .std(ddof=1)
        )

    # --------------------------------------------------------
    # Garman-Klass
    # --------------------------------------------------------

    elif method == "garman_klass":

        if not has_ohlc or "open" not in market.columns:
            raise ValueError(
                "Garman-Klass requires open/high/low/close."
            )

        open_price = market["open"]
        high = market["high"]
        low = market["low"]
        close_price = market["close"]

        # Garman-Klass variance estimator
        rs = (
            0.5 * np.log(high / low) ** 2
            - (2.0 * np.log(2.0) - 1.0)
            * np.log(close_price / open_price) ** 2
        )

        # Numerical protection
        rs = rs.clip(lower=0.0)

        volatility = np.sqrt(
            rs.rolling(
                window=window,
                min_periods=min_periods,
            ).mean()
        )

    else:
        raise ValueError(
            "method must be one of: "
            "'ewm', 'rolling', 'garman_klass'"
        )

    return volatility.rename("trgt")


# ============================================================
# VERTICAL BARRIER
# ============================================================

def add_vertical_barrier(
    events,
    prices: PriceInput,
    max_holding: int = 10,
) -> pd.Series:
    """
    Create vertical/time barriers.

    max_holding is measured in bars.

    Example:

        t0 = bar 100
        max_holding = 10

        vertical barrier = bar 110
    """

    market, _ = _prepare_market_data(prices)

    if max_holding < 1:
        raise ValueError(
            "max_holding must be >= 1."
        )

    event_index = pd.DatetimeIndex(
        events
    ).sort_values().unique()

    positions = market.index.get_indexer(
        event_index
    )

    result = pd.Series(
        pd.NaT,
        index=event_index,
        dtype="datetime64[ns]",
        name="t1",
    )

    for timestamp, position in zip(
        event_index,
        positions,
    ):

        if position < 0:
            continue

        end_position = position + max_holding

        if end_position < len(market):
            result.loc[timestamp] = (
                market.index[end_position]
            )

    return result


# ============================================================
# BARRIER HIT DETECTION
# ============================================================

def _detect_barrier_hit(
    row: pd.Series,
    side: float,
    pt_price: float,
    sl_price: float,
    has_ohlc: bool,
) -> tuple[bool, bool]:
    """
    Detect whether PT and/or SL was touched by one bar.

    Returns
    -------
    pt_hit, sl_hit
    """

    close_price = float(row["close"])

    open_price = np.nan

    if "open" in row.index:
        if pd.notna(row["open"]):
            open_price = float(row["open"])

    if has_ohlc:

        high = float(row["high"])
        low = float(row["low"])

        if side == 1.0:

            pt_hit = (
                (
                    np.isfinite(open_price)
                    and open_price >= pt_price
                )
                or high >= pt_price
            )

            sl_hit = (
                (
                    np.isfinite(open_price)
                    and open_price <= sl_price
                )
                or low <= sl_price
            )

        else:

            # Short:
            # Profit = price falls
            # Stop   = price rises

            pt_hit = (
                (
                    np.isfinite(open_price)
                    and open_price <= pt_price
                )
                or low <= pt_price
            )

            sl_hit = (
                (
                    np.isfinite(open_price)
                    and open_price >= sl_price
                )
                or high >= sl_price
            )

    else:

        # Close-only fallback
        if side == 1.0:

            pt_hit = close_price >= pt_price
            sl_hit = close_price <= sl_price

        else:

            pt_hit = close_price <= pt_price
            sl_hit = close_price >= sl_price

    return bool(pt_hit), bool(sl_hit)


# ============================================================
# MAIN TRIPLE-BARRIER ENGINE
# ============================================================

def triple_barrier_labeling(
    prices: PriceInput,
    events,
    pt_sl=(2.0, 1.0),
    max_holding: int = 10,
    volatility: Optional[pd.Series] = None,
    volatility_method: str = "ewm",
    volatility_window: int = 50,
    volatility_min_periods: int = 20,
    min_ret: float = 0.0,
    side: Optional[pd.Series] = None,
    time_barrier_label: str = "zero",
    tie_break: str = "stop",
    use_open_for_gap: bool = True,
) -> pd.DataFrame:
    """
    Triple-Barrier Labeling.

    Labels
    ------

    +1:
        Take-profit barrier touched first.

    -1:
        Stop-loss barrier touched first.

     0:
        Vertical/time barrier reached first.

    Parameters
    ----------

    prices:
        Either:

        pd.Series
            close prices

        OR

        pd.DataFrame
            open/high/low/close

    events:
        DatetimeIndex of event timestamps.

    pt_sl:
        Tuple:

            (profit_target_multiplier,
             stop_loss_multiplier)

        Barrier widths are:

            PT = volatility * pt_multiplier
            SL = volatility * sl_multiplier

    max_holding:
        Maximum holding period in bars.

    volatility:
        Optional externally calculated volatility.

        This can be:
        - EWMA volatility
        - ATR-derived target
        - GARCH volatility
        - any other forward-safe target

    volatility_method:
        "ewm"
        "rolling"
        "garman_klass"

    min_ret:
        Minimum volatility target required.
        Events below this threshold are discarded.

    side:
        Optional primary-model side:

            +1 = long
            -1 = short

        When supplied, the algorithm creates:

            meta_label = 1
                profitable / favorable event

            meta_label = 0
                unfavorable event

    time_barrier_label:
        "zero"
            Standard Triple-Barrier behavior.

        "directional"
            On vertical expiry:
                +1 if final price > entry
                -1 if final price < entry
                 0 otherwise

        Recommended for ML labeling:
            "zero"

    tie_break:
        If both PT and SL are touched within the same OHLC bar:

        "stop"
            Conservative default.

        "profit"
            Optimistic.

        "close"
            Decide from that bar's close.

        "drop"
            Mark label as NaN.

    use_open_for_gap:
        If the next bar opens beyond a barrier,
        use the observed open as the exit price
        instead of pretending an exact fill at the barrier.

    Returns
    -------

    pd.DataFrame

    Columns
    -------

    t1
        Actual first-touch / exit timestamp.

    vertical_barrier
        Maximum holding timestamp.

    trgt
        Volatility target.

    side
        Position side.

    pt
        PT multiplier.

    sl
        SL multiplier.

    pt_price
        Absolute PT level.

    sl_price
        Absolute SL level.

    exit_price
        Estimated exit price.

    ret
        Simple realized return.

    log_ret
        Log return.

    label
        Triple-Barrier label.

    meta_label
        1 if PT was reached.
        0 otherwise.

    touch
        "pt", "sl", "vertical", "ambiguous".

    duration_bars
        Number of elapsed bars.

    ambiguous
        Whether PT and SL were both touched
        within the same bar.
    """

    market, has_ohlc = _prepare_market_data(
        prices
    )

    if not (
        isinstance(pt_sl, (tuple, list))
        and len(pt_sl) == 2
    ):
        raise ValueError(
            "pt_sl must be a 2-element tuple/list."
        )

    pt_multiplier = float(pt_sl[0])
    sl_multiplier = float(pt_sl[1])

    if (
        pt_multiplier < 0
        or sl_multiplier < 0
    ):
        raise ValueError(
            "pt/sl multipliers cannot be negative."
        )

    if (
        pt_multiplier == 0
        and sl_multiplier == 0
    ):
        raise ValueError(
            "At least one barrier must be enabled."
        )

    if max_holding < 1:
        raise ValueError(
            "max_holding must be >= 1."
        )

    if min_ret < 0:
        raise ValueError(
            "min_ret cannot be negative."
        )

    if time_barrier_label not in {
        "zero",
        "directional",
    }:
        raise ValueError(
            "time_barrier_label must be "
            "'zero' or 'directional'."
        )

    if tie_break not in {
        "stop",
        "profit",
        "close",
        "drop",
    }:
        raise ValueError(
            "tie_break must be "
            "'stop', 'profit', 'close', or 'drop'."
        )

    close = market["close"]

    # --------------------------------------------------------
    # Volatility target
    # --------------------------------------------------------

    if volatility is None:

        target = estimate_volatility(
            market,
            method=volatility_method,
            window=volatility_window,
            min_periods=volatility_min_periods,
        )

    else:

        if not isinstance(
            volatility,
            pd.Series,
        ):
            raise TypeError(
                "volatility must be a pandas Series."
            )

        target = volatility.astype(float).reindex(
            market.index
        )

    # --------------------------------------------------------
    # Events
    # --------------------------------------------------------

    event_index = pd.DatetimeIndex(
        events
    ).sort_values().unique()

    positions = market.index.get_indexer(
        event_index
    )

    records = []

    # --------------------------------------------------------
    # Optional side
    # --------------------------------------------------------

    if side is not None:

        if not isinstance(
            side,
            pd.Series,
        ):
            raise TypeError(
                "side must be a pandas Series."
            )

        side_aligned = side.astype(float).reindex(
            market.index
        )

    else:

        side_aligned = None

    # ========================================================
    # EVENT LOOP
    # ========================================================

    for t0, position in zip(
        event_index,
        positions,
    ):

        if position < 0:
            continue

        # Need a complete future horizon.
        vertical_position = (
            position + max_holding
        )

        if vertical_position >= len(market):
            continue

        # ----------------------------------------------------
        # Volatility target
        # ----------------------------------------------------

        trgt = target.loc[t0]

        if (
            pd.isna(trgt)
            or not np.isfinite(trgt)
            or trgt <= 0
            or trgt < min_ret
        ):
            continue

        trgt = float(trgt)

        # ----------------------------------------------------
        # Entry
        # ----------------------------------------------------

        entry_price = float(
            close.loc[t0]
        )

        # ----------------------------------------------------
        # Position side
        # ----------------------------------------------------

        if side_aligned is None:

            event_side = 1.0

        else:

            raw_side = side_aligned.loc[t0]

            if (
                pd.isna(raw_side)
                or raw_side not in (-1.0, 1.0)
            ):
                continue

            event_side = float(raw_side)

        # ----------------------------------------------------
        # Barrier prices
        # ----------------------------------------------------

        if event_side == 1.0:

            # LONG

            pt_price = (
                entry_price
                * (
                    1.0
                    + pt_multiplier * trgt
                )
                if pt_multiplier > 0
                else np.nan
            )

            sl_price = (
                entry_price
                * (
                    1.0
                    - sl_multiplier * trgt
                )
                if sl_multiplier > 0
                else np.nan
            )

        else:

            # SHORT

            pt_price = (
                entry_price
                * (
                    1.0
                    - pt_multiplier * trgt
                )
                if pt_multiplier > 0
                else np.nan
            )

            sl_price = (
                entry_price
                * (
                    1.0
                    + sl_multiplier * trgt
                )
                if sl_multiplier > 0
                else np.nan
            )

        # ----------------------------------------------------
        # Default outcome
        # ----------------------------------------------------

        exit_position = vertical_position

        exit_type = "vertical"

        label = 0

        ambiguous = False

        exit_price = float(
            close.iloc[vertical_position]
        )

        # ====================================================
        # CRITICAL:
        #
        # Start from position + 1.
        #
        # The event is generated at t0 close, therefore
        # the high/low of t0 happened before entry.
        # ====================================================

        for current_position in range(
            position + 1,
            vertical_position + 1,
        ):

            row = market.iloc[
                current_position
            ]

            pt_enabled = (
                pt_multiplier > 0
            )

            sl_enabled = (
                sl_multiplier > 0
            )

            pt_hit = False
            sl_hit = False

            # ----------------------------------------------
            # Detect barrier touches
            # ----------------------------------------------

            if pt_enabled or sl_enabled:

                pt_hit, sl_hit = (
                    _detect_barrier_hit(
                        row=row,
                        side=event_side,
                        pt_price=pt_price,
                        sl_price=sl_price,
                        has_ohlc=has_ohlc,
                    )
                )

            # ----------------------------------------------
            # No horizontal barrier hit
            # ----------------------------------------------

            if not pt_hit and not sl_hit:
                continue

            # ----------------------------------------------
            # Both touched in same candle
            # ----------------------------------------------

            if pt_hit and sl_hit:

                ambiguous = True

                if tie_break == "drop":

                    exit_type = "ambiguous"

                    label = np.nan

                    exit_position = (
                        current_position
                    )

                    exit_price = float(
                        row["close"]
                    )

                    break

                if tie_break == "profit":

                    chosen_barrier = "pt"

                elif tie_break == "close":

                    candle_close = float(
                        row["close"]
                    )

                    signed_return = (
                        event_side
                        * (
                            candle_close
                            / entry_price
                            - 1.0
                        )
                    )

                    chosen_barrier = (
                        "pt"
                        if signed_return >= 0
                        else "sl"
                    )

                else:

                    # Conservative default
                    chosen_barrier = "sl"

            elif pt_hit:

                chosen_barrier = "pt"

            else:

                chosen_barrier = "sl"

            # ----------------------------------------------
            # Exit information
            # ----------------------------------------------

            exit_type = chosen_barrier

            exit_position = (
                current_position
            )

            if chosen_barrier == "pt":

                label = 1

                barrier_price = pt_price

                # Gap-through PT
                if (
                    use_open_for_gap
                    and "open" in row.index
                    and pd.notna(row["open"])
                ):

                    open_price = float(
                        row["open"]
                    )

                    if event_side == 1.0:

                        gap_crossed = (
                            open_price >= pt_price
                        )

                    else:

                        gap_crossed = (
                            open_price <= pt_price
                        )

                    exit_price = (
                        open_price
                        if gap_crossed
                        else barrier_price
                    )

                else:

                    exit_price = barrier_price

            else:

                label = -1

                barrier_price = sl_price

                # Gap-through SL
                if (
                    use_open_for_gap
                    and "open" in row.index
                    and pd.notna(row["open"])
                ):

                    open_price = float(
                        row["open"]
                    )

                    if event_side == 1.0:

                        gap_crossed = (
                            open_price <= sl_price
                        )

                    else:

                        gap_crossed = (
                            open_price >= sl_price
                        )

                    exit_price = (
                        open_price
                        if gap_crossed
                        else barrier_price
                    )

                else:

                    exit_price = barrier_price

            break

        # ====================================================
        # VERTICAL BARRIER
        # ====================================================

        if exit_type == "vertical":

            final_price = float(
                close.iloc[vertical_position]
            )

            exit_price = final_price

            if (
                time_barrier_label
                == "directional"
            ):

                final_return = (
                    final_price
                    / entry_price
                    - 1.0
                )

                if final_return > 0:

                    label = 1

                elif final_return < 0:

                    label = -1

                else:

                    label = 0

            else:

                # Standard Triple Barrier
                label = 0

        # ====================================================
        # RETURNS
        # ====================================================

        if exit_price <= 0:

            continue

        simple_return = (
            exit_price
            / entry_price
            - 1.0
        )

        log_return = float(
            np.log(
                exit_price
                / entry_price
            )
        )

        duration = (
            exit_position - position
        )

        records.append(
            {
                "t0": t0,
                "t1": market.index[
                    exit_position
                ],
                "vertical_barrier": market.index[
                    vertical_position
                ],
                "trgt": trgt,
                "side": event_side,
                "pt": pt_multiplier,
                "sl": sl_multiplier,
                "pt_price": pt_price,
                "sl_price": sl_price,
                "exit_price": exit_price,
                "ret": simple_return,
                "log_ret": log_return,
                "label": label,
                "touch": exit_type,
                "duration_bars": duration,
                "ambiguous": ambiguous,
            }
        )

    # ========================================================
    # EMPTY RESULT
    # ========================================================

    if not records:

        columns = [
            "t1",
            "vertical_barrier",
            "trgt",
            "side",
            "pt",
            "sl",
            "pt_price",
            "sl_price",
            "exit_price",
            "ret",
            "log_ret",
            "label",
            "touch",
            "duration_bars",
            "ambiguous",
            "meta_label",
        ]

        return pd.DataFrame(
            columns=columns,
            index=pd.DatetimeIndex(
                [],
                name="t0",
            ),
        )

    # ========================================================
    # FINAL DATAFRAME
    # ========================================================

    result = (
        pd.DataFrame(records)
        .set_index("t0")
        .sort_index()
    )

    result.index.name = "t0"

    # If the label is not NaN, use integer dtype.
    if not result["label"].isna().any():

        result["label"] = (
            result["label"]
            .astype(int)
        )

    # ========================================================
    # META LABEL
    #
    # Important:
    # Meta-label answers:
    #
    # "Was the primary side prediction correct?"
    #
    # ========================================================

    result["meta_label"] = (
        result["touch"] == "pt"
    ).astype(int)

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    log.info(
        "[TB] Labeled %d events",
        len(result),
    )

    log.info(
        "[TB] Label distribution: %s",
        result["label"]
        .value_counts(dropna=False)
        .to_dict(),
    )

    log.info(
        "[TB] Barrier distribution: %s",
        result["touch"]
        .value_counts(dropna=False)
        .to_dict(),
    )

    return result


# ============================================================
# GET BINS
# ============================================================

def get_bins(
    triple_barrier_events: pd.DataFrame,
    prices: PriceInput,
) -> pd.DataFrame:
    """
    Convert Triple-Barrier events into ML-ready outcome bins.

    Produces:

        ret
        log_ret
        label

    If 'side' exists:

        side
        meta_ret

    If the main Triple-Barrier engine has already produced
    'exit_price', that price is preferred.

    Otherwise the close price at t1 is used.
    """

    if not isinstance(
        triple_barrier_events,
        pd.DataFrame,
    ):
        raise TypeError(
            "triple_barrier_events must be "
            "a pandas DataFrame."
        )

    if "t1" not in triple_barrier_events.columns:
        raise ValueError(
            "triple_barrier_events must contain 't1'."
        )

    market, _ = _prepare_market_data(
        prices
    )

    rows = []

    events = (
        triple_barrier_events
        .dropna(subset=["t1"])
    )

    for t0, event in events.iterrows():

        if t0 not in market.index:
            continue

        entry_price = float(
            market.loc[t0, "close"]
        )

        # ----------------------------------------------------
        # Prefer actual estimated exit_price
        # ----------------------------------------------------

        if (
            "exit_price" in event.index
            and pd.notna(event["exit_price"])
        ):

            exit_price = float(
                event["exit_price"]
            )

        else:

            t1 = pd.Timestamp(
                event["t1"]
            )

            position = market.index.searchsorted(
                t1,
                side="left",
            )

            if position >= len(market):
                continue

            exit_price = float(
                market.iloc[
                    position
                ]["close"]
            )

        if (
            entry_price <= 0
            or exit_price <= 0
        ):
            continue

        log_return = float(
            np.log(
                exit_price
                / entry_price
            )
        )

        simple_return = float(
            np.expm1(log_return)
        )

        row = {
            "ret": simple_return,
            "log_ret": log_return,
            "label": event.get(
                "label",
                np.nan,
            ),
        }

        # ----------------------------------------------------
        # Meta-labeling
        # ----------------------------------------------------

        if "side" in event.index:

            event_side = float(
                event["side"]
            )

            row["side"] = event_side

            row["meta_ret"] = (
                log_return
                * event_side
            )

            row["meta_label"] = int(
                row["meta_ret"] > 0
            )

        rows.append(
            (t0, row)
        )

    if not rows:

        columns = [
            "ret",
            "log_ret",
            "label",
        ]

        if "side" in triple_barrier_events.columns:

            columns += [
                "side",
                "meta_ret",
                "meta_label",
            ]

        return pd.DataFrame(
            columns=columns,
            index=pd.DatetimeIndex(
                [],
                name=triple_barrier_events.index.name,
            ),
        )

    result = pd.DataFrame(
        [row for _, row in rows],
        index=[index for index, _ in rows],
    )

    result.index.name = (
        triple_barrier_events.index.name
    )

    return result


# ============================================================
# DROP RARE LABELS
# ============================================================

def drop_rare_labels(
    events: pd.DataFrame,
    min_pct: float = 0.05,
    label_col: str = "label",
) -> pd.DataFrame:
    """
    Recursively remove very rare labels.

    Example:

        +1 = 48%
         0 = 50%
        -1 = 2%

    with min_pct=0.05

    the -1 class is removed.

    This follows the practical idea used in AFML/MLFinPy.

    IMPORTANT:
    Do not use this blindly on small datasets.
    """

    if label_col not in events.columns:
        raise ValueError(
            f"Missing column: {label_col}"
        )

    if not (
        0.0
        < min_pct
        < 1.0
    ):
        raise ValueError(
            "min_pct must be between 0 and 1."
        )

    result = events.copy()

    while len(result) > 0:

        distribution = (
            result[label_col]
            .value_counts(
                normalize=True,
                dropna=True,
            )
        )

        if distribution.empty:
            break

        rare_labels = distribution[
            distribution < min_pct
        ]

        if rare_labels.empty:
            break

        # Remove the rarest class first.
        rarest_label = (
            rare_labels
            .sort_values()
            .index[0]
        )

        result = result[
            result[label_col]
            != rarest_label
        ]

    return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - "
            "%(levelname)s - "
            "%(message)s"
        ),
    )

    print("\n" + "=" * 70)
    print("TRIPLE-BARRIER TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Reproducible synthetic OHLC data
    # --------------------------------------------------------

    rng = np.random.default_rng(42)

    n = 500

    index = pd.date_range(
        "2024-01-01",
        periods=n,
        freq="h",
    )

    close = (
        100.0
        * np.exp(
            np.cumsum(
                rng.normal(
                    0.0,
                    0.01,
                    n,
                )
            )
        )
    )

    open_price = np.r_[
        close[0],
        close[:-1],
    ]

    high = (
        np.maximum(
            open_price,
            close,
        )
        * (
            1.0
            + rng.uniform(
                0.0,
                0.003,
                n,
            )
        )
    )

    low = (
        np.minimum(
            open_price,
            close,
        )
        * (
            1.0
            - rng.uniform(
                0.0,
                0.003,
                n,
            )
        )
    )

    prices = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
        },
        index=index,
    )

    # --------------------------------------------------------
    # Event timestamps
    # --------------------------------------------------------

    events = index[
        50:-20:10
    ]

    # --------------------------------------------------------
    # Triple Barrier
    # --------------------------------------------------------

    result = triple_barrier_labeling(
        prices=prices,
        events=events,
        pt_sl=(2.0, 1.0),
        max_holding=20,
        volatility_method="ewm",
        volatility_window=50,
        volatility_min_periods=20,
        min_ret=0.0,
        tie_break="stop",
        time_barrier_label="zero",
    )

    print("\nFirst events:")
    print(
        result[
            [
                "t1",
                "vertical_barrier",
                "trgt",
                "pt_price",
                "sl_price",
                "exit_price",
                "ret",
                "label",
                "touch",
                "duration_bars",
            ]
        ].head(10)
    )

    print("\nLabel distribution:")
    print(
        result["label"]
        .value_counts()
        .sort_index()
    )

    print("\nBarrier distribution:")
    print(
        result["touch"]
        .value_counts()
    )

    # --------------------------------------------------------
    # Test side-aware labeling
    # --------------------------------------------------------

    sides = pd.Series(
        np.where(
            np.arange(n) % 2 == 0,
            1.0,
            -1.0,
        ),
        index=index,
    )

    side_result = triple_barrier_labeling(
        prices=prices,
        events=events,
        pt_sl=(2.0, 1.0),
        max_holding=20,
        volatility_method="ewm",
        volatility_window=50,
        volatility_min_periods=20,
        side=sides,
        tie_break="stop",
    )

    print("\nMeta-label distribution:")
    print(
        side_result["meta_label"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # get_bins test
    # --------------------------------------------------------

    bins = get_bins(
        side_result,
        prices,
    )

    print("\nBins:")
    print(
        bins.head(10)
    )

    # --------------------------------------------------------
    # Rare-label filtering
    # --------------------------------------------------------

    filtered = drop_rare_labels(
        result,
        min_pct=0.05,
    )

    print("\nFiltered distribution:")
    print(
        filtered["label"]
        .value_counts()
        .sort_index()
    )

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
