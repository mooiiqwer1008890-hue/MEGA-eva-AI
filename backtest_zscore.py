"""
backtest_zscore.py
===================
Backtests the Z-score mean-reversion signal used in quant_alert.py
against real historical BTC/USDT data, to answer one question
honestly: does |Z-Score| > 2 actually predict anything, or is it
just an arbitrary number?

Run this ONCE (it is not a long-running service, unlike quant_alert.py)
via Railway's Console tab: `python backtest_zscore.py`
It needs network access to fetch historical candles from Binance,
which is why it must run on Railway (or anywhere with real internet),
not in a restricted sandbox.

Design principles (same as our earlier feature-engineering work):
- NO LOOKAHEAD: the Z-score at candle t uses only candles up to t.
  The "forward return" used to evaluate it uses candles AFTER t,
  and is clearly separated as the evaluation target, never mixed
  into the signal itself.
- NO SINGLE-NUMBER VERDICT: a Sharpe ratio or win-rate on its own is
  easy to overstate. This script reports sample size (how many times
  the signal even fired), the split between the first and second
  half of the period (a crude but honest robustness check), and a
  plain-language significance note - not just "profitable: yes/no".
- THIS TESTS THE THRESHOLD AS-IS. It does not search for a "better"
  threshold. Optimizing the threshold on this same data would be
  a form of overfitting; if this test is unfavorable, the honest
  next step is a fresh hypothesis, not tuning this one until it
  passes.
"""

import time
import requests
import numpy as np

SYMBOL = "BTCUSDT"
INTERVAL = "15m"
Z_PERIOD = 20            # same rolling window as the live bot
FORWARD_HORIZONS = [5, 10, 20]   # candles ahead = 75min, 150min, 300min
Z_THRESHOLD = 2.0         # the exact threshold used in quant_alert.py
DAYS_OF_HISTORY = 180     # ~6 months of 15m candles


def fetch_historical_candles(symbol, interval, days):
    """
    Binance limits klines to 1000 per request, so we page backwards
    in time until we have `days` worth of candles. Returns columns:
    [open_time, open, high, low, close] oldest-first.
    """
    url = "https://data-api.binance.vision/api/v3/klines"
    candles_per_day = 96  # 24h * 4 (15-min candles)
    total_needed = days * candles_per_day

    all_rows = []
    end_time = None
    fetched = 0

    while fetched < total_needed:
        params = {"symbol": symbol, "interval": interval, "limit": 1000}
        if end_time is not None:
            params["endTime"] = end_time
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            break
        all_rows = rows + all_rows  # prepend (older data)
        end_time = rows[0][0] - 1   # next page ends right before this batch
        fetched += len(rows)
        time.sleep(0.3)  # be polite to the public endpoint

    data = np.array(
        [[float(r[1]), float(r[2]), float(r[3]), float(r[4])] for r in all_rows]
    )
    return data  # columns: open, high, low, close


def rolling_zscore(closes, period):
    n = len(closes)
    z = np.full(n, np.nan)
    for i in range(period, n):
        window = closes[i - period:i]
        mean, std = window.mean(), window.std()
        if std > 0:
            z[i] = (closes[i] - mean) / std
    return z


def evaluate(z, closes, horizon, threshold, label, split_start=0, split_end=None):
    """
    For the slice [split_start:split_end], finds every point where the
    Z-score crossed the threshold and measures the ACTUAL forward
    return `horizon` candles later. Compares that to the unconditional
    (baseline) forward return over the same slice.
    """
    end = split_end if split_end is not None else len(closes) - horizon
    idx_slice = np.arange(split_start, end)

    fwd_ret = np.log(closes[idx_slice + horizon] / closes[idx_slice])
    z_slice = z[idx_slice]

    buy_mask = z_slice < -threshold
    sell_mask = z_slice > threshold

    baseline_mean = np.nanmean(fwd_ret)
    baseline_winrate = np.mean(fwd_ret > 0)

    def summarize(mask, name):
        n = mask.sum()
        if n == 0:
            print(f"    {name}: 0 occurrences in this period — no signal, no evaluation possible.")
            return
        rets = fwd_ret[mask]
        mean_ret = rets.mean()
        winrate = np.mean(rets > 0)
        # crude one-sample z-test vs baseline mean (not vs zero) —
        # tells us whether the signal's average return differs from
        # what you'd get by chance in this same period.
        se = rets.std() / np.sqrt(n) if n > 1 else np.nan
        z_stat = (mean_ret - baseline_mean) / se if se and se > 0 else np.nan
        print(f"    {name}: n={n} | mean fwd return={mean_ret*100:.3f}% "
              f"(baseline {baseline_mean*100:.3f}%) | win rate={winrate*100:.1f}% "
              f"(baseline {baseline_winrate*100:.1f}%) | z-stat vs baseline={z_stat:.2f}")

    print(f"  [{label}] horizon={horizon} candles, threshold=|Z|>{threshold}")
    summarize(buy_mask, "BUY signal (Z < -2)")
    summarize(sell_mask, "SELL signal (Z > 2)")


def main():
    print(f"Fetching ~{DAYS_OF_HISTORY} days of {INTERVAL} candles for {SYMBOL}...")
    candles = fetch_historical_candles(SYMBOL, INTERVAL, DAYS_OF_HISTORY)
    closes = candles[:, 3]
    print(f"Got {len(closes)} candles.\n")

    z = rolling_zscore(closes, Z_PERIOD)
    mid = len(closes) // 2

    for horizon in FORWARD_HORIZONS:
        print("=" * 70)
        evaluate(z, closes, horizon, Z_THRESHOLD, "FULL PERIOD", 0, len(closes) - horizon)
        evaluate(z, closes, horizon, Z_THRESHOLD, "FIRST HALF (older)", 0, mid)
        evaluate(z, closes, horizon, Z_THRESHOLD, "SECOND HALF (newer)", mid, len(closes) - horizon)
        print()

    print("=" * 70)
    print("HOW TO READ THIS:")
    print("- 'n=' is the sample size. If n is small (under ~30), don't")
    print("  trust the percentages no matter how good they look.")
    print("- Compare each signal's mean/win-rate to the BASELINE in the")
    print("  same period, not to zero — the baseline already captures")
    print("  BTC's general trend over that window.")
    print("- If FIRST HALF and SECOND HALF disagree substantially, the")
    print("  effect is not stable and should not be trusted as-is.")
    print("- 'z-stat vs baseline' beyond about +/-2 suggests the signal's")
    print("  average return this period was unlikely to be pure chance —")
    print("  it is a rough guide, not a proof.")


if __name__ == "__main__":
    main()
