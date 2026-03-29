"""
seasonality_optimizer.py — Find the best months + hours for each algo/instrument.

For each (symbol, algo) pair, this module:
1. Splits trades into monthly and hourly buckets.
2. Computes win rate, avg return, and Sharpe per bucket.
3. Returns the active_months and active_hours with positive expectancy.

Results feed into the AutoSelector's seasonal filters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

_log = logging.getLogger("research")

_MIN_TRADES_PER_BUCKET = 5   # skip buckets with fewer than this many trades


@dataclass
class SeasonalProfile:
    """Optimal seasonal windows for one (symbol, algo) pair."""
    symbol:         str
    algo_name:      str
    active_months:  Set[int]   = field(default_factory=lambda: set(range(1, 13)))
    active_hours:   Set[int]   = field(default_factory=lambda: set(range(0, 24)))
    monthly_sharpe: Dict[int, float] = field(default_factory=dict)
    hourly_sharpe:  Dict[int, float] = field(default_factory=dict)
    best_sharpe:    float = 0.0

    def allows(self, dt: pd.Timestamp) -> bool:
        """Return True if dt falls within the optimal seasonal window."""
        return dt.month in self.active_months and dt.hour in self.active_hours


def _trade_returns(
    m15_df:    pd.DataFrame,
    algo,
    h1_df:     pd.DataFrame,
    month_filter: Optional[Set[int]] = None,
    hour_filter:  Optional[Set[int]] = None,
) -> Tuple[List[float], List[pd.Timestamp]]:
    """
    Simulate all trades for the algo on the full dataset.

    Returns (returns_list, entry_times_list).
    Each return is the fractional P&L: (exit - entry) / entry × direction.
    """
    returns: List[float] = []
    times:   List[pd.Timestamp] = []

    close = m15_df["close"].astype(float)
    n = len(m15_df)

    for i in range(50, n - 1, 4):   # stride=4 for speed
        ts = m15_df.index[i]
        if not hasattr(ts, "month"):
            ts = pd.Timestamp(ts)

        if month_filter and ts.month not in month_filter:
            continue
        if hour_filter and ts.hour not in hour_filter:
            continue

        try:
            sig = algo.generate(m15_df.iloc[:i + 1], h1_df, i)
        except Exception:
            continue

        if sig is None:
            continue

        # Simulate next-bar fill at open; exit at TP or SL within next 8 bars
        for j in range(1, min(9, n - i)):
            bar = m15_df.iloc[i + j]
            hi  = float(bar["high"])
            lo  = float(bar["low"])
            cl  = float(bar["close"])

            if sig.direction == "long":
                if lo <= sig.sl:
                    pnl = (sig.sl - sig.entry) / sig.entry
                    break
                if hi >= sig.tp:
                    pnl = (sig.tp - sig.entry) / sig.entry
                    break
                if j == 8:
                    pnl = (cl - sig.entry) / sig.entry
            else:
                if hi >= sig.sl:
                    pnl = (sig.entry - sig.sl) / sig.entry
                    break
                if lo <= sig.tp:
                    pnl = (sig.entry - sig.tp) / sig.entry
                    break
                if j == 8:
                    pnl = (sig.entry - cl) / sig.entry
        else:
            continue

        returns.append(pnl)
        times.append(ts)

    return returns, times


def _bucket_sharpe(returns: List[float], times: List[pd.Timestamp], attr: str) -> Dict[int, float]:
    """Compute Sharpe per bucket (month or hour)."""
    buckets: Dict[int, List[float]] = {}
    for r, t in zip(returns, times):
        key = getattr(t, attr)
        buckets.setdefault(key, []).append(r)

    result = {}
    for key, rets in buckets.items():
        if len(rets) < _MIN_TRADES_PER_BUCKET:
            continue
        arr  = np.array(rets)
        mean = arr.mean()
        std  = arr.std()
        if std < 1e-10:
            result[key] = 1.0 if mean > 0 else 0.0
        else:
            result[key] = float(mean / std * np.sqrt(1000))
    return result


def optimize_seasonality(
    symbol:    str,
    algo,
    m15_df:    pd.DataFrame,
    h1_df:     pd.DataFrame,
    min_sharpe: float = 0.3,
) -> SeasonalProfile:
    """
    Find the months and hours where this algo has positive expectancy.

    Parameters
    ----------
    min_sharpe : float
        Buckets with Sharpe below this are excluded from active windows.
        0.0 = only exclude losing buckets.

    Returns a SeasonalProfile with filtered active_months and active_hours.
    """
    _log.debug("Seasonal optimizer: %s %s — simulating trades...", symbol, algo.name)

    all_returns, all_times = _trade_returns(m15_df, algo, h1_df)

    if len(all_returns) < 20:
        _log.debug("  Too few trades (%d) for seasonality — using all months/hours", len(all_returns))
        return SeasonalProfile(symbol=symbol, algo_name=algo.name)

    monthly = _bucket_sharpe(all_returns, all_times, "month")
    hourly  = _bucket_sharpe(all_returns, all_times, "hour")

    good_months = {m for m, s in monthly.items() if s >= min_sharpe}
    good_hours  = {h for h, s in hourly.items()  if s >= min_sharpe}

    # Fallback: if too few good buckets, use all
    if len(good_months) < 3:
        good_months = set(range(1, 13))
    if len(good_hours) < 6:
        good_hours = set(range(0, 24))

    # Overall Sharpe on good windows
    filtered = [r for r, t in zip(all_returns, all_times)
                if t.month in good_months and t.hour in good_hours]
    if len(filtered) >= 10:
        arr = np.array(filtered)
        mean, std = arr.mean(), arr.std()
        best_sharpe = float(mean / std * np.sqrt(1000)) if std > 1e-10 else 0.0
    else:
        best_sharpe = 0.0

    _log.debug(
        "  %s %s: active_months=%s good_hours=%d best_sharpe=%.2f",
        symbol, algo.name,
        sorted(good_months), len(good_hours), best_sharpe,
    )

    return SeasonalProfile(
        symbol=symbol,
        algo_name=algo.name,
        active_months=good_months,
        active_hours=good_hours,
        monthly_sharpe=monthly,
        hourly_sharpe=hourly,
        best_sharpe=best_sharpe,
    )
