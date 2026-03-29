"""
walk_forward_grid.py — Walk-forward grid search over all algo × season combinations.

For each (symbol, algo) pair:
1. Split data: TRAIN 60% / VAL 20% / TEST 20%.
2. Run seasonality optimizer on TRAIN only.
3. Evaluate algo + seasonal filter on VAL window.
4. If val Sharpe ≥ gate: evaluate on TEST.
5. Record results.

Returns a ranked DataFrame of (symbol, algo, val_sharpe, test_sharpe, params).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from research.algos.base         import BaseAlgo, AlgoSignal
from research.seasonality_optimizer import optimize_seasonality, SeasonalProfile

_log = logging.getLogger("research")


@dataclass
class GridResult:
    symbol:       str
    algo_name:    str
    val_sharpe:   float
    test_sharpe:  float
    n_train:      int
    n_val:        int
    n_test:       int
    active_months: List[int]
    active_hours:  List[int]
    passed_gate:  bool
    train_start:  str
    test_end:     str


def _simulate_window(
    m15_df: pd.DataFrame,
    h1_df:  pd.DataFrame,
    algo:   BaseAlgo,
    profile: Optional[SeasonalProfile] = None,
) -> Tuple[float, int]:
    """
    Simulate algo trades on a data window.

    Returns (annualised_sharpe, n_trades).
    """
    close = m15_df["close"].astype(float)
    n = len(m15_df)
    returns = []

    for i in range(50, n - 1, 4):   # stride=4: check every 4th bar (1-hour cadence)
        ts = m15_df.index[i]
        if not hasattr(ts, "month"):
            ts = pd.Timestamp(ts)

        if profile and not profile.allows(ts):
            continue

        try:
            sig = algo.generate(m15_df.iloc[:i + 1], h1_df, i)
        except Exception:
            continue

        if sig is None:
            continue

        # Simulate exit within next 8 bars
        pnl = None
        for j in range(1, min(9, n - i)):
            bar = m15_df.iloc[i + j]
            hi  = float(bar["high"])
            lo  = float(bar["low"])
            cl  = float(bar["close"])

            if sig.direction == "long":
                if lo <= sig.sl:
                    pnl = (sig.sl - sig.entry) / max(sig.entry, 1e-10)
                    break
                if hi >= sig.tp:
                    pnl = (sig.tp - sig.entry) / max(sig.entry, 1e-10)
                    break
                if j == 8:
                    pnl = (cl - sig.entry) / max(sig.entry, 1e-10)
            else:
                if hi >= sig.sl:
                    pnl = (sig.entry - sig.sl) / max(sig.entry, 1e-10)
                    break
                if lo <= sig.tp:
                    pnl = (sig.entry - sig.tp) / max(sig.entry, 1e-10)
                    break
                if j == 8:
                    pnl = (sig.entry - cl) / max(sig.entry, 1e-10)

        if pnl is not None:
            returns.append(pnl)

    if len(returns) < 10:
        return 0.0, len(returns)

    arr  = np.array(returns)
    mean = arr.mean()
    std  = arr.std()
    if std < 1e-10:
        return (1.0 if mean > 0 else 0.0), len(returns)

    sharpe = float(mean / std * np.sqrt(1000))
    return sharpe, len(returns)


def run_grid(
    symbol:     str,
    m15_df:     pd.DataFrame,
    h1_df:      pd.DataFrame,
    algos:      List[BaseAlgo],
    val_gate:   float = 0.5,
    train_frac: float = 0.60,
    val_frac:   float = 0.20,
) -> List[GridResult]:
    """
    Run walk-forward grid search for all algos on one symbol.

    Parameters
    ----------
    val_gate : float
        Minimum val Sharpe to pass to test window.
    """
    n = len(m15_df)
    if n < 500:
        _log.warning("Grid search %s: too few bars (%d)", symbol, n)
        return []

    # Split indices
    train_end = int(n * train_frac)
    val_end   = int(n * (train_frac + val_frac))

    m15_train = m15_df.iloc[:train_end]
    m15_val   = m15_df.iloc[train_end:val_end]
    m15_test  = m15_df.iloc[val_end:]

    # Align H1 windows
    def h1_for(window_df):
        if h1_df is None:
            return None
        idx_start = window_df.index[0]
        idx_end   = window_df.index[-1]
        mask = (h1_df.index >= idx_start) & (h1_df.index <= idx_end)
        sub = h1_df[mask]
        return sub if len(sub) >= 10 else h1_df

    h1_train = h1_for(m15_train)
    h1_val   = h1_for(m15_val)
    h1_test  = h1_for(m15_test)

    results = []

    for algo in algos:
        _log.debug("  Grid: %s × %s ...", symbol, algo.name)

        try:
            # Step 1: optimize seasonality on TRAIN
            h1_for_season = h1_train if h1_train is not None else pd.DataFrame()
            profile = optimize_seasonality(
                symbol, algo, m15_train, h1_for_season, min_sharpe=0.1
            )

            # Step 2: evaluate on VAL
            val_sharpe, n_val = _simulate_window(m15_val, h1_val, algo, profile)

            passed = val_sharpe >= val_gate

            # Step 3: evaluate on TEST (only if passed gate)
            test_sharpe, n_test = (0.0, 0)
            if passed and len(m15_test) >= 100:
                test_sharpe, n_test = _simulate_window(m15_test, h1_test, algo, profile)

            results.append(GridResult(
                symbol       = symbol,
                algo_name    = algo.name,
                val_sharpe   = round(val_sharpe, 3),
                test_sharpe  = round(test_sharpe, 3),
                n_train      = len(m15_train),
                n_val        = n_val,
                n_test       = n_test,
                active_months = sorted(profile.active_months),
                active_hours  = sorted(profile.active_hours),
                passed_gate  = passed,
                train_start  = str(m15_df.index[0])[:10],
                test_end     = str(m15_df.index[-1])[:10],
            ))

        except Exception as exc:
            _log.error("Grid %s × %s error: %s", symbol, algo.name, exc, exc_info=True)
            continue

    results.sort(key=lambda r: r.val_sharpe, reverse=True)
    return results


def grid_to_dataframe(all_results: Dict[str, List[GridResult]]) -> pd.DataFrame:
    """Flatten all results into a ranked DataFrame."""
    rows = []
    for sym, results in all_results.items():
        for r in results:
            rows.append({
                "symbol":        r.symbol,
                "algo":          r.algo_name,
                "val_sharpe":    r.val_sharpe,
                "test_sharpe":   r.test_sharpe,
                "passed_gate":   r.passed_gate,
                "n_val_trades":  r.n_val,
                "n_test_trades": r.n_test,
                "active_months": str(r.active_months),
                "train_start":   r.train_start,
                "test_end":      r.test_end,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["symbol", "val_sharpe"], ascending=[True, False])
    return df
