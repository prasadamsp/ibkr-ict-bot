"""
Liquidity Analysis — Equal Highs/Lows and Liquidity Sweeps.

ICT Concepts:
─────────────
Equal Highs (EQH): Two or more swing highs within a tight tolerance band.
    → Retail stop-losses cluster above these levels.
    → Smart money engineers a sweep (spike above) to grab liquidity, then reverses.

Equal Lows (EQL): Two or more swing lows within a tight tolerance band.
    → Retail stops cluster below these levels.
    → Sweep below, then reversal.

Liquidity Sweep Detection:
    1. Identify EQH/EQL level.
    2. Bar[i].high > EQH level (wick pierces above).
    3. Bar[i].close < EQH level (candle closes BACK below).
    → This confirms a sweep: liquidity grabbed, rejection confirmed.
    → Same logic inverted for EQL sweeps.

We only call it a sweep after the bar closes — no repainting.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from strategy.structure import find_swing_highs, find_swing_lows


@dataclass
class LiquidityLevel:
    """Represents an identified liquidity pool (EQH or EQL)."""
    kind: str           # "eqh" or "eql"
    price: float        # level price
    touches: int        # how many times price has touched/tested
    first_bar: int      # bar index of first occurrence
    last_bar: int       # bar index of most recent test
    swept: bool = False
    sweep_bar: Optional[int] = None


@dataclass
class LiquiditySweep:
    """A confirmed liquidity sweep event."""
    kind: str           # "eqh_sweep" or "eql_sweep"
    level: float        # the level that was swept
    sweep_bar: int      # bar index of the sweep candle
    sweep_time: pd.Timestamp
    wick_extension: float   # how far price pierced beyond level
    close_gap: float        # how far close returned below/above level
    direction: str          # "bearish" (after EQH sweep) or "bullish" (after EQL sweep)


def find_liquidity_levels(
    df: pd.DataFrame,
    lookback: int = 5,
    tolerance_pct: float = 0.0003,
    min_lookback_bars: int = 10,
    max_lookback_bars: int = 60,
) -> List[LiquidityLevel]:
    """
    Find Equal Highs and Equal Lows in the data.

    Two swing highs are "equal" if |h1 - h2| / h1 <= tolerance_pct.
    """
    sh_mask = find_swing_highs(df, lookback)
    sl_mask = find_swing_lows(df, lookback)

    sh_df = df.loc[sh_mask, "high"]
    sl_df = df.loc[sl_mask, "low"]

    # Build list of (bar_idx, price) for each swing
    sh_indices = [df.index.get_loc(t) for t in sh_df.index]
    sl_indices = [df.index.get_loc(t) for t in sl_df.index]

    highs = [(sh_indices[i], float(sh_df.iloc[i])) for i in range(len(sh_df))]
    lows = [(sl_indices[i], float(sl_df.iloc[i])) for i in range(len(sl_df))]

    levels: List[LiquidityLevel] = []

    # Find EQH clusters
    levels.extend(
        _cluster_levels(highs, "eqh", tolerance_pct, min_lookback_bars, max_lookback_bars)
    )
    # Find EQL clusters
    levels.extend(
        _cluster_levels(lows, "eql", tolerance_pct, min_lookback_bars, max_lookback_bars)
    )

    return levels


def _cluster_levels(
    swing_points: list,
    kind: str,
    tolerance_pct: float,
    min_bars: int,
    max_bars: int,
) -> List[LiquidityLevel]:
    """Group nearby swing points into liquidity clusters."""
    levels = []
    used = set()

    for i in range(len(swing_points)):
        if i in used:
            continue
        idx_i, price_i = swing_points[i]
        cluster = [i]

        for j in range(i + 1, len(swing_points)):
            if j in used:
                continue
            idx_j, price_j = swing_points[j]

            bar_gap = idx_j - idx_i
            if bar_gap < min_bars or bar_gap > max_bars:
                continue

            if abs(price_j - price_i) / price_i <= tolerance_pct:
                cluster.append(j)

        if len(cluster) >= 2:
            prices = [swing_points[k][1] for k in cluster]
            idxs = [swing_points[k][0] for k in cluster]
            levels.append(
                LiquidityLevel(
                    kind=kind,
                    price=float(np.mean(prices)),
                    touches=len(cluster),
                    first_bar=min(idxs),
                    last_bar=max(idxs),
                )
            )
            used.update(cluster)

    return levels


def detect_sweeps(
    df: pd.DataFrame,
    levels: List[LiquidityLevel],
) -> List[LiquiditySweep]:
    """
    Scan bars for liquidity sweeps of the identified levels.

    EQH sweep: bar[i].high > level AND bar[i].close < level
    EQL sweep: bar[i].low < level AND bar[i].close > level

    Only confirmed on bar close — no repainting.
    """
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    n = len(df)

    sweeps: List[LiquiditySweep] = []

    for level in levels:
        search_start = level.last_bar + 1  # only look after the level formed

        for i in range(search_start, n):
            if level.swept:
                break

            if level.kind == "eqh":
                # Wick pierces above, close comes back below
                if highs[i] > level.price and closes[i] < level.price:
                    wick_ext = highs[i] - level.price
                    close_gap = level.price - closes[i]

                    sweeps.append(
                        LiquiditySweep(
                            kind="eqh_sweep",
                            level=level.price,
                            sweep_bar=i,
                            sweep_time=df.index[i],
                            wick_extension=wick_ext,
                            close_gap=close_gap,
                            direction="bearish",  # swept highs → expect downside
                        )
                    )
                    level.swept = True
                    level.sweep_bar = i

            elif level.kind == "eql":
                # Wick pierces below, close comes back above
                if lows[i] < level.price and closes[i] > level.price:
                    wick_ext = level.price - lows[i]
                    close_gap = closes[i] - level.price

                    sweeps.append(
                        LiquiditySweep(
                            kind="eql_sweep",
                            level=level.price,
                            sweep_bar=i,
                            sweep_time=df.index[i],
                            wick_extension=wick_ext,
                            close_gap=close_gap,
                            direction="bullish",  # swept lows → expect upside
                        )
                    )
                    level.swept = True
                    level.sweep_bar = i

    return sweeps


def get_recent_sweep(
    sweeps: List[LiquiditySweep],
    direction: str,
    current_bar: int,
    max_age_bars: int = 10,
) -> Optional[LiquiditySweep]:
    """
    Return the most recent sweep in the given direction within max_age_bars.
    Used by the signal engine to check for fresh liquidity grabs.
    """
    candidates = [
        s for s in sweeps
        if s.direction == direction
        and 0 < (current_bar - s.sweep_bar) <= max_age_bars
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda s: s.sweep_bar)
