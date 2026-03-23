"""
Fair Value Gap (FVG) Detection.

Definition (strict):
────────────────────
A Fair Value Gap is a 3-candle pattern representing a price inefficiency
that price is likely to revisit.

Bullish FVG (imbalance to the upside):
    bar[i-2].high < bar[i].low
    → The range [bar[i-2].high, bar[i].low] was never traded.
    → Price may return to fill this gap (acting as support).

Bearish FVG (imbalance to the downside):
    bar[i-2].low > bar[i].high
    → The range [bar[i].high, bar[i-2].low] was never traded.
    → Price may return to fill this gap (acting as resistance).

Detection rules:
    1. Gap must be at least fvg_min_size_pct of price (filters noise).
    2. Evaluated on bar[i] CLOSE — no lookahead.
    3. Gaps are "active" until price closes inside them (partial/full fill).
    4. Gaps expire after fvg_max_age_bars bars.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class FVG:
    """Represents a single Fair Value Gap."""
    direction: str          # "bullish" or "bearish"
    top: float              # upper boundary of gap
    bottom: float           # lower boundary of gap
    bar_index: int          # bar where FVG was formed (bar[i])
    bar_time: pd.Timestamp  # timestamp of formation bar
    size: float             # gap size in price units
    size_pct: float         # gap size as % of price
    filled: bool = False
    fill_bar_index: Optional[int] = None


def detect_fvgs(
    df: pd.DataFrame,
    min_size_pct: float = 0.0002,
    max_age_bars: int = 50,
) -> List[FVG]:
    """
    Scan all bars and return a list of FVG objects.

    Only returns UNFILLED gaps that are still within max_age_bars.
    """
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    n = len(df)

    all_fvgs: List[FVG] = []

    # First pass: detect all FVGs
    for i in range(2, n):
        mid_price = closes[i]

        # Bullish FVG: gap between bar[i-2].high and bar[i].low
        if lows[i] > highs[i - 2]:
            gap_bottom = highs[i - 2]
            gap_top = lows[i]
            gap_size = gap_top - gap_bottom
            gap_pct = gap_size / mid_price

            if gap_pct >= min_size_pct:
                fvg = FVG(
                    direction="bullish",
                    top=gap_top,
                    bottom=gap_bottom,
                    bar_index=i,
                    bar_time=df.index[i],
                    size=gap_size,
                    size_pct=gap_pct,
                )
                all_fvgs.append(fvg)

        # Bearish FVG: gap between bar[i].high and bar[i-2].low
        if highs[i] < lows[i - 2]:
            gap_top = lows[i - 2]
            gap_bottom = highs[i]
            gap_size = gap_top - gap_bottom
            gap_pct = gap_size / mid_price

            if gap_pct >= min_size_pct:
                fvg = FVG(
                    direction="bearish",
                    top=gap_top,
                    bottom=gap_bottom,
                    bar_index=i,
                    bar_time=df.index[i],
                    size=gap_size,
                    size_pct=gap_pct,
                )
                all_fvgs.append(fvg)

    # Second pass: mark fills
    for fvg in all_fvgs:
        for j in range(fvg.bar_index + 1, n):
            if fvg.direction == "bullish":
                # Filled when close drops below gap bottom
                if closes[j] < fvg.bottom:
                    fvg.filled = True
                    fvg.fill_bar_index = j
                    break
            else:
                # Filled when close rises above gap top
                if closes[j] > fvg.top:
                    fvg.filled = True
                    fvg.fill_bar_index = j
                    break

    # Filter: only unfilled, within age limit
    last_bar = n - 1
    active = [
        f for f in all_fvgs
        if not f.filled and (last_bar - f.bar_index) <= max_age_bars
    ]

    return active


def get_nearest_fvg(
    fvgs: List[FVG],
    direction: str,
    current_price: float,
    max_distance_pct: float = 0.005,
) -> Optional[FVG]:
    """
    Return the nearest active FVG in the given direction within max_distance_pct.

    For a bullish setup: look for bullish FVG below current price (potential support).
    For a bearish setup: look for bearish FVG above current price (potential resistance).
    """
    candidates = []

    for fvg in fvgs:
        if fvg.direction != direction:
            continue

        if direction == "bullish":
            # FVG should be below current price (unfilled support)
            if fvg.top < current_price:
                distance_pct = (current_price - fvg.top) / current_price
                if distance_pct <= max_distance_pct:
                    candidates.append((distance_pct, fvg))
        else:
            # Bearish FVG should be above current price
            if fvg.bottom > current_price:
                distance_pct = (fvg.bottom - current_price) / current_price
                if distance_pct <= max_distance_pct:
                    candidates.append((distance_pct, fvg))

    if not candidates:
        return None

    # Return closest
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def fvg_to_dataframe(fvgs: List[FVG]) -> pd.DataFrame:
    """Convert FVG list to DataFrame for logging/display."""
    if not fvgs:
        return pd.DataFrame()
    rows = [
        {
            "bar_time": f.bar_time,
            "direction": f.direction,
            "top": f.top,
            "bottom": f.bottom,
            "size": f.size,
            "size_pct": f.size_pct,
            "filled": f.filled,
        }
        for f in fvgs
    ]
    return pd.DataFrame(rows)


def price_in_fvg(price: float, fvg: FVG, buffer_pct: float = 0.0001) -> bool:
    """Return True if price is inside the FVG (with small buffer)."""
    buf = price * buffer_pct
    return (fvg.bottom - buf) <= price <= (fvg.top + buf)
