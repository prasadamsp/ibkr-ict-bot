"""
Breaker Blocks — ICT Advanced Order Flow Concept.

A Breaker Block is a former Order Block that has been FULLY MITIGATED
(price closed through the entire zone), followed by a Break of Structure
in the OPPOSITE direction. The old zone now acts as institutional support
or resistance in the new direction.

Identification:
──────────────
  Bullish Breaker:
    1. Was a BEARISH Order Block (last bearish candle before bullish impulse)
    2. Price mitigated it (traded back through it)
    3. A BEARISH BOS occurred after mitigation → price reversed lower
    4. That former bearish OB now becomes a BULLISH breaker:
       when price pulls back into the zone → long entry

  Bearish Breaker:
    1. Was a BULLISH Order Block (last bullish candle before bearish impulse)
    2. Price mitigated it
    3. A BULLISH BOS occurred after mitigation → price reversed higher
    4. That former bullish OB now becomes a BEARISH breaker:
       when price rallies back into the zone → short entry

Why breakers work:
──────────────────
Institutions accumulated orders at the OB. When the OB was mitigated and
price reversed, those positions are now in a different direction. The zone
is a natural liquidity pool — institutions defend it on the retrace.

Breakers are HIGHER PROBABILITY than fresh OBs because:
  - They have proven price reacts there (two reactions)
  - They align with the current market structure direction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from strategy.order_blocks import OrderBlock, detect_order_blocks
from strategy.structure import detect_bos


@dataclass
class BreakerBlock:
    """A former OB that has flipped polarity after mitigation + BOS."""
    direction: str       # "bullish" (former bearish OB) or "bearish" (former bullish OB)
    top: float
    bottom: float
    origin_bar_index: int    # index of the original OB candle
    break_bar_index: int     # index when confirming BOS occurred
    active: bool = True      # False once price trades through the breaker itself

    @property
    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2


def detect_breakers(
    df: pd.DataFrame,
    lookback: int = 5,
    max_age_bars: int = 80,
) -> List[BreakerBlock]:
    """
    Detect all active Breaker Blocks in the bar history.

    Algorithm:
    1. Detect all order blocks (with mitigation flags).
    2. For each MITIGATED OB, look for a BOS in the SAME direction as the OB
       (e.g. bearish OB mitigated → look for bearish BOS after mitigation).
    3. If found, the OB zone becomes a breaker in the OPPOSITE direction.
    4. Mark breaker as inactive if price has since traded through it.

    Args:
        df:            OHLCV DataFrame (M15 or H1 bars).
        lookback:      Swing lookback for BOS detection.
        max_age_bars:  Ignore breakers older than this many bars.

    Returns:
        List of BreakerBlock, newest first.
    """
    if len(df) < lookback * 4:
        return []

    obs        = detect_order_blocks(df, lookback=lookback, max_age_bars=max_age_bars * 2)
    bos_df     = detect_bos(df, lookback=lookback)
    closes     = df["close"].values
    highs      = df["high"].values
    lows       = df["low"].values
    n          = len(df)
    current_bar = n - 1

    breakers: List[BreakerBlock] = []

    for ob in obs:
        # Only process mitigated OBs
        if not ob.mitigated or ob.mitigation_bar_index is None:
            continue

        mit_idx = ob.mitigation_bar_index

        # Age check — skip very old breakers
        if current_bar - ob.bar_index > max_age_bars:
            continue

        if ob.direction == "bearish":
            # Bearish OB mitigated → look for BEARISH BOS after mitigation
            # → becomes a BULLISH breaker (price should bounce up at old zone)
            bos_after = bos_df["bos_bearish"].iloc[mit_idx:].values
            if not bos_after.any():
                continue
            bos_offset = int(np.argmax(bos_after))
            break_idx  = mit_idx + bos_offset

            breaker = BreakerBlock(
                direction="bullish",
                top=ob.top,
                bottom=ob.bottom,
                origin_bar_index=ob.bar_index,
                break_bar_index=break_idx,
                active=True,
            )

        else:  # ob.direction == "bullish"
            # Bullish OB mitigated → look for BULLISH BOS after mitigation
            # → becomes a BEARISH breaker
            bos_after = bos_df["bos_bullish"].iloc[mit_idx:].values
            if not bos_after.any():
                continue
            bos_offset = int(np.argmax(bos_after))
            break_idx  = mit_idx + bos_offset

            breaker = BreakerBlock(
                direction="bearish",
                top=ob.top,
                bottom=ob.bottom,
                origin_bar_index=ob.bar_index,
                break_bar_index=break_idx,
                active=True,
            )

        # Deactivate if price has since traded through the breaker zone
        if breaker.direction == "bullish":
            # Deactivated if close has gone below breaker bottom after break
            lows_after_break = lows[break_idx:]
            if len(lows_after_break) > 0 and lows_after_break.min() < breaker.bottom:
                breaker.active = False
        else:
            highs_after_break = highs[break_idx:]
            if len(highs_after_break) > 0 and highs_after_break.max() > breaker.top:
                breaker.active = False

        breakers.append(breaker)

    # Sort newest first
    breakers.sort(key=lambda b: b.break_bar_index, reverse=True)
    return breakers


def get_nearest_breaker(
    breakers: List[BreakerBlock],
    direction: str,
    current_price: float,
    max_distance_pct: float = 0.008,
) -> Optional[BreakerBlock]:
    """
    Return the nearest active Breaker Block in the given direction that price
    is approaching (within max_distance_pct of the zone).

    For bullish: price should be at or just below the breaker zone.
    For bearish: price should be at or just above the breaker zone.
    """
    best: Optional[BreakerBlock] = None
    best_dist = float("inf")

    for bb in breakers:
        if not bb.active or bb.direction != direction:
            continue

        if direction == "bullish":
            # Price approaching from below — within distance of zone bottom
            if current_price <= bb.top:
                dist = abs(current_price - bb.midpoint) / current_price
                if dist < max_distance_pct and dist < best_dist:
                    best_dist = dist
                    best = bb

        else:  # bearish
            # Price approaching from above — within distance of zone top
            if current_price >= bb.bottom:
                dist = abs(current_price - bb.midpoint) / current_price
                if dist < max_distance_pct and dist < best_dist:
                    best_dist = dist
                    best = bb

    return best
