"""
IPDA Data Ranges — Interbank Price Delivery Algorithm.

ICT's IPDA defines three lookback windows that describe the price range
the algorithm has ALREADY delivered over the past N trading days.
The far extremes of these windows become the primary liquidity targets
the algorithm will seek next.

Data ranges:
────────────
  20-day range:  ~1 month. Near-term liquidity targets. Highest probability.
  40-day range:  ~2 months. Intermediate targets. Tested after 20D cleared.
  60-day range:  ~1 quarter. Major institutional draw. Swing trade targets.

How to use:
───────────
  If price is at the LOW of the 20-day range → likely draw is to the HIGH.
  If price is at the HIGH of the 20-day range → likely draw is to the LOW.
  When 20D target is reached, it gets re-framed and the 40D target is next.

  In a bullish environment:
    - Discount (bottom half of 60D range): accumulate longs
    - Target: 20D high → 40D high → 60D high
    - Each level is a partial TP opportunity

  In a bearish environment:
    - Premium (top half of 60D range): distribute / short
    - Target: 20D low → 40D low → 60D low

TP improvement over raw swing highs:
─────────────────────────────────────
  Old code: TP = nearest M15 swing high above entry (often too close, RR 1–2)
  IPDA code: TP = nearest IPDA range extreme above entry (often 3–5R+ targets)
  This dramatically improves average RR by aligning TPs with institutional draws.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


IPDA_PERIODS = (20, 40, 60)


@dataclass
class IPDARange:
    """IPDA data range for one lookback period."""
    period: int
    high: float
    low: float
    midpoint: float
    # What fraction of the range has been delivered (0.0 to 1.0)
    # 0 = price at bottom, 1 = price at top
    delivery_pct: float

    @property
    def range_size(self) -> float:
        return self.high - self.low

    def is_at_premium(self, price: float, threshold: float = 0.70) -> bool:
        """Price in top threshold% of range → premium / sell zone."""
        if self.range_size == 0:
            return False
        return (price - self.low) / self.range_size > threshold

    def is_at_discount(self, price: float, threshold: float = 0.30) -> bool:
        """Price in bottom threshold% of range → discount / buy zone."""
        if self.range_size == 0:
            return False
        return (price - self.low) / self.range_size < threshold


def calculate_ipda_ranges(
    d1_df: pd.DataFrame,
    current_price: float,
    periods: Tuple[int, ...] = IPDA_PERIODS,
) -> Dict[int, IPDARange]:
    """
    Calculate IPDA data ranges from daily bars.

    Args:
        d1_df:         Daily OHLCV DataFrame (UTC-indexed, closed bars).
        current_price: Current market price (to compute delivery_pct).
        periods:       Lookback periods in trading days (default: 20, 40, 60).

    Returns:
        Dict mapping period → IPDARange. Shorter periods may be missing if
        insufficient bars are available.
    """
    result: Dict[int, IPDARange] = {}

    for period in periods:
        if len(d1_df) < period:
            # Use whatever we have but require at least 5 bars
            if len(d1_df) < 5:
                continue
            window = d1_df
        else:
            window = d1_df.iloc[-period:]

        high = float(window["high"].max())
        low  = float(window["low"].min())
        rng  = high - low

        if rng <= 0:
            continue

        mid          = (high + low) / 2
        delivery_pct = (current_price - low) / rng

        result[period] = IPDARange(
            period=period,
            high=high,
            low=low,
            midpoint=mid,
            delivery_pct=delivery_pct,
        )

    return result


def get_ipda_tp(
    direction: str,
    entry_price: float,
    ipda_ranges: Dict[int, IPDARange],
    min_rr: float = 2.0,
    stop_distance: float = 0.0,
) -> Optional[float]:
    """
    Return the best IPDA-based take-profit level.

    For longs:  find the nearest IPDA range HIGH above entry that gives RR ≥ min_rr.
                Prefer 20D first, then 40D, then 60D.

    For shorts: find the nearest IPDA range LOW below entry that gives RR ≥ min_rr.
                Prefer 20D first, then 40D, then 60D.

    Args:
        direction:    "bullish" or "bearish".
        entry_price:  Planned entry price.
        ipda_ranges:  From calculate_ipda_ranges().
        min_rr:       Minimum reward/risk ratio required.
        stop_distance: Distance from entry to stop (for RR calc). If 0, skip RR filter.

    Returns:
        TP price, or None if no valid IPDA level found.
    """
    candidates: List[Tuple[float, float, int]] = []  # (tp, rr, period)

    for period in sorted(ipda_ranges.keys()):
        r = ipda_ranges[period]

        if direction == "bullish":
            tp = r.high
            if tp <= entry_price:
                continue
            reward = tp - entry_price
        else:
            tp = r.low
            if tp >= entry_price:
                continue
            reward = entry_price - tp

        if stop_distance > 0:
            rr = reward / stop_distance
            if rr < min_rr:
                continue
        else:
            rr = reward / (entry_price * 0.001)  # proxy

        candidates.append((tp, rr, period))

    if not candidates:
        return None

    # Return nearest valid level (smallest RR that still meets minimum)
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


def ipda_bias(
    ipda_ranges: Dict[int, IPDARange],
    current_price: float,
) -> int:
    """
    Determine macro directional bias from IPDA delivery.

    If price is in the lower half of the 60D range → bullish draw (price seeks highs).
    If price is in the upper half of the 60D range → bearish draw (price seeks lows).

    Returns: 1 (bullish), -1 (bearish), 0 (neutral/mid-range).
    """
    r60 = ipda_ranges.get(60) or ipda_ranges.get(40)
    if r60 is None:
        return 0

    if r60.is_at_discount(current_price, threshold=0.40):
        return 1
    if r60.is_at_premium(current_price, threshold=0.60):
        return -1
    return 0
