"""
Power of 3 — ICT AMD (Accumulation / Manipulation / Distribution) Model.

The market moves in a 3-act intraday cycle every 24 hours:

  Phase 1 — ACCUMULATION (Asian session: 23:00–03:00 UTC)
    Institutions build positions quietly inside a compressed range.
    Price consolidates. Range is tight. Retail is confused.
    → The Asian high and low define the 'accumulation range'.

  Phase 2 — MANIPULATION (London open: 07:00–10:00 UTC)
    A FALSE move is engineered to trigger retail stop losses.
    If distribution will be bullish: London will FIRST sweep the Asian LOW
    (triggering sell stops, filling institutional longs cheaply).
    If distribution will be bearish: London will FIRST sweep the Asian HIGH.
    → The Judas Swing. Retail follows the false breakout and gets trapped.

  Phase 3 — DISTRIBUTION (New York: 12:00–17:00 UTC)
    The true directional move begins. Institutions distribute their positions
    into retail trader's losing positions.
    → Long if Asian low was swept in London.
    → Short if Asian high was swept in London.

Algorithm use:
──────────────
1. At NY open (12:00 UTC), check what happened in London session.
2. If London swept the Asian LOW → ONLY take LONGS in NY session.
3. If London swept the Asian HIGH → ONLY take SHORTS in NY session.
4. This acts as a directional filter ON TOP of the existing confluence model.
5. During Asian session: reduce position size (accumulation = low conviction).
6. During London manipulation: be cautious — the sweep sets up the real move.

Asian Range definition:
────────────────────────
  23:00 UTC previous day → 03:00 UTC current day
  Must have at least 2 M15 bars to form a valid range.
  Min range size: 0.02% of price (avoid extremely flat sessions).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd


@dataclass
class AMDState:
    """Current state of the daily AMD cycle."""
    phase: str                       # "accumulation" | "manipulation" | "distribution" | "dead"
    asian_high: float
    asian_low: float
    asian_range_size: float
    manipulation_direction: str      # "swept_lows" | "swept_highs" | "none" | "unknown"
    distribution_bias: str           # "bullish" | "bearish" | "neutral"
    london_extreme: float            # level swept in London (0.0 if not yet swept)
    is_valid: bool                   # False if Asian range too small or missing

    @property
    def asian_midpoint(self) -> float:
        return (self.asian_high + self.asian_low) / 2

    def size_multiplier(self) -> float:
        """Position size multiplier based on AMD phase conviction."""
        return {
            "accumulation": 0.5,    # Low conviction — range building
            "manipulation": 0.75,   # Moderate — setup forming
            "distribution": 1.0,    # Full size — confirmed direction
            "dead": 0.0,            # No trades
        }.get(self.phase, 0.5)


# Session boundaries (UTC)
_ASIAN_START_H  = 23
_ASIAN_END_H    = 3
_LONDON_START_H = 7
_LONDON_END_H   = 11
_NY_START_H     = 12
_NY_END_H       = 17
_DEAD_START_H   = 17
_DEAD_END_H     = 23

_MIN_ASIAN_RANGE_PCT = 0.0002   # 0.02% of price


def get_current_phase(current_time: pd.Timestamp) -> str:
    """Return the current AMD phase based on UTC hour."""
    h = current_time.hour
    if _ASIAN_START_H <= h or h < _ASIAN_END_H:
        return "accumulation"
    if _LONDON_START_H <= h < _LONDON_END_H:
        return "manipulation"
    if _NY_START_H <= h < _NY_END_H:
        return "distribution"
    return "dead"


def get_amd_state(
    m15_df: pd.DataFrame,
    current_time: pd.Timestamp,
) -> AMDState:
    """
    Analyze today's AMD cycle from M15 bars.

    Looks back through the M15 history to find:
      1. Today's Asian session bars → accumulation range
      2. Today's London session bars → manipulation sweep direction
      3. Current phase

    Args:
        m15_df:       M15 OHLCV DataFrame with UTC timestamps (closed bars).
        current_time: Timestamp of the current bar.

    Returns:
        AMDState with full AMD context for today.
    """
    if m15_df is None or len(m15_df) < 4:
        return _empty_state("accumulation")

    phase = get_current_phase(current_time)

    # Extract today's Asian session bars
    asian_bars = _extract_asian_bars(m15_df, current_time)

    if asian_bars is None or len(asian_bars) < 2:
        return _empty_state(phase)

    asian_high = float(asian_bars["high"].max())
    asian_low  = float(asian_bars["low"].min())
    asian_rng  = asian_high - asian_low

    # Validate Asian range isn't degenerate
    mid = (asian_high + asian_low) / 2 if (asian_high + asian_low) > 0 else 1.0
    if mid == 0 or asian_rng / mid < _MIN_ASIAN_RANGE_PCT:
        return AMDState(
            phase=phase,
            asian_high=asian_high,
            asian_low=asian_low,
            asian_range_size=asian_rng,
            manipulation_direction="unknown",
            distribution_bias="neutral",
            london_extreme=0.0,
            is_valid=False,
        )

    # Determine manipulation direction from London session
    manipulation_direction, london_extreme = _detect_london_sweep(
        m15_df, current_time, asian_high, asian_low
    )

    # Distribution bias = OPPOSITE of what was swept
    if manipulation_direction == "swept_lows":
        distribution_bias = "bullish"    # Lows swept → trap sellers → true move up
    elif manipulation_direction == "swept_highs":
        distribution_bias = "bearish"   # Highs swept → trap buyers → true move down
    else:
        distribution_bias = "neutral"

    return AMDState(
        phase=phase,
        asian_high=asian_high,
        asian_low=asian_low,
        asian_range_size=asian_rng,
        manipulation_direction=manipulation_direction,
        distribution_bias=distribution_bias,
        london_extreme=london_extreme,
        is_valid=True,
    )


def _extract_asian_bars(
    m15_df: pd.DataFrame,
    current_time: pd.Timestamp,
) -> Optional[pd.DataFrame]:
    """
    Extract M15 bars from today's (or yesterday's) Asian session.

    Asian = 23:00 UTC previous calendar day → 03:00 UTC current/today.
    """
    # Determine today's date in UTC
    today = current_time.normalize()

    # Asian window: yesterday 23:00 → today 03:00
    asian_start = today - pd.Timedelta(hours=1)   # yesterday 23:00
    asian_start = asian_start.replace(hour=23, minute=0)
    asian_end   = today.replace(hour=3, minute=0)

    mask = (m15_df.index >= asian_start) & (m15_df.index < asian_end)
    bars = m15_df.loc[mask]

    if bars.empty:
        # Try 2 days back (weekend / holiday)
        asian_start2 = asian_start - pd.Timedelta(days=1)
        asian_end2   = asian_end   - pd.Timedelta(days=1)
        mask2 = (m15_df.index >= asian_start2) & (m15_df.index < asian_end2)
        bars = m15_df.loc[mask2]

    return bars if not bars.empty else None


def _detect_london_sweep(
    m15_df: pd.DataFrame,
    current_time: pd.Timestamp,
    asian_high: float,
    asian_low: float,
    sweep_buffer_pct: float = 0.0001,
) -> tuple[str, float]:
    """
    Detect whether London session swept the Asian high or Asian low.

    A sweep = wick or close extends BEYOND the Asian level.
    Uses a small buffer to avoid false triggers on tiny wicks.

    Returns:
        (direction: str, london_extreme: float)
        direction: "swept_lows" | "swept_highs" | "none"
        london_extreme: the price level that was swept (0.0 if none)
    """
    today      = current_time.normalize()
    lon_start  = today.replace(hour=_LONDON_START_H, minute=0)
    lon_end    = today.replace(hour=_LONDON_END_H,   minute=0)

    mask     = (m15_df.index >= lon_start) & (m15_df.index < lon_end)
    lon_bars = m15_df.loc[mask]

    if lon_bars.empty:
        return "none", 0.0

    buffer_h = asian_high * sweep_buffer_pct
    buffer_l = asian_low  * sweep_buffer_pct

    swept_high = (lon_bars["high"] > asian_high + buffer_h).any()
    swept_low  = (lon_bars["low"]  < asian_low  - buffer_l).any()

    if swept_low and not swept_high:
        # London went below Asian lows → manipulation of lows → bullish distribution
        london_extreme = float(lon_bars["low"].min())
        return "swept_lows", london_extreme

    if swept_high and not swept_low:
        # London went above Asian highs → manipulation of highs → bearish distribution
        london_extreme = float(lon_bars["high"].max())
        return "swept_highs", london_extreme

    if swept_low and swept_high:
        # Both swept — use whichever came FIRST
        first_low_bar  = lon_bars[lon_bars["low"]  < asian_low  - buffer_l].index[0]
        first_high_bar = lon_bars[lon_bars["high"] > asian_high + buffer_h].index[0]
        if first_low_bar < first_high_bar:
            return "swept_lows",  float(lon_bars["low"].min())
        else:
            return "swept_highs", float(lon_bars["high"].max())

    return "none", 0.0


def _empty_state(phase: str) -> AMDState:
    return AMDState(
        phase=phase,
        asian_high=0.0,
        asian_low=0.0,
        asian_range_size=0.0,
        manipulation_direction="unknown",
        distribution_bias="neutral",
        london_extreme=0.0,
        is_valid=False,
    )


def amd_confluence_score(
    amd: AMDState,
    signal_direction: str,
) -> float:
    """
    Score how well a signal direction aligns with the AMD distribution bias.

    Returns 0.0–1.0:
      1.0 = perfect alignment (distribution_bias matches signal)
      0.5 = AMD neutral or not yet in distribution phase
      0.0 = AMD says opposite direction (counter-trend — skip)
    """
    if not amd.is_valid:
        return 0.5   # Unknown — neither help nor penalty

    if amd.phase == "dead":
        return 0.0   # No trading in dead zone

    if amd.phase == "accumulation":
        return 0.5   # Building phase — no directional edge yet

    # Manipulation and distribution phases
    if amd.distribution_bias == "neutral":
        return 0.5

    if amd.distribution_bias == signal_direction:
        return 1.0   # Perfect alignment
    else:
        return 0.0   # Counter-AMD — do not trade
