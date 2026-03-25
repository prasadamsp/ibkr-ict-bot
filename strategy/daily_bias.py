"""
Daily Bias — HTF directional context from D1 bars.

The current implementation uses H1 EMA proxies for HTF bias.
This module adds TRUE daily structure: D1 swing highs/lows, D1 BOS, and
weekly range context — preventing counter-trend entries on the wrong side
of the daily/weekly structure.

ICT principle:
──────────────
"Never trade against the daily and weekly bias. The M15 is just the
 delivery mechanism — if D1 says up, only take M15 longs."

Daily bias logic:
─────────────────
1. Detect D1 swing highs and lows (lookback = 3 D1 bars = 3 days)
2. If the most recent D1 BOS is bullish → bias = 1 (bullish)
   If the most recent D1 BOS is bearish → bias = -1 (bearish)
3. If no recent D1 BOS (ranging day) → bias = 0, fall back to H1 EMA

Weekly range:
─────────────
Current week's D1 bars (Mon–Fri):
  - Weekly high / weekly low
  - If price is below weekly midpoint → bullish bias
  - If price is above weekly midpoint → bearish bias
  This prevents buying at the top of the weekly range.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Tuple

import pandas as pd

from strategy.structure import detect_bos, find_swing_highs, find_swing_lows


def get_daily_bias(
    d1_df: pd.DataFrame,
    lookback: int = 3,
    bos_lookback_bars: int = 5,
) -> int:
    """
    Determine HTF directional bias from D1 swing structure.

    Args:
        d1_df:             Daily OHLCV DataFrame (UTC-indexed, closed bars).
        lookback:          D1 swing detection lookback (default=3 days each side).
        bos_lookback_bars: How many recent D1 bars to look back for a BOS event.

    Returns:
        1  = bullish  (D1 structure is making HH + HL)
        -1 = bearish  (D1 structure is making LH + LL)
        0  = neutral / unclear
    """
    if d1_df is None or len(d1_df) < lookback * 2 + 2:
        return 0

    bos_df = detect_bos(d1_df, lookback=lookback)

    # Look at the last N D1 bars for a recent BOS
    recent_bullish = bos_df["bos_bullish"].iloc[-bos_lookback_bars:].any()
    recent_bearish = bos_df["bos_bearish"].iloc[-bos_lookback_bars:].any()

    if recent_bullish and not recent_bearish:
        return 1
    if recent_bearish and not recent_bullish:
        return -1
    if recent_bullish and recent_bearish:
        # Both — use the most recent one
        last_bull_idx = bos_df["bos_bullish"].iloc[-bos_lookback_bars:].values[::-1].argmax()
        last_bear_idx = bos_df["bos_bearish"].iloc[-bos_lookback_bars:].values[::-1].argmax()
        return 1 if last_bull_idx < last_bear_idx else -1

    # No recent BOS — fall back to swing structure
    return _swing_structure_bias(d1_df, lookback)


def _swing_structure_bias(d1_df: pd.DataFrame, lookback: int) -> int:
    """Fallback: determine bias from last 2 D1 swing highs and lows."""
    sh_mask = find_swing_highs(d1_df, lookback)
    sl_mask = find_swing_lows(d1_df, lookback)

    sh_prices = d1_df.loc[sh_mask, "high"]
    sl_prices = d1_df.loc[sl_mask, "low"]

    if len(sh_prices) < 2 or len(sl_prices) < 2:
        return 0

    hh = sh_prices.iloc[-1] > sh_prices.iloc[-2]
    hl = sl_prices.iloc[-1] > sl_prices.iloc[-2]
    lh = sh_prices.iloc[-1] < sh_prices.iloc[-2]
    ll = sl_prices.iloc[-1] < sl_prices.iloc[-2]

    if hh and hl:
        return 1
    if lh and ll:
        return -1
    return 0


def get_weekly_range(
    d1_df: pd.DataFrame,
    current_dt: Optional[pd.Timestamp] = None,
) -> Tuple[float, float, float]:
    """
    Get the current week's high, low, and midpoint from D1 bars.

    Args:
        d1_df:      Daily OHLCV DataFrame.
        current_dt: Reference datetime (defaults to last bar).

    Returns:
        (weekly_high, weekly_low, weekly_mid)
        Returns (0.0, 0.0, 0.0) if insufficient data.
    """
    if d1_df is None or d1_df.empty:
        return 0.0, 0.0, 0.0

    ref = current_dt or d1_df.index[-1]
    if hasattr(ref, "tz_localize") and ref.tzinfo is None:
        ref = ref.tz_localize("UTC")

    # ISO week Monday = 0, Friday = 4
    week_start = ref - pd.Timedelta(days=ref.weekday())
    week_bars  = d1_df[d1_df.index >= week_start]

    if week_bars.empty or len(week_bars) < 2:
        # Fall back to last 5 D1 bars
        week_bars = d1_df.iloc[-5:]

    weekly_high = float(week_bars["high"].max())
    weekly_low  = float(week_bars["low"].min())
    weekly_mid  = (weekly_high + weekly_low) / 2

    return weekly_high, weekly_low, weekly_mid


def weekly_bias(
    d1_df: pd.DataFrame,
    current_price: float,
    current_dt: Optional[pd.Timestamp] = None,
) -> int:
    """
    Determine bias relative to the current week's range midpoint.

    Below weekly mid → bullish (price drawn toward weekly high).
    Above weekly mid → bearish (price drawn toward weekly low).
    Within 10% of mid → neutral.

    Returns: 1, -1, or 0.
    """
    wh, wl, wm = get_weekly_range(d1_df, current_dt)
    if wm == 0:
        return 0

    rng = wh - wl
    if rng == 0:
        return 0

    pos = (current_price - wl) / rng  # 0.0 = at low, 1.0 = at high

    if pos < 0.40:
        return 1    # discount — bullish
    if pos > 0.60:
        return -1   # premium — bearish
    return 0        # equilibrium — neutral


def d1_pd_score(
    d1_df: pd.DataFrame,
    current_price: float,
    direction: str,
) -> float:
    """
    Score how well the current price aligns with the D1 premium/discount context.

    Returns 0.0–1.0 where 1.0 = perfect alignment.
    """
    sh_mask = find_swing_highs(d1_df, lookback=3)
    sl_mask = find_swing_lows(d1_df, lookback=3)

    sh_prices = d1_df.loc[sh_mask, "high"]
    sl_prices = d1_df.loc[sl_mask, "low"]

    if sh_prices.empty or sl_prices.empty:
        return 0.5

    d1_high = float(sh_prices.iloc[-1])
    d1_low  = float(sl_prices.iloc[-1])
    d1_rng  = d1_high - d1_low

    if d1_rng <= 0:
        return 0.5

    pos = (current_price - d1_low) / d1_rng

    if direction == "bullish":
        # Best = deepest discount (pos near 0)
        return max(0.0, 1.0 - pos * 2)   # 1.0 at bottom, 0.0 at midpoint, negative clipped
    else:
        # Best = highest premium (pos near 1)
        return max(0.0, (pos - 0.5) * 2)  # 0.0 at midpoint, 1.0 at top
