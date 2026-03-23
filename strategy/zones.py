"""
Premium / Discount Zone Analysis.

ICT Definition:
───────────────
Within any dealing range (from last significant swing low to swing high):

    50% level (Equilibrium):   midpoint of the range.
    Discount zone:             lower 50% — price is "cheap". Look to BUY here.
    Premium zone:              upper 50% — price is "expensive". Look to SELL here.

    Optimal Trade Entry (OTE) — Fibonacci zones:
        Discount OTE:  61.8% – 78.6% retracement (buy zone in uptrend)
        Premium OTE:   61.8% – 78.6% retracement (sell zone in downtrend)

Usage in strategy:
    - In a bullish trend (H1): only take longs when M15 price is in DISCOUNT zone.
    - In a bearish trend (H1): only take shorts when M15 price is in PREMIUM zone.
    - This prevents buying at the top and selling at the bottom.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

from strategy.structure import find_swing_highs, find_swing_lows


@dataclass
class DealingRange:
    """The current dealing range with zone boundaries."""
    swing_low: float
    swing_high: float
    midpoint: float        # 50% — equilibrium
    discount_top: float    # 50% level (same as midpoint)
    premium_bottom: float  # 50% level
    ote_buy_low: float     # 61.8% retrace from high (buy zone bottom)
    ote_buy_high: float    # 78.6% retrace from high (buy zone top)
    ote_sell_low: float    # 21.4% retrace from low (sell zone bottom) = 78.6% retrace from top
    ote_sell_high: float   # 38.2% retrace from low (sell zone top)
    range_size: float


def calculate_dealing_range(df: pd.DataFrame, lookback: int = 5) -> Optional[DealingRange]:
    """
    Calculate the current dealing range from the most recent significant
    swing low and swing high.

    Uses the LAST confirmed swing high and swing low in the data.
    """
    sh_mask = find_swing_highs(df, lookback)
    sl_mask = find_swing_lows(df, lookback)

    sh_prices = df.loc[sh_mask, "high"]
    sl_prices = df.loc[sl_mask, "low"]

    if sh_prices.empty or sl_prices.empty:
        return None

    last_sh = float(sh_prices.iloc[-1])
    last_sl = float(sl_prices.iloc[-1])
    rng = last_sh - last_sl

    if rng <= 0:
        return None

    mid = last_sl + rng * 0.5

    # Fibonacci retracement levels for OTE
    # In uptrend: price retraces from high toward low
    # OTE buy zone = 61.8% to 78.6% retracement FROM the high
    ote_buy_low = last_sh - rng * 0.786   # 78.6% retrace (deeper)
    ote_buy_high = last_sh - rng * 0.618  # 61.8% retrace (shallower)

    # In downtrend: price retraces from low toward high
    # OTE sell zone = 61.8% to 78.6% retracement FROM the low
    ote_sell_low = last_sl + rng * 0.618
    ote_sell_high = last_sl + rng * 0.786

    return DealingRange(
        swing_low=last_sl,
        swing_high=last_sh,
        midpoint=mid,
        discount_top=mid,
        premium_bottom=mid,
        ote_buy_low=ote_buy_low,
        ote_buy_high=ote_buy_high,
        ote_sell_low=ote_sell_low,
        ote_sell_high=ote_sell_high,
        range_size=rng,
    )


def classify_price(price: float, dr: DealingRange) -> str:
    """
    Classify where the current price sits within the dealing range.

    Returns: "discount", "premium", "equilibrium", "ote_buy", "ote_sell", "above_range", "below_range"
    """
    if price > dr.swing_high:
        return "above_range"
    if price < dr.swing_low:
        return "below_range"

    if dr.ote_buy_low <= price <= dr.ote_buy_high:
        return "ote_buy"
    if dr.ote_sell_low <= price <= dr.ote_sell_high:
        return "ote_sell"
    if price < dr.midpoint:
        return "discount"
    if price > dr.midpoint:
        return "premium"
    return "equilibrium"


def price_in_discount(price: float, dr: DealingRange) -> bool:
    """True if price is in the discount zone (below 50%)."""
    return price < dr.midpoint


def price_in_premium(price: float, dr: DealingRange) -> bool:
    """True if price is in the premium zone (above 50%)."""
    return price > dr.midpoint


def price_in_ote_buy(price: float, dr: DealingRange) -> bool:
    """True if price is in the Optimal Trade Entry buy zone."""
    return dr.ote_buy_low <= price <= dr.ote_buy_high


def price_in_ote_sell(price: float, dr: DealingRange) -> bool:
    """True if price is in the Optimal Trade Entry sell zone."""
    return dr.ote_sell_low <= price <= dr.ote_sell_high


def pd_score(price: float, dr: DealingRange, direction: str) -> float:
    """
    Return a 0–1 score for how well price position aligns with direction.

    Bullish trade: higher score = deeper in discount/OTE buy zone.
    Bearish trade: higher score = deeper in premium/OTE sell zone.
    """
    pos = (price - dr.swing_low) / dr.range_size  # 0 = at swing low, 1 = at swing high

    if direction == "bullish":
        # Best entry: deeply discounted (pos near 0.2–0.4)
        if pos <= 0.5:
            return 1.0 - (pos / 0.5)  # 1.0 at bottom, 0.0 at midpoint
        else:
            return 0.0
    else:  # bearish
        # Best entry: deeply premium (pos near 0.6–0.8)
        if pos >= 0.5:
            return (pos - 0.5) / 0.5   # 0.0 at midpoint, 1.0 at top
        else:
            return 0.0
