"""
Order Block (OB) Detection.

Definition (strict, no lookahead):
────────────────────────────────────
An Order Block is the last imbalance candle before a strong impulsive move that
breaks structure (BOS). Price later returns to this zone, which acts as an
institutional support/resistance area.

Bullish Order Block:
    bar[i] is bearish (close < open).
    Any of bar[i+1..i+3] closes ABOVE bar[i].high (impulsive bullish BOS).
    OB zone: bottom = bar[i].close, top = bar[i].open  (body of bearish candle).
    Entry: price pulls back into zone from above → potential long.
    Invalidation: any subsequent bar closes BELOW bar[i].low.

Bearish Order Block:
    bar[i] is bullish (close > open).
    Any of bar[i+1..i+3] closes BELOW bar[i].low (impulsive bearish BOS).
    OB zone: bottom = bar[i].open, top = bar[i].close  (body of bullish candle).
    Entry: price pulls back into zone from below → potential short.
    Invalidation: any subsequent bar closes ABOVE bar[i].high.

Duplicate handling:
    When multiple consecutive qualifying candles exist before one impulsive
    move, only the LAST one (closest to the impulse) is kept — that candle
    carries the most institutional significance.

Detection is evaluated on closed bars only — no lookahead bias.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class OrderBlock:
    """Represents a single Order Block zone."""
    direction: str                       # "bullish" or "bearish"
    top: float                           # upper boundary of OB zone
    bottom: float                        # lower boundary of OB zone
    ob_extreme: float                    # outer wick used for invalidation
                                         #   bullish → bar[i].low  (close below = mitigated)
                                         #   bearish → bar[i].high (close above = mitigated)
    bar_index: int                       # index of the OB candle within df
    bar_time: pd.Timestamp               # timestamp of OB candle
    confirmation_bar_index: int          # bar index that confirmed the BOS
    mitigated: bool = False
    mitigation_bar_index: Optional[int] = None


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_order_blocks(
    df: pd.DataFrame,
    lookback: int = 5,
    max_age_bars: int = 50,
) -> List[OrderBlock]:
    """
    Detect bullish and bearish order blocks.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV dataframe with columns open, high, low, close.
        Index must be datetime (pd.Timestamp).
    lookback : int
        Not used for the core OB scan (kept for API consistency with other
        strategy modules). Reserved for future BOS-swing integration.
    max_age_bars : int
        OBs whose bar_index is more than this many bars before the last bar
        are dropped from the result even if non-mitigated.

    Returns
    -------
    List[OrderBlock]
        Active (non-mitigated), recent order blocks, ordered oldest → newest.
    """
    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    n = len(df)

    # We need at least i + 3 confirmation bars, so scan up to n-4.
    # Bar i can use confirmation bars i+1, i+2, i+3 — all must be valid indices.
    all_obs: List[OrderBlock] = []

    # Track the bar index of the last OB candle we recorded for each impulse
    # move so we can enforce the "last bearish/bullish candle only" rule.
    # Key = confirmation bar index, Value = list position in all_obs
    _bullish_by_confirm: dict = {}   # confirm_bar_idx → index into all_obs
    _bearish_by_confirm: dict = {}

    for i in range(1, n - 1):

        # ── Bullish OB candidate: current bar is bearish ──────────────────
        if closes[i] < opens[i]:
            for j in range(i + 1, min(i + 4, n)):
                if closes[j] > highs[i]:
                    # Impulsive bullish break above OB candle high
                    ob = OrderBlock(
                        direction="bullish",
                        top=opens[i],           # body top = open of bearish candle
                        bottom=closes[i],       # body bottom = close of bearish candle
                        ob_extreme=lows[i],     # invalidation threshold (close below)
                        bar_index=i,
                        bar_time=df.index[i],
                        confirmation_bar_index=j,
                    )
                    # Enforce "last candle only": if a previous bullish OB was
                    # confirmed by the same impulse bar j, replace it — the
                    # current bar[i] is closer to the impulse move.
                    if j in _bullish_by_confirm:
                        all_obs[_bullish_by_confirm[j]] = ob
                    else:
                        _bullish_by_confirm[j] = len(all_obs)
                        all_obs.append(ob)
                    break   # only the first confirming bar matters

        # ── Bearish OB candidate: current bar is bullish ──────────────────
        if closes[i] > opens[i]:
            for j in range(i + 1, min(i + 4, n)):
                if closes[j] < lows[i]:
                    # Impulsive bearish break below OB candle low
                    ob = OrderBlock(
                        direction="bearish",
                        top=closes[i],          # body top = close of bullish candle
                        bottom=opens[i],        # body bottom = open of bullish candle
                        ob_extreme=highs[i],    # invalidation threshold (close above)
                        bar_index=i,
                        bar_time=df.index[i],
                        confirmation_bar_index=j,
                    )
                    if j in _bearish_by_confirm:
                        all_obs[_bearish_by_confirm[j]] = ob
                    else:
                        _bearish_by_confirm[j] = len(all_obs)
                        all_obs.append(ob)
                    break

    # ── Second pass: mark mitigation ──────────────────────────────────────
    for ob in all_obs:
        # Scan every bar after the confirmation bar
        scan_start = ob.confirmation_bar_index + 1
        for j in range(scan_start, n):
            if ob.direction == "bullish":
                # Mitigated when price closes below the OB low (ob_extreme)
                if closes[j] < ob.ob_extreme:
                    ob.mitigated = True
                    ob.mitigation_bar_index = j
                    break
            else:
                # Mitigated when price closes above the OB high (ob_extreme)
                if closes[j] > ob.ob_extreme:
                    ob.mitigated = True
                    ob.mitigation_bar_index = j
                    break

    # ── Filter: active and within age window ──────────────────────────────
    last_bar = n - 1
    active = [
        ob for ob in all_obs
        if not ob.mitigated and (last_bar - ob.bar_index) <= max_age_bars
    ]

    return active


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_nearest_ob(
    obs: List[OrderBlock],
    direction: str,
    current_price: float,
    max_distance_pct: float = 0.005,
) -> Optional[OrderBlock]:
    """
    Return the nearest unmitigated OB that price is approaching.

    Bullish OB: zone sits below current price (price pulling back toward it).
    Bearish OB: zone sits above current price (price pulling back toward it).

    Parameters
    ----------
    obs : List[OrderBlock]
        Active order blocks returned by detect_order_blocks.
    direction : str
        "bullish" or "bearish".
    current_price : float
        Latest market price (e.g. last close).
    max_distance_pct : float
        Maximum distance from current price to OB zone expressed as a fraction
        of current_price (default 0.5 %).

    Returns
    -------
    OrderBlock or None
        Closest qualifying OB, or None if none found within distance.
    """
    candidates: List[tuple] = []

    for ob in obs:
        if ob.direction != direction:
            continue

        if direction == "bullish":
            # OB zone should be below current price
            if ob.top < current_price:
                distance_pct = (current_price - ob.top) / current_price
                if distance_pct <= max_distance_pct:
                    candidates.append((distance_pct, ob))
        else:
            # Bearish OB zone should be above current price
            if ob.bottom > current_price:
                distance_pct = (ob.bottom - current_price) / current_price
                if distance_pct <= max_distance_pct:
                    candidates.append((distance_pct, ob))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def price_in_ob(price: float, ob: OrderBlock, buffer_pct: float = 0.0001) -> bool:
    """
    Return True if price is inside the OB zone (body), with a small buffer.

    Parameters
    ----------
    price : float
        Current price to test.
    ob : OrderBlock
        The order block zone to test against.
    buffer_pct : float
        Fractional buffer added to each boundary (default 0.01 %).
    """
    buf = price * buffer_pct
    return (ob.bottom - buf) <= price <= (ob.top + buf)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def ob_to_dataframe(obs: List[OrderBlock]) -> pd.DataFrame:
    """Convert an OrderBlock list to a DataFrame for logging or display."""
    if not obs:
        return pd.DataFrame()
    rows = [
        {
            "bar_time": ob.bar_time,
            "direction": ob.direction,
            "top": ob.top,
            "bottom": ob.bottom,
            "ob_extreme": ob.ob_extreme,
            "bar_index": ob.bar_index,
            "confirmation_bar_index": ob.confirmation_bar_index,
            "mitigated": ob.mitigated,
            "mitigation_bar_index": ob.mitigation_bar_index,
        }
        for ob in obs
    ]
    return pd.DataFrame(rows)
