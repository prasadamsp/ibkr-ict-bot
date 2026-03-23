"""
Market Structure Analysis — BOS and MSS detection.

Definitions (strict, no repainting):
─────────────────────────────────────
Break of Structure (BOS):
    Bullish BOS: Close of bar[i] > highest high of all bars since last swing low.
    Bearish BOS: Close of bar[i] < lowest low of all bars since last swing high.
    → Confirms continuation of current trend.

Market Structure Shift (MSS):
    After a series of higher highs + higher lows (uptrend):
        Price makes a lower high, then close breaks below the most recent swing low.
    After a series of lower highs + lower lows (downtrend):
        Price makes a higher low, then close breaks above the most recent swing high.
    → Signals potential trend reversal.

Swing High / Swing Low:
    A swing high at bar[i] requires:
        bar[i].high > bar[i-N].high AND bar[i].high > bar[i+N].high
        for N = swing_lookback.
    Only confirmed after N bars have closed to the right.
    This means swing detection always lags by N bars — no repainting.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Swing detection
# ---------------------------------------------------------------------------

def find_swing_highs(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """
    Return a boolean Series where True = confirmed swing high.
    Bar[i] is a swing high if it has the highest high among [i-lookback, i+lookback].
    Last `lookback` bars are always False (not yet confirmed).
    """
    highs = df["high"].values
    n = len(highs)
    result = np.zeros(n, dtype=bool)

    for i in range(lookback, n - lookback):
        window = highs[i - lookback : i + lookback + 1]
        if highs[i] == window.max() and np.sum(window == highs[i]) == 1:
            result[i] = True

    return pd.Series(result, index=df.index)


def find_swing_lows(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """
    Return a boolean Series where True = confirmed swing low.
    """
    lows = df["low"].values
    n = len(lows)
    result = np.zeros(n, dtype=bool)

    for i in range(lookback, n - lookback):
        window = lows[i - lookback : i + lookback + 1]
        if lows[i] == window.min() and np.sum(window == lows[i]) == 1:
            result[i] = True

    return pd.Series(result, index=df.index)


def get_swing_levels(
    df: pd.DataFrame, lookback: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return two DataFrames:
        swing_highs_df: index=datetime, columns=[price]
        swing_lows_df:  index=datetime, columns=[price]
    """
    sh_mask = find_swing_highs(df, lookback)
    sl_mask = find_swing_lows(df, lookback)

    sh = df.loc[sh_mask, "high"].rename("price").to_frame()
    sl = df.loc[sl_mask, "low"].rename("price").to_frame()

    return sh, sl


# ---------------------------------------------------------------------------
# Break of Structure (BOS)
# ---------------------------------------------------------------------------

def detect_bos(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """
    Scan the entire bar history and mark BOS events.

    Returns a DataFrame with columns:
        bos_bullish: True where a bullish BOS occurred (close > last swing high)
        bos_bearish: True where a bearish BOS occurred (close < last swing low)
        bos_level:   Price level that was broken

    Notes:
    - Uses closed bar closes only (no lookahead).
    - A BOS is marked on the bar whose CLOSE broke the level.
    - After a BOS, the reference level resets.
    """
    sh_mask = find_swing_highs(df, lookback)
    sl_mask = find_swing_lows(df, lookback)

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    bos_bullish = np.zeros(n, dtype=bool)
    bos_bearish = np.zeros(n, dtype=bool)
    bos_level = np.full(n, np.nan)

    # Track the most recent confirmed swing levels
    last_sh_price: Optional[float] = None
    last_sl_price: Optional[float] = None

    for i in range(n):
        # Update reference levels when new swings confirm
        if sh_mask.iloc[i]:
            last_sh_price = highs[i]
        if sl_mask.iloc[i]:
            last_sl_price = lows[i]

        # Check for BOS on close
        if last_sh_price is not None and closes[i] > last_sh_price:
            bos_bullish[i] = True
            bos_level[i] = last_sh_price
            last_sh_price = None  # consumed — reset so we don't repeat

        if last_sl_price is not None and closes[i] < last_sl_price:
            bos_bearish[i] = True
            bos_level[i] = last_sl_price
            last_sl_price = None

    return pd.DataFrame(
        {"bos_bullish": bos_bullish, "bos_bearish": bos_bearish, "bos_level": bos_level},
        index=df.index,
    )


# ---------------------------------------------------------------------------
# Market Structure Shift (MSS)
# ---------------------------------------------------------------------------

def detect_mss(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """
    Detect Market Structure Shifts.

    Bullish MSS (downtrend reversal):
        1. We have been making lower highs (LH) and lower lows (LL).
        2. Price forms a higher low (HL) — i.e., current swing low > previous swing low.
        3. Close breaks above the most recent swing high.

    Bearish MSS (uptrend reversal):
        1. We have been making higher highs (HH) and higher lows (HL).
        2. Price forms a lower high (LH) — i.e., current swing high < previous swing high.
        3. Close breaks below the most recent swing low.

    Returns DataFrame with:
        mss_bullish: bar index where bullish MSS confirmed
        mss_bearish: bar index where bearish MSS confirmed
        mss_level:   price level broken to confirm MSS
    """
    sh_mask = find_swing_highs(df, lookback)
    sl_mask = find_swing_lows(df, lookback)

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    mss_bullish = np.zeros(n, dtype=bool)
    mss_bearish = np.zeros(n, dtype=bool)
    mss_level = np.full(n, np.nan)

    # Rolling swing history (keep last 3 of each)
    swing_highs: list = []   # list of (index, price)
    swing_lows: list = []

    for i in range(n):
        if sh_mask.iloc[i]:
            swing_highs.append((i, highs[i]))
            if len(swing_highs) > 3:
                swing_highs.pop(0)

        if sl_mask.iloc[i]:
            swing_lows.append((i, lows[i]))
            if len(swing_lows) > 3:
                swing_lows.pop(0)

        # Need at least 2 of each to determine trend
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            continue

        prev_sh = swing_highs[-2][1]
        last_sh = swing_highs[-1][1]
        prev_sl = swing_lows[-2][1]
        last_sl = swing_lows[-1][1]

        # Bullish MSS: downtrend (LH, LL) → higher low → close above last swing high
        in_downtrend = (last_sh < prev_sh) and (last_sl < prev_sl)
        if in_downtrend and last_sl > prev_sl:  # higher low formed
            if closes[i] > last_sh:
                mss_bullish[i] = True
                mss_level[i] = last_sh

        # Bearish MSS: uptrend (HH, HL) → lower high → close below last swing low
        in_uptrend = (last_sh > prev_sh) and (last_sl > prev_sl)
        if in_uptrend and last_sh < prev_sh:  # lower high formed
            if closes[i] < last_sl:
                mss_bearish[i] = True
                mss_level[i] = last_sl

    return pd.DataFrame(
        {"mss_bullish": mss_bullish, "mss_bearish": mss_bearish, "mss_level": mss_level},
        index=df.index,
    )


# ---------------------------------------------------------------------------
# Trend bias (simplified — used by signal engine)
# ---------------------------------------------------------------------------

def get_trend_bias(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """
    Return a Series of trend bias per bar: 1 = bullish, -1 = bearish, 0 = neutral.

    Logic: track last 2 swing highs and 2 swing lows.
    HH + HL → bullish. LH + LL → bearish. Mixed → neutral.
    """
    sh_mask = find_swing_highs(df, lookback)
    sl_mask = find_swing_lows(df, lookback)

    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    bias = np.zeros(n, dtype=int)
    swing_highs = []
    swing_lows = []

    for i in range(n):
        if sh_mask.iloc[i]:
            swing_highs.append(highs[i])
            if len(swing_highs) > 2:
                swing_highs.pop(0)
        if sl_mask.iloc[i]:
            swing_lows.append(lows[i])
            if len(swing_lows) > 2:
                swing_lows.pop(0)

        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            hh = swing_highs[-1] > swing_highs[-2]
            hl = swing_lows[-1] > swing_lows[-2]
            lh = swing_highs[-1] < swing_highs[-2]
            ll = swing_lows[-1] < swing_lows[-2]

            if hh and hl:
                bias[i] = 1
            elif lh and ll:
                bias[i] = -1
            else:
                bias[i] = 0
        elif i > 0:
            bias[i] = bias[i - 1]  # carry forward

    return pd.Series(bias, index=df.index, name="trend_bias")
