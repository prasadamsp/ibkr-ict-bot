"""
macro_filters.py — Macro context filters for algorithm selection.

These are applied as pre-filters to each bar before signal generation.
They capture the macro regime that can invalidate a signal even if the
local price structure looks valid.

Filters available:
  DXY proxy         — USD strength from EURUSD inverse slope
  Carry proxy       — Interest rate differential direction (for JPY crosses)
  Risk-on/off       — ATR-based volatility regime (replaces VIX)
  Trend alignment   — Is H1 EMA in the same direction as the signal?
  Session filter    — Only trade during active session hours

All filters return a bool (True = allow trade, False = block).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Hard-coded carry differentials (update quarterly)
# Positive = first currency has higher rates → carry long bias
# ---------------------------------------------------------------------------
_CARRY_DIFFERENTIAL = {
    "GBPJPY":  1,    # GBP rates > JPY rates → long GBP/JPY carry
    "EURUSD":  -1,   # USD rates > EUR rates → short EUR/USD carry bias
    "GBPUSD":  0,    # roughly neutral late 2025
    "XAUUSD":  1,    # gold bullish when real rates fall
    "XAGUSD":  1,
    "NAS100":  1,    # equity bullish when rates stable/falling
    "BTC":     1,    # bull phase (post-halving 2024)
    "OIL":     0,    # neutral
}

# Trading sessions (UTC hour ranges) per instrument
_ACTIVE_SESSIONS = {
    "EURUSD":  [(7, 17)],
    "GBPUSD":  [(7, 17)],
    "GBPJPY":  [(7, 9)],      # Tokyo+London overlap only
    "XAUUSD":  [(7, 17)],
    "XAGUSD":  [(7, 17)],
    "NAS100":  [(13, 21)],    # US session
    "BTC":     [(0, 24)],     # 24/7
    "OIL":     [(13, 17)],    # US session for WTI
}


def usd_strength_bias(eurusd_h1: Optional[pd.DataFrame]) -> int:
    """
    Estimate USD strength from EURUSD H1.
    Returns:  1 = USD strong (EURUSD falling)
              0 = neutral
             -1 = USD weak (EURUSD rising)
    """
    if eurusd_h1 is None or len(eurusd_h1) < 20:
        return 0
    close = eurusd_h1["close"].astype(float)
    ema20 = close.ewm(span=20, adjust=False).mean()
    slope = float(ema20.iloc[-1]) - float(ema20.iloc[-5])
    norm  = slope / float(close.iloc[-1]) if float(close.iloc[-1]) != 0 else 0
    if norm < -0.0002:
        return 1   # EURUSD falling → USD strong
    elif norm > 0.0002:
        return -1  # EURUSD rising → USD weak
    return 0


def carry_bias(symbol: str) -> int:
    """Return carry differential bias for the symbol (+1, 0, -1)."""
    return _CARRY_DIFFERENTIAL.get(symbol, 0)


def in_active_session(symbol: str, current_dt: datetime) -> bool:
    """Return True if current time is within the active trading session."""
    sessions = _ACTIVE_SESSIONS.get(symbol, [(0, 24)])
    h = current_dt.hour
    return any(start <= h < end for start, end in sessions)


def risk_on_off(h1_df: pd.DataFrame) -> int:
    """
    Volatility regime proxy (no VIX required).
    Returns:  1 = risk-on  (low vol, trend up)
              0 = neutral
             -1 = risk-off (vol spike, safe-haven bid)
    """
    if h1_df is None or len(h1_df) < 30:
        return 0

    close = h1_df["close"].astype(float)
    high  = h1_df["high"].astype(float)
    low   = h1_df["low"].astype(float)
    tr    = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)

    atr14 = float(tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1])
    atr5  = float(tr.ewm(alpha=1/5,  adjust=False).mean().iloc[-1])
    price = float(close.iloc[-1])

    if price <= 0:
        return 0

    atr_pct = atr14 / price
    # Spike: short ATR >> long ATR
    if atr5 / atr14 > 1.8 or atr_pct > 0.018:
        return -1   # risk-off
    elif atr5 / atr14 < 1.2 and atr_pct < 0.008:
        return 1    # risk-on
    return 0


def macro_confluence_delta(
    symbol: str,
    direction: str,
    h1_df: pd.DataFrame,
    eurusd_h1: Optional[pd.DataFrame] = None,
) -> float:
    """
    Return how much to RAISE the confluence threshold when macro is a headwind.

    Instead of blocking trades outright, this raises the bar so that only
    higher-quality setups survive against the macro tide.

    Returns
    -------
    float — add this to min_confluence_score before the threshold check.
        0.00  → macro aligned or neutral — trade normally
        0.10  → mild headwind — require slightly more confirmation
        0.15  → strong headwind — require meaningfully more confirmation

    Examples
    --------
    - USD strong + signal is long EURUSD → +0.15 (strong headwind)
      (note: outright block already handled by macro_allows_signal;
       this fires for the cases that slip through allow_counter_carry=True)
    - Risk-off + long NAS100 → +0.10 (neutral macro blocks this outright,
      but if allow_risk_off=True, raise the bar)
    - Carry against direction, neutral USD → +0.08
    """
    delta = 0.0
    usd_bias = usd_strength_bias(eurusd_h1)
    roo      = risk_on_off(h1_df) if (h1_df is not None and len(h1_df) >= 30) else 0
    cb       = carry_bias(symbol)

    # USD headwind for FX pairs
    usd_sensitive = {"EURUSD", "GBPUSD", "GBPJPY"}
    if symbol in usd_sensitive:
        if usd_bias == 1 and direction == "long":
            delta = max(delta, 0.15)   # trading long cable/euro vs strong USD
        elif usd_bias == -1 and direction == "short":
            delta = max(delta, 0.15)

    # Risk-off headwind for risk assets
    risk_assets = {"NAS100", "OIL"}
    if roo == -1 and symbol in risk_assets:
        delta = max(delta, 0.10)

    # BTC: in strong risk-off (vol spike), require more confirmation
    if symbol == "BTC" and roo == -1:
        delta = max(delta, 0.10)

    # Carry headwind
    if cb == 1 and direction == "short":
        delta = max(delta, 0.08)
    elif cb == -1 and direction == "long":
        delta = max(delta, 0.08)

    return delta


def macro_allows_signal(
    symbol: str,
    direction: str,
    current_dt: datetime,
    h1_df: pd.DataFrame,
    eurusd_h1: Optional[pd.DataFrame] = None,
    allow_counter_carry: bool = False,
) -> bool:
    """
    Combined macro gate — returns True if macro context allows the trade.

    Rules:
    1. Must be in active session.
    2. Risk-off regime: only allow long Gold/Silver/JPY (safe havens). Block equities/crypto/oil.
    3. USD strong: favour long USD pairs (short EURUSD, short GBPUSD). Block long EURUSD/GBPUSD.
    4. Carry: if not allow_counter_carry, block signals against the carry direction.
    """
    # 1. Session check
    if not in_active_session(symbol, current_dt):
        return False

    # 2. Risk-off regime
    roo = risk_on_off(h1_df)
    if roo == -1:
        safe_havens = {"XAUUSD", "XAGUSD"}
        # In risk-off: only long safe havens, block everything else
        if symbol not in safe_havens:
            return False
        if direction != "long":
            return False

    # 3. USD strength bias
    usd_bias = usd_strength_bias(eurusd_h1)
    usd_sensitive = {"EURUSD": "short", "GBPUSD": "short", "GBPJPY": "short"}
    if usd_bias == 1 and symbol in usd_sensitive:
        # USD strong → favour short the non-USD leg
        if direction == "long":
            return False
    elif usd_bias == -1 and symbol in usd_sensitive:
        if direction == "short":
            return False

    # 4. Carry filter
    if not allow_counter_carry:
        cb = carry_bias(symbol)
        if cb == 1 and direction == "short":
            return False
        if cb == -1 and direction == "long":
            return False

    return True
