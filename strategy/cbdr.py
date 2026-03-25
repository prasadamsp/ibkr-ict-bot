"""
Central Bank Dealers Range (CBDR) — ICT 2016 Mentorship Month 8.

Three concepts implemented here:

────────────────────────────────────────────────────────────────────────────
1. CBDR — Central Bank Dealers Range
────────────────────────────────────────────────────────────────────────────
The CBDR is the price range established by institutional (central bank)
dealers during the NY close / pre-Asian window: 20:00–00:00 UTC.

This range represents WHERE institutions set their dealing prices for the
next day. It is the "seed" from which the next day's delivery expands.

  Key rules:
    - Price contained inside CBDR → accumulation still ongoing
    - Price breaks ABOVE CBDR high → bullish delivery / long bias
    - Price breaks BELOW CBDR low  → bearish delivery / short bias
    - Small CBDR range (< 0.03% of price) → choppy / avoid
    - Large CBDR range (> 0.3% of price) → high-volatility session → reduce size

────────────────────────────────────────────────────────────────────────────
2. Projecting Daily Highs & Lows
────────────────────────────────────────────────────────────────────────────
Daily highs and lows are NOT random — they are projected from the CBDR.

  Projected High = CBDR High + (CBDR Range × expansion_factor)
  Projected Low  = CBDR Low  - (CBDR Range × expansion_factor)

  expansion_factor defaults to 1.5 (ICT: "price wants to expand 1.5× the
  dealing range into the next session").

  These become:
    - TP targets for intraday trades
    - Reversal zones if price reaches them early in the day
    - Stop-run targets (buy-side / sell-side liquidity above/below)

────────────────────────────────────────────────────────────────────────────
3. Intraday Profile Classification
────────────────────────────────────────────────────────────────────────────
Each day follows a repeatable behavioral archetype based on how price
interacts with the CBDR during London and NY sessions:

  PROFILE_A  (Classic AMD / Continuation):
    Asian accumulates → London manipulates one CBDR extreme →
    NY distributes in the opposite direction.
    → Entry: after London sweep of CBDR low/high

  PROFILE_B  (Reversal):
    Asian accumulates → London breaks CBDR in one direction →
    NY fully reverses — both CBDR extremes swept in one session.
    → Entry: on the NY reversal, confirmed by MSS

  PROFILE_C  (Inside Day / Choppy):
    Price never clearly breaks either CBDR extreme.
    All sessions range-bound within the CBDR.
    → No trade / size reduction

  PROFILE_D  (Directional / Trending):
    Price breaks CBDR in one direction and never looks back.
    No London sweep of the opposite side.
    → Continuation entry only, no counter-trend

────────────────────────────────────────────────────────────────────────────
CBDR Window (UTC): 20:00 – 00:00
Min bars required: 4 M15 bars
Min range: 0.03% of price (filter flat sessions)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CBDR_START_H = 20    # 20:00 UTC
_CBDR_END_H   = 0     # 00:00 UTC (midnight)

_MIN_RANGE_PCT      = 0.0003   # 0.03% — skip flat sessions
_HIGH_VOL_PCT       = 0.003    # 0.30% — high volatility flag
_EXPANSION_FACTOR   = 1.5      # projected H/L = CBDR ± (range × 1.5)
_SWEEP_BUFFER_PCT   = 0.0001   # 0.01% buffer for sweep detection


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class CBDRState:
    """
    Full CBDR context for the current trading day.

    Attributes
    ----------
    cbdr_high : float       Highest high during the CBDR window.
    cbdr_low  : float       Lowest low during the CBDR window.
    cbdr_range : float      cbdr_high - cbdr_low.
    is_valid  : bool        False if range too small or missing bars.
    is_high_vol : bool      True if CBDR range > HIGH_VOL_PCT (reduce size).
    projected_high : float  Bullish TP target: cbdr_high + range × 1.5
    projected_low  : float  Bearish TP target: cbdr_low  - range × 1.5
    breakout_direction : str  "bullish" | "bearish" | "none"
                              Whether price has already broken the CBDR.
    intraday_profile : str  "A_classic" | "B_reversal" | "C_inside" | "D_directional"
    expansion_bias : str    "bullish" | "bearish" | "neutral"
                            Expected delivery direction for current session.
    """
    cbdr_high: float
    cbdr_low: float
    cbdr_range: float
    is_valid: bool
    is_high_vol: bool
    projected_high: float
    projected_low: float
    breakout_direction: str     # "bullish" | "bearish" | "none"
    intraday_profile: str       # "A_classic" | "B_reversal" | "C_inside" | "D_directional"
    expansion_bias: str         # "bullish" | "bearish" | "neutral"

    @property
    def midpoint(self) -> float:
        return (self.cbdr_high + self.cbdr_low) / 2

    def size_multiplier(self) -> float:
        """Position size multiplier based on CBDR context."""
        if not self.is_valid:
            return 0.75
        if self.is_high_vol:
            return 0.75    # Reduce size on volatile sessions
        if self.intraday_profile == "C_inside":
            return 0.5     # Choppy day — minimal exposure
        return 1.0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_cbdr_state(
    m15_df: pd.DataFrame,
    current_time: pd.Timestamp,
    expansion_factor: float = _EXPANSION_FACTOR,
) -> CBDRState:
    """
    Compute the full CBDR state for the current trading day.

    Args:
        m15_df:       M15 OHLCV DataFrame with UTC timestamps (closed bars).
        current_time: Timestamp of the current M15 bar.
        expansion_factor: Multiplier for projecting daily H/L from CBDR.

    Returns:
        CBDRState with all CBDR context for today's session.
    """
    if m15_df is None or len(m15_df) < 8:
        return _empty_state()

    # Extract CBDR window bars (previous night's 20:00–00:00 UTC)
    cbdr_bars = _extract_cbdr_bars(m15_df, current_time)
    if cbdr_bars is None or len(cbdr_bars) < 4:
        return _empty_state()

    cbdr_high = float(cbdr_bars["high"].max())
    cbdr_low  = float(cbdr_bars["low"].min())
    cbdr_range = cbdr_high - cbdr_low

    # Validate range
    mid = (cbdr_high + cbdr_low) / 2.0
    if mid == 0:
        return _empty_state()

    is_valid   = (cbdr_range / mid) >= _MIN_RANGE_PCT
    is_high_vol = (cbdr_range / mid) >= _HIGH_VOL_PCT

    if not is_valid:
        return CBDRState(
            cbdr_high=cbdr_high,
            cbdr_low=cbdr_low,
            cbdr_range=cbdr_range,
            is_valid=False,
            is_high_vol=False,
            projected_high=cbdr_high,
            projected_low=cbdr_low,
            breakout_direction="none",
            intraday_profile="C_inside",
            expansion_bias="neutral",
        )

    # Project daily high/low
    projected_high = cbdr_high + cbdr_range * expansion_factor
    projected_low  = cbdr_low  - cbdr_range * expansion_factor

    # Analyse today's session bars to detect breakout + classify profile
    today_bars = _extract_today_session_bars(m15_df, current_time)

    breakout_dir = _detect_breakout(today_bars, cbdr_high, cbdr_low) if today_bars is not None else "none"
    profile      = _classify_profile(today_bars, cbdr_high, cbdr_low) if today_bars is not None else "C_inside"

    # Expansion bias: if we have a clear breakout, bias follows the breakout
    if breakout_dir == "bullish":
        expansion_bias = "bullish"
    elif breakout_dir == "bearish":
        expansion_bias = "bearish"
    else:
        expansion_bias = "neutral"

    return CBDRState(
        cbdr_high=cbdr_high,
        cbdr_low=cbdr_low,
        cbdr_range=cbdr_range,
        is_valid=is_valid,
        is_high_vol=is_high_vol,
        projected_high=projected_high,
        projected_low=projected_low,
        breakout_direction=breakout_dir,
        intraday_profile=profile,
        expansion_bias=expansion_bias,
    )


# ---------------------------------------------------------------------------
# CBDR confluence score (used in strategy.py weighted scoring)
# ---------------------------------------------------------------------------

def cbdr_confluence_score(
    cbdr: CBDRState,
    signal_direction: str,
    current_price: float,
) -> float:
    """
    Score how well the signal aligns with CBDR context.

    Returns 0.0–1.0:
      1.0 = signal direction matches CBDR expansion bias
      0.5 = CBDR neutral or invalid (no edge either way)
      0.0 = signal is counter-CBDR (counter-directional to breakout)

    Extra rules:
      - If intraday_profile == C_inside → 0.5 (no clear direction)
      - If is_high_vol → cap at 0.75 (volatile day — reduce confidence)
    """
    if not cbdr.is_valid:
        return 0.5

    if cbdr.intraday_profile == "C_inside":
        return 0.5    # No breakout yet — no directional edge

    if cbdr.expansion_bias == "neutral":
        return 0.5

    score = 1.0 if cbdr.expansion_bias == signal_direction else 0.0

    if cbdr.is_high_vol:
        score = min(score, 0.75)

    return score


# ---------------------------------------------------------------------------
# TP projection from CBDR
# ---------------------------------------------------------------------------

def get_cbdr_tp(
    direction: str,
    cbdr: CBDRState,
    entry_price: float,
    min_rr: float = 2.0,
    stop_distance: float = 0.0,
) -> Optional[float]:
    """
    Return the CBDR projected TP level if it provides adequate RR.

    For a bullish trade: TP = projected_high (if above entry with min_rr)
    For a bearish trade: TP = projected_low  (if below entry with min_rr)

    Returns None if CBDR is invalid or RR is insufficient.
    """
    if not cbdr.is_valid or stop_distance <= 0:
        return None

    if direction == "bullish":
        tp = cbdr.projected_high
        if tp <= entry_price:
            return None
    else:
        tp = cbdr.projected_low
        if tp >= entry_price:
            return None

    rr = abs(tp - entry_price) / stop_distance
    return tp if rr >= min_rr else None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_cbdr_bars(
    m15_df: pd.DataFrame,
    current_time: pd.Timestamp,
) -> Optional[pd.DataFrame]:
    """
    Extract M15 bars from the most recent CBDR window (20:00–00:00 UTC).

    The CBDR window spans two calendar days (20:00 yesterday → 00:00 today).
    """
    today = current_time.normalize()   # 00:00 UTC today

    # Window: yesterday 20:00 → today 00:00
    cbdr_start = (today - pd.Timedelta(days=1)).replace(hour=20, minute=0)
    cbdr_end   = today   # 00:00 UTC today

    mask  = (m15_df.index >= cbdr_start) & (m15_df.index < cbdr_end)
    bars  = m15_df.loc[mask]

    if bars.empty:
        # Try 2 days back (weekend / holiday)
        cbdr_start2 = cbdr_start - pd.Timedelta(days=1)
        cbdr_end2   = cbdr_end   - pd.Timedelta(days=1)
        mask2 = (m15_df.index >= cbdr_start2) & (m15_df.index < cbdr_end2)
        bars  = m15_df.loc[mask2]

    return bars if not bars.empty else None


def _extract_today_session_bars(
    m15_df: pd.DataFrame,
    current_time: pd.Timestamp,
) -> Optional[pd.DataFrame]:
    """
    Extract today's London + NY session bars (07:00–17:00 UTC today).
    These bars are used to detect breakouts relative to the CBDR.
    """
    today = current_time.normalize()
    session_start = today.replace(hour=7, minute=0)
    session_end   = today.replace(hour=17, minute=0)

    mask = (m15_df.index >= session_start) & (m15_df.index <= current_time)
    bars = m15_df.loc[mask]

    # Only include bars up to current_time (no lookahead)
    return bars if not bars.empty else None


def _detect_breakout(
    today_bars: pd.DataFrame,
    cbdr_high: float,
    cbdr_low: float,
) -> str:
    """
    Detect if today's session has broken out of the CBDR.

    Returns "bullish" | "bearish" | "both" | "none"
    If both extremes were swept, return whichever came LAST
    (the final breakout direction is the true one).
    """
    buf_h = cbdr_high * _SWEEP_BUFFER_PCT
    buf_l = cbdr_low  * _SWEEP_BUFFER_PCT

    broke_high = (today_bars["high"] > cbdr_high + buf_h).any()
    broke_low  = (today_bars["low"]  < cbdr_low  - buf_l).any()

    if broke_high and not broke_low:
        return "bullish"
    if broke_low and not broke_high:
        return "bearish"
    if broke_high and broke_low:
        # Both broken — use whichever happened LAST (most recent directional commitment)
        last_high_break = today_bars[today_bars["high"] > cbdr_high + buf_h].index[-1]
        last_low_break  = today_bars[today_bars["low"]  < cbdr_low  - buf_l].index[-1]
        return "bullish" if last_high_break > last_low_break else "bearish"

    return "none"


def _classify_profile(
    today_bars: pd.DataFrame,
    cbdr_high: float,
    cbdr_low: float,
) -> str:
    """
    Classify the intraday profile based on CBDR interaction.

    Profiles:
      A_classic    — London sweeps one extreme, NY reverses (AMD)
      B_reversal   — Both extremes swept in one session (false breakout then reversal)
      C_inside     — Price never left the CBDR range (choppy / avoid)
      D_directional — Price broke CBDR in one direction and held (trending day)
    """
    if today_bars is None or today_bars.empty:
        return "C_inside"

    buf_h = cbdr_high * _SWEEP_BUFFER_PCT
    buf_l = cbdr_low  * _SWEEP_BUFFER_PCT

    broke_high = (today_bars["high"] > cbdr_high + buf_h).any()
    broke_low  = (today_bars["low"]  < cbdr_low  - buf_l).any()

    if not broke_high and not broke_low:
        return "C_inside"      # Never left the range

    if broke_high and broke_low:
        return "B_reversal"    # Both sides swept — reversal day

    # Only one side broken — determine if it held or reversed
    if broke_high:
        # Broke high — check if last close is still above CBDR high
        last_close = float(today_bars["close"].iloc[-1])
        if last_close > cbdr_high:
            return "D_directional"   # Held above — trending up
        else:
            return "A_classic"       # Broke high then came back — classic AMD sweep

    else:  # broke_low
        last_close = float(today_bars["close"].iloc[-1])
        if last_close < cbdr_low:
            return "D_directional"   # Held below — trending down
        else:
            return "A_classic"       # Broke low then reversed — classic Judas swing


def _empty_state() -> CBDRState:
    return CBDRState(
        cbdr_high=0.0,
        cbdr_low=0.0,
        cbdr_range=0.0,
        is_valid=False,
        is_high_vol=False,
        projected_high=0.0,
        projected_low=0.0,
        breakout_direction="none",
        intraday_profile="C_inside",
        expansion_bias="neutral",
    )
