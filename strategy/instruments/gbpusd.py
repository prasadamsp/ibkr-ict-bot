"""
gbpusd.py — GBPUSD Instrument Strategy

Strategy logic overview
-----------------------
GBPUSD is notorious for its London open "Judas Swing" — an aggressive false
move engineered to stop out retail traders before the real session direction
is revealed.  This module hunts exclusively for that pattern.

The complete trading logic:

1. Volatility gate
   Only trade when M15 ATR(14) × 4 > 40 pips (0.0040).  This is a proxy for
   the hourly ATR.  Below this level the market is too quiet for reliable
   ICT setups — spreads are wide relative to moves, and fake-outs are smaller
   and harder to exploit.

2. Judas Swing detection (07:00–08:30 UTC)
   At the London open, the market frequently makes a sharp move of 20+ pips
   against the "expected" direction.  The expected direction is determined by
   the H1 market structure: the most recent confirmed Break of Structure (BOS)
   direction is the true bias.
   - If H1 BOS is bullish (uptrend) and price drops 20+ pips → Judas Swing
     long opportunity (market will reverse upward).
   - If H1 BOS is bearish (downtrend) and price rises 20+ pips → Judas Swing
     short opportunity.

3. M15 FVG entry
   After the Judas move, wait for a Fair Value Gap to form in the snap-back
   direction on M15.  Enter at the FVG midpoint (limit order).

4. Session expiry
   If no qualifying Judas Swing is detected before 09:00 UTC, the session is
   skipped entirely.  Quality over quantity.

Risk / Reward
-------------
- Stop: beyond the Judas Swing wick extreme (0.02% buffer).
- Target: 2× stop distance minimum (placed toward previous session liquidity,
  approximated as the nearest M15 swing in the target direction).
- Minimum 2R.

Note: GBPUSD has no meaningful seasonal edge.  The seasonality multiplier
is always 1.0 (neutral) per SeasonalityCalendar.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd

from strategy.fvg import detect_fvgs, get_nearest_fvg
from strategy.instruments.base import BaseInstrumentStrategy
from strategy.sessions import session_score
from strategy.strategy import TradeSignal
from strategy.structure import detect_bos, find_swing_highs, find_swing_lows

_log = logging.getLogger("strategy")

# Volatility gate: M15 ATR(14) * 4 must exceed this many price units (~40 pips)
_MIN_HOURLY_ATR = 0.0040
# Judas Swing: price must move at least this far against expected direction
_JUDAS_MIN_MOVE = 0.0020   # 20 pips
# Session windows (UTC hour, inclusive start / exclusive end)
_LONDON_OPEN_START = 7
_LONDON_OPEN_END   = 9      # signal expires at 09:00
_JUDAS_WINDOW_END  = 9      # skip session after this hour

_MIN_RR = 2.0
_SL_BUFFER = 0.0002


class GBPUSDStrategy(BaseInstrumentStrategy):
    """Judas Swing strategy for GBPUSD at the London open.

    See module docstring for full strategy description.
    """

    def generate_signal(
        self,
        m15_df: pd.DataFrame,
        h1_df: pd.DataFrame,
        current_dt: datetime,
    ) -> Optional[TradeSignal]:
        try:
            return self._generate(m15_df, h1_df, current_dt)
        except Exception as exc:
            _log.error("GBPUSD generate_signal error: %s", exc, exc_info=True)
            return None

    # ------------------------------------------------------------------

    def _generate(
        self,
        m15_df: pd.DataFrame,
        h1_df: pd.DataFrame,
        current_dt: datetime,
    ) -> Optional[TradeSignal]:

        # --- Minimum bar guard ---
        if len(m15_df) < 50 or len(h1_df) < 30:
            _log.debug("GBPUSD: insufficient bars (m15=%d h1=%d)", len(m15_df), len(h1_df))
            return None

        # --- Session window check: only trade in Judas window ---
        h = current_dt.hour
        if not (_LONDON_OPEN_START <= h < _LONDON_OPEN_END):
            _log.debug("GBPUSD: outside Judas Swing window (hour=%d UTC)", h)
            return None

        # --- Volatility gate ---
        atr14 = self._calc_atr14(m15_df)
        hourly_atr_proxy = atr14 * 4.0
        if hourly_atr_proxy < _MIN_HOURLY_ATR:
            _log.debug(
                "GBPUSD: hourly ATR proxy %.5f < %.5f minimum, skipping",
                hourly_atr_proxy, _MIN_HOURLY_ATR,
            )
            return None

        # --- H1 expected direction from BOS ---
        expected_direction = self._h1_bos_direction(h1_df)
        if expected_direction is None:
            _log.debug("GBPUSD: no confirmed H1 BOS direction")
            return None

        # --- Detect Judas Swing: 20+ pip move against expected direction ---
        entry_direction, judas_extreme = self._detect_judas_swing(
            m15_df, expected_direction
        )
        if entry_direction is None:
            _log.debug("GBPUSD: no Judas Swing detected")
            return None

        current_price = float(m15_df.iloc[-1]["close"])

        # --- M15 FVG in snap-back direction ---
        fvgs = detect_fvgs(m15_df, min_size_pct=0.0001, max_age_bars=20)
        nearest_fvg = get_nearest_fvg(fvgs, entry_direction, current_price, max_distance_pct=0.006)
        if nearest_fvg is None:
            _log.debug("GBPUSD: no M15 FVG found after Judas Swing")
            return None

        # --- Entry / SL / TP ---
        entry = (nearest_fvg.top + nearest_fvg.bottom) / 2.0
        if entry_direction == "bullish":
            sl = judas_extreme * (1.0 - _SL_BUFFER)
            sh_mask = find_swing_highs(m15_df, lookback=5)
            tp_candidates = m15_df.loc[sh_mask, "high"]
            tp_candidates = tp_candidates[tp_candidates > entry]
            if tp_candidates.empty:
                return None
            tp = float(tp_candidates.iloc[0])
        else:
            sl = judas_extreme * (1.0 + _SL_BUFFER)
            sl_mask = find_swing_lows(m15_df, lookback=5)
            tp_candidates = m15_df.loc[sl_mask, "low"]
            tp_candidates = tp_candidates[tp_candidates < entry]
            if tp_candidates.empty:
                return None
            tp = float(tp_candidates.iloc[-1])

        risk = abs(entry - sl)
        reward = abs(tp - entry)
        if risk <= 0:
            return None
        rr = reward / risk
        if rr < _MIN_RR:
            _log.debug("GBPUSD: RR %.2f < %.1f minimum", rr, _MIN_RR)
            return None

        # --- Confluence score ---
        score_fvg = 1.0
        score_session = session_score(pd.Timestamp(current_dt))
        # Judas Swing pattern is high-conviction when all filters align
        score_judas = 1.0
        confluence = round(
            0.40 * score_judas + 0.35 * score_fvg + 0.25 * score_session,
            3,
        )

        size_mult = self.get_size_multiplier(current_dt)

        return TradeSignal(
            symbol=self.symbol,
            direction=entry_direction,
            entry_price=round(entry, 5),
            stop_loss=round(sl, 5),
            take_profit=round(tp, 5),
            rr_ratio=round(rr, 2),
            score_fvg=score_fvg,
            score_session=score_session,
            confluence_score=confluence,
            h1_bias=1 if entry_direction == "bullish" else -1,
            fvg_ref=nearest_fvg,
            bar_time=pd.Timestamp(current_dt),
            entry_type="limit",
            notes=(
                f"size_mult={size_mult:.2f} | judas_extreme={judas_extreme:.5f} | "
                f"expected_dir={expected_direction} | hourly_atr={hourly_atr_proxy:.5f}"
            ),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _calc_atr14(self, df: pd.DataFrame) -> float:
        """Compute ATR(14) on M15 using Wilder's method (simplified)."""
        if len(df) < 15:
            return 0.0
        highs  = df["high"].values
        lows   = df["low"].values
        closes = df["close"].values
        tr_values = []
        for i in range(1, len(df)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i]  - closes[i - 1]),
            )
            tr_values.append(tr)
        # Simple ATR(14) approximation: EWM
        import numpy as np
        tr_series = pd.Series(tr_values)
        atr = float(tr_series.ewm(span=14, adjust=False).mean().iloc[-1])
        return atr

    def _h1_bos_direction(self, h1_df: pd.DataFrame) -> Optional[str]:
        """
        Return the most recent confirmed H1 BOS direction.

        Returns "bullish", "bearish", or None if no recent BOS.
        """
        bos_df = detect_bos(h1_df, lookback=5)
        # Look back 10 H1 bars for a recent BOS
        bullish_recent = bool(bos_df["bos_bullish"].iloc[-10:].any())
        bearish_recent = bool(bos_df["bos_bearish"].iloc[-10:].any())

        # Prefer the most recent one
        if bullish_recent and bearish_recent:
            # Find which occurred last
            last_bull = bos_df["bos_bullish"].iloc[-10:].values[::-1].argmax()
            last_bear = bos_df["bos_bearish"].iloc[-10:].values[::-1].argmax()
            return "bullish" if last_bull < last_bear else "bearish"
        if bullish_recent:
            return "bullish"
        if bearish_recent:
            return "bearish"
        return None

    def _detect_judas_swing(
        self,
        m15_df: pd.DataFrame,
        expected_direction: str,
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Detect a Judas Swing: ≥ 20 pip move against expected_direction.

        Looks at the last 6 M15 bars (covers ~90 min of London open).

        Returns
        -------
        (entry_direction, judas_extreme)
          entry_direction: snap-back direction (opposite of Judas move)
          judas_extreme:   the high or low of the Judas wick
        """
        lookback = min(6, len(m15_df))
        open_price = float(m15_df.iloc[-lookback]["open"])
        recent = m15_df.iloc[-lookback:]

        if expected_direction == "bullish":
            # Judas: sharp downward move
            swing_low = float(recent["low"].min())
            move_size = open_price - swing_low
            if move_size >= _JUDAS_MIN_MOVE:
                return "bullish", swing_low
        else:
            # Judas: sharp upward move
            swing_high = float(recent["high"].max())
            move_size = swing_high - open_price
            if move_size >= _JUDAS_MIN_MOVE:
                return "bearish", swing_high

        return None, None
