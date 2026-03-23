"""
oil.py — OIL (Crude Oil) Instrument Strategy

Strategy logic overview
-----------------------
Crude oil combines two distinct modes depending on whether it is an EIA
report day (Wednesday) or a normal trading day.

EIA Mode (every Wednesday, 14:30–16:00 UTC)
--------------------------------------------
The U.S. EIA weekly petroleum inventory report is the single most impactful
scheduled event for crude oil.  When the report is imminent or just released:

  1. Only activate between 14:30 and 16:00 UTC.
  2. Enter in the MOMENTUM direction — whichever direction the most recent
     M15 bar closed harder (compare close vs open of the 2nd bar after 14:30).
  3. Entry type is "market" (no waiting for a pullback; news momentum).
  4. Base confluence = 0.70 (the event itself is the edge).
  5. Volatility gate still applies.

Normal Day Mode
---------------
1. H4 bias (approximated from last 4 H1 bars, same method as XAUUSD).
2. M15 Order Block entry in H4 trend direction.
3. Session: London or NY only.

Shared filters (both modes)
----------------------------
- Volatility gate: ATR% on H1 must be ≥ 0.8%.  Below this the oil market is
  too quiet (typically a holiday or OPEC decision lull).
- Seasonal multiplier: Feb–Aug is the bullish seasonal window for oil
  (refinery maintenance → tight supply, then driving season).
  The multiplier is passed through in notes for the router.

Risk / Reward
-------------
- EIA mode: SL = 2× ATR below/above entry; TP = 3× ATR.
- Normal mode: SL = beyond OB extreme; TP = nearest swing high/low.
- Minimum 2R (both modes).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from strategy.instruments.base import BaseInstrumentStrategy
from strategy.order_blocks import detect_order_blocks, get_nearest_ob
from strategy.sessions import get_session, session_score
from strategy.strategy import TradeSignal
from strategy.structure import find_swing_highs, find_swing_lows

_log = logging.getLogger("strategy")

_MIN_ATR_PCT     = 0.008     # H1 ATR% minimum (0.8%)
_EIA_HOUR_START  = 14        # 14:30 UTC
_EIA_MIN_START   = 30
_EIA_HOUR_END    = 16        # window closes at 16:00 UTC
_EIA_BASE_CONF   = 0.70
_MIN_RR          = 2.0
_SL_BUFFER       = 0.0002


class OILStrategy(BaseInstrumentStrategy):
    """EIA event + H4 trend strategy for WTI Crude Oil (OIL CFD).

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
            _log.error("OIL generate_signal error: %s", exc, exc_info=True)
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
            _log.debug("OIL: insufficient bars (m15=%d h1=%d)", len(m15_df), len(h1_df))
            return None

        # --- Volatility gate ---
        regime = self.get_regime(h1_df)
        if regime.atr_pct < _MIN_ATR_PCT:
            _log.debug(
                "OIL: ATR%% %.4f < %.4f minimum, skipping",
                regime.atr_pct, _MIN_ATR_PCT,
            )
            return None

        # --- Seasonal multiplier ---
        size_mult = self.get_size_multiplier(current_dt)
        season_note = self._seasonality.get_note(self.symbol, current_dt)

        # --- EIA Wednesday mode ---
        is_wednesday = current_dt.weekday() == 2   # 0=Mon, 2=Wed
        in_eia_window = (
            is_wednesday and
            self._in_eia_window(current_dt)
        )

        if in_eia_window:
            return self._eia_signal(m15_df, h1_df, current_dt, size_mult, season_note, regime)
        else:
            return self._normal_signal(m15_df, h1_df, current_dt, size_mult, season_note)

    # ------------------------------------------------------------------
    # EIA mode
    # ------------------------------------------------------------------

    def _eia_signal(
        self,
        m15_df: pd.DataFrame,
        h1_df: pd.DataFrame,
        current_dt: datetime,
        size_mult: float,
        season_note: str,
        regime,
    ) -> Optional[TradeSignal]:
        """Enter market-direction momentum 2 M15 bars after 14:30 UTC."""
        # Determine momentum direction from the 2 most recent M15 closes
        if len(m15_df) < 3:
            return None

        c_curr = float(m15_df.iloc[-1]["close"])
        c_prev = float(m15_df.iloc[-2]["close"])

        direction = "bullish" if c_curr > c_prev else "bearish"

        # SL and TP based on ATR
        atr = regime.atr if regime.atr > 0 else c_curr * regime.atr_pct
        current_price = c_curr

        if direction == "bullish":
            sl = current_price - 2.0 * atr
            tp = current_price + 3.0 * atr
        else:
            sl = current_price + 2.0 * atr
            tp = current_price - 3.0 * atr

        risk = abs(current_price - sl)
        reward = abs(tp - current_price)
        if risk <= 0:
            return None
        rr = reward / risk
        if rr < _MIN_RR:
            return None

        confluence = _EIA_BASE_CONF * min(size_mult, 1.5)
        confluence = round(min(confluence, 1.0), 3)

        return TradeSignal(
            symbol=self.symbol,
            direction=direction,
            entry_price=round(current_price, 3),
            stop_loss=round(sl, 3),
            take_profit=round(tp, 3),
            rr_ratio=round(rr, 2),
            confluence_score=confluence,
            h1_bias=1 if direction == "bullish" else -1,
            bar_time=pd.Timestamp(current_dt),
            entry_type="market",
            notes=(
                f"size_mult={size_mult:.2f} | {season_note} | "
                f"mode=EIA_wednesday | atr={atr:.3f}"
            ),
        )

    # ------------------------------------------------------------------
    # Normal mode
    # ------------------------------------------------------------------

    def _normal_signal(
        self,
        m15_df: pd.DataFrame,
        h1_df: pd.DataFrame,
        current_dt: datetime,
        size_mult: float,
        season_note: str,
    ) -> Optional[TradeSignal]:
        """H4 trend + M15 OB entry for non-EIA days."""
        # Session filter: London or NY
        session = get_session(pd.Timestamp(current_dt))
        if session not in {"london_kill", "london", "london_close", "ny_kill", "ny"}:
            _log.debug("OIL: outside tradeable session (%s)", session)
            return None

        # H4 bias from last 4 H1 bars
        h4_bias = self._h4_bias(h1_df)
        if h4_bias == 0:
            _log.debug("OIL: H4 bias neutral, skipping")
            return None
        direction = "bullish" if h4_bias == 1 else "bearish"

        current_price = float(m15_df.iloc[-1]["close"])

        # M15 OB in H4 direction
        obs = detect_order_blocks(m15_df, lookback=5, max_age_bars=50)
        nearest_ob = get_nearest_ob(obs, direction, current_price, max_distance_pct=0.006)
        if nearest_ob is None:
            _log.debug("OIL: no qualifying OB")
            return None

        entry = (nearest_ob.top + nearest_ob.bottom) / 2.0
        if direction == "bullish":
            sl = nearest_ob.ob_extreme * (1.0 - _SL_BUFFER)
            sh_mask = find_swing_highs(m15_df, lookback=5)
            candidates = m15_df.loc[sh_mask, "high"]
            above = candidates[candidates > entry]
            if above.empty:
                return None
            tp = float(above.iloc[0])
        else:
            sl = nearest_ob.ob_extreme * (1.0 + _SL_BUFFER)
            sl_mask = find_swing_lows(m15_df, lookback=5)
            candidates = m15_df.loc[sl_mask, "low"]
            below = candidates[candidates < entry]
            if below.empty:
                return None
            tp = float(below.iloc[-1])

        risk = abs(entry - sl)
        reward = abs(tp - entry)
        if risk <= 0:
            return None
        rr = reward / risk
        if rr < _MIN_RR:
            _log.debug("OIL: RR %.2f < %.1f minimum", rr, _MIN_RR)
            return None

        score_ob = 1.0
        score_session = session_score(pd.Timestamp(current_dt))
        confluence = round(0.50 * score_ob + 0.30 * score_session + 0.20 * (size_mult / 1.5), 3)

        return TradeSignal(
            symbol=self.symbol,
            direction=direction,
            entry_price=round(entry, 3),
            stop_loss=round(sl, 3),
            take_profit=round(tp, 3),
            rr_ratio=round(rr, 2),
            score_order_block=score_ob,
            score_session=score_session,
            confluence_score=confluence,
            h1_bias=h4_bias,
            ob_ref=nearest_ob,
            bar_time=pd.Timestamp(current_dt),
            entry_type="limit",
            notes=(
                f"size_mult={size_mult:.2f} | {season_note} | "
                f"mode=normal | h4_bias={'bull' if h4_bias == 1 else 'bear'} | "
                f"session={session}"
            ),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _in_eia_window(self, dt: datetime) -> bool:
        """Return True if dt is between 14:30 and 16:00 UTC."""
        total_min = dt.hour * 60 + dt.minute
        start_min = _EIA_HOUR_START * 60 + _EIA_MIN_START
        end_min   = _EIA_HOUR_END * 60
        return start_min <= total_min < end_min

    def _h4_bias(self, h1_df: pd.DataFrame) -> int:
        """
        Approximate H4 trend from last 4 closed H1 bars.

        Returns +1 (bullish), -1 (bearish), 0 (neutral).
        """
        if len(h1_df) < 4:
            return 0
        last4 = h1_df.iloc[-4:]
        mean_close = float(last4["close"].mean())
        last_close = float(h1_df.iloc[-1]["close"])
        if last_close > mean_close * 1.0005:
            return 1
        if last_close < mean_close * 0.9995:
            return -1
        return 0
