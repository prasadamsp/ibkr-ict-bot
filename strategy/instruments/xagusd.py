"""
xagusd.py — XAGUSD (Silver) Instrument Strategy

Strategy logic overview
-----------------------
Silver is a hybrid commodity: it tracks gold (precious metal) but is also
industrially driven.  The Gold-Silver ratio (XAU/XAG) is the primary macro
filter.  Additionally, silver's direction must align with gold's direction to
avoid taking contrary bets within the metals complex.

1. Gold-Silver Ratio (GSR) logic
   The caller (StrategyRouter) passes the current ratio (XAU/XAG price).
   - GSR > 85: silver is historically cheap vs gold → strong LONG bias
   - GSR < 70: silver is historically expensive vs gold → SHORT bias
   - 70–85:    neutral range → both directions permitted

   These thresholds are calibrated to historical mean-reversion levels:
   ratios above 85 have historically reverted to 70–80 over 6–18 months.

2. Gold direction alignment
   The caller also provides `gold_direction` ("bullish", "bearish", or
   "neutral") from the XAUUSDStrategy signal.  The silver trade direction
   must match.  If gold_direction is "neutral", both directions are allowed
   (the GSR bias alone determines direction).

3. Seasonal bias
   Same window as XAUUSD: Sep–Feb bullish, spring weak.  During the bullish
   window, short entries are blocked (mirrors the gold seasonal rule).

4. Entry
   M15 Order Block in the confirmed direction.  Limit order at OB midpoint.

5. Signal signature
   generate_signal() accepts two extra keyword parameters:
     ratio: float = 80.0         current XAU/XAG ratio
     gold_direction: str = "neutral"   direction from XAUUSDStrategy

Risk / Reward
-------------
- Stop: beyond the OB extreme (0.02% buffer).
- Target: nearest M15 swing high (long) or swing low (short).
- Minimum 2R.
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

_GSR_STRONG_LONG  = 85.0   # above this → long bias
_GSR_STRONG_SHORT = 70.0   # below this → short bias
_BULL_MONTHS = {9, 10, 11, 12, 1, 2}
_MIN_RR = 2.0
_SL_BUFFER = 0.0002


class XAGUSDStrategy(BaseInstrumentStrategy):
    """Gold-Silver ratio + ICT OB strategy for XAGUSD (Silver CFD).

    See module docstring for full strategy description.
    """

    def generate_signal(
        self,
        m15_df: pd.DataFrame,
        h1_df: pd.DataFrame,
        current_dt: datetime,
        ratio: float = 80.0,
        gold_direction: str = "neutral",
    ) -> Optional[TradeSignal]:
        """
        Analyse M15/H1 data and return a silver signal, or None.

        Parameters
        ----------
        m15_df : pd.DataFrame
        h1_df  : pd.DataFrame
        current_dt : datetime
        ratio : float
            Current Gold-Silver ratio (XAU/XAG).  Default 80 = neutral.
        gold_direction : str
            XAUUSD signal direction: "bullish", "bearish", or "neutral".
            Silver direction must align with this.
        """
        try:
            return self._generate(m15_df, h1_df, current_dt, ratio, gold_direction)
        except Exception as exc:
            _log.error("XAGUSD generate_signal error: %s", exc, exc_info=True)
            return None

    # ------------------------------------------------------------------

    def _generate(
        self,
        m15_df: pd.DataFrame,
        h1_df: pd.DataFrame,
        current_dt: datetime,
        ratio: float,
        gold_direction: str,
    ) -> Optional[TradeSignal]:

        # --- Minimum bar guard ---
        if len(m15_df) < 50 or len(h1_df) < 30:
            _log.debug("XAGUSD: insufficient bars (m15=%d h1=%d)", len(m15_df), len(h1_df))
            return None

        # --- Session filter ---
        session = get_session(pd.Timestamp(current_dt))
        if session not in {"london_kill", "london", "london_close", "ny_kill", "ny"}:
            _log.debug("XAGUSD: outside tradeable session (%s)", session)
            return None

        # --- GSR bias ---
        gsr_bias = self._gsr_bias(ratio)

        # --- Gold direction alignment ---
        gold_dir = gold_direction.lower().strip()

        # Determine allowed directions
        if gsr_bias == "long":
            gsr_allowed = {"bullish"}
        elif gsr_bias == "short":
            gsr_allowed = {"bearish"}
        else:
            gsr_allowed = {"bullish", "bearish"}

        if gold_dir == "bullish":
            gold_allowed = {"bullish"}
        elif gold_dir == "bearish":
            gold_allowed = {"bearish"}
        else:
            gold_allowed = {"bullish", "bearish"}

        allowed_directions = gsr_allowed & gold_allowed
        if not allowed_directions:
            _log.debug(
                "XAGUSD: GSR bias (%s) conflicts with gold direction (%s) — skipping",
                gsr_bias, gold_direction,
            )
            return None

        # --- Seasonal filter: block shorts in bull months ---
        month = current_dt.month
        if month in _BULL_MONTHS:
            allowed_directions -= {"bearish"}
        if not allowed_directions:
            _log.debug("XAGUSD: seasonal bull window blocks short (month=%d)", month)
            return None

        current_price = float(m15_df.iloc[-1]["close"])

        # --- M15 OB in allowed direction ---
        obs = detect_order_blocks(m15_df, lookback=5, max_age_bars=50)

        signal_candidate = None
        for direction in sorted(allowed_directions):  # deterministic iteration
            nearest_ob = get_nearest_ob(obs, direction, current_price, max_distance_pct=0.005)
            if nearest_ob is None:
                continue

            entry, sl, tp = self._calc_levels(direction, nearest_ob, m15_df)
            if entry is None:
                continue

            risk = abs(entry - sl)
            reward = abs(tp - entry)
            if risk <= 0:
                continue
            rr = reward / risk
            if rr < _MIN_RR:
                continue

            # --- Confluence score ---
            score_ob = 1.0
            score_gsr = 1.0 if gsr_bias in ("long", "short") else 0.5
            score_gold = 1.0 if gold_dir in ("bullish", "bearish") else 0.5
            score_session = session_score(pd.Timestamp(current_dt))
            confluence = round(
                0.35 * score_ob + 0.30 * score_gsr + 0.20 * score_gold + 0.15 * score_session,
                3,
            )

            season_note = self._seasonality.get_note(self.symbol, current_dt)
            size_mult = self.get_size_multiplier(current_dt)

            signal_candidate = TradeSignal(
                symbol=self.symbol,
                direction=direction,
                entry_price=round(entry, 5),
                stop_loss=round(sl, 5),
                take_profit=round(tp, 5),
                rr_ratio=round(rr, 2),
                score_order_block=score_ob,
                score_session=score_session,
                confluence_score=confluence,
                h1_bias=1 if direction == "bullish" else -1,
                ob_ref=nearest_ob,
                bar_time=pd.Timestamp(current_dt),
                entry_type="limit",
                notes=(
                    f"size_mult={size_mult:.2f} | {season_note} | "
                    f"gsr={ratio:.1f} gsr_bias={gsr_bias} | "
                    f"gold_dir={gold_direction} | session={session}"
                ),
            )
            break   # take the first qualifying direction

        if signal_candidate is None:
            _log.debug("XAGUSD: no qualifying OB for directions %s", allowed_directions)

        return signal_candidate

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _gsr_bias(self, ratio: float) -> str:
        """Classify the Gold-Silver ratio into "long", "short", or "neutral"."""
        if ratio > _GSR_STRONG_LONG:
            return "long"
        if ratio < _GSR_STRONG_SHORT:
            return "short"
        return "neutral"

    def _calc_levels(self, direction: str, ob, df: pd.DataFrame):
        """Calculate entry, SL, TP from an OrderBlock."""
        entry = (ob.top + ob.bottom) / 2.0
        if direction == "bullish":
            sl = ob.ob_extreme * (1.0 - _SL_BUFFER)
            sh_mask = find_swing_highs(df, lookback=5)
            candidates = df.loc[sh_mask, "high"]
            above = candidates[candidates > entry]
            if above.empty:
                return None, None, None
            tp = float(above.iloc[0])
        else:
            sl = ob.ob_extreme * (1.0 + _SL_BUFFER)
            sl_mask = find_swing_lows(df, lookback=5)
            candidates = df.loc[sl_mask, "low"]
            below = candidates[candidates < entry]
            if below.empty:
                return None, None, None
            tp = float(below.iloc[-1])
        return entry, sl, tp
