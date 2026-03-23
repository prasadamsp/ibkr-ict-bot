"""
btc.py — BTC Instrument Strategy

Strategy logic overview
-----------------------
Bitcoin's price action is heavily influenced by its 4-year halving cycle.
This module uses HalvingClock to set a macro directional bias, then overlays
EMA-trend and ICT entry methods to time entries within that macro context.

1. Halving Cycle Gate
   HalvingClock determines the current cycle phase and direction bias:
     "bull"         → long_only      (only long entries permitted)
     "accumulation" → long_preferred (longs preferred; shorts require strong
                                       bearish OB — confluence >= 0.70)
     "distribution" → short_preferred (inverse of accumulation)
     "bear"         → short_only     (only short entries permitted)

2. EMA Trend (H1)
   EMA(21) vs EMA(55) on H1 gives a medium-term trend signal:
     EMA21 > EMA55 → long bias
     EMA21 < EMA55 → short bias
   The trade direction must align with BOTH the halving cycle bias AND the
   EMA trend (unless the cycle is "accumulation" or "distribution", where the
   non-preferred direction is allowed with higher confluence).

3. Dead Market Guard
   If the 24-hour price range (H1 bars over last 24 bars) is less than 2% of
   the current price, BTC is consolidating tightly.  Skip — there is not
   enough volatility for reliable ICT entries.

4. Entry: M15 OB or FVG
   After all filters pass, look for an Order Block or Fair Value Gap on M15
   in the confirmed direction.  Enter at the zone midpoint.

5. Size from HalvingClock
   HalvingClock.get_size_multiplier() returns the cycle-phase size multiplier
   (0.3–1.5).  This is stored in the signal notes for the router.

Risk / Reward
-------------
- Stop: below the OB extreme / FVG boundary.
- Target: 3R minimum (BTC's large moves justify wider targets).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from strategy.btc_cycle import HalvingClock
from strategy.fvg import detect_fvgs, get_nearest_fvg
from strategy.instruments.base import BaseInstrumentStrategy
from strategy.order_blocks import detect_order_blocks, get_nearest_ob
from strategy.sessions import session_score
from strategy.strategy import TradeSignal
from strategy.structure import find_swing_highs, find_swing_lows

_log = logging.getLogger("strategy")

_MIN_24H_RANGE_PCT = 0.02    # 24h range must be > 2% of price
_MIN_RR = 3.0                # BTC targets 3R minimum
_SL_BUFFER = 0.0003          # 0.03% buffer (BTC spreads are wider)

# Confluence thresholds per phase for non-preferred direction
_NON_PREFERRED_MIN_CONFLUENCE = 0.70


class BTCStrategy(BaseInstrumentStrategy):
    """Halving-cycle + EMA + ICT entry strategy for BTC.

    See module docstring for full strategy description.
    """

    def __init__(self, symbol, strategy_cfg, regime_detector, seasonality):
        super().__init__(symbol, strategy_cfg, regime_detector, seasonality)
        self._halving_clock = HalvingClock()

    def generate_signal(
        self,
        m15_df: pd.DataFrame,
        h1_df: pd.DataFrame,
        current_dt: datetime,
    ) -> Optional[TradeSignal]:
        try:
            return self._generate(m15_df, h1_df, current_dt)
        except Exception as exc:
            _log.error("BTC generate_signal error: %s", exc, exc_info=True)
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
            _log.debug("BTC: insufficient bars (m15=%d h1=%d)", len(m15_df), len(h1_df))
            return None

        # --- Dead market guard: 24h range < 2% ---
        if not self._has_sufficient_volatility(h1_df):
            _log.debug("BTC: 24h range < %.1f%% — dead market, skipping", _MIN_24H_RANGE_PCT * 100)
            return None

        # --- Halving cycle phase and direction bias ---
        phase = self._halving_clock.get_phase(current_dt)
        bias  = self._halving_clock.get_direction_bias(current_dt)
        size_mult = self._halving_clock.get_size_multiplier(current_dt)

        # --- EMA(21) vs EMA(55) trend on H1 ---
        h1_closes = h1_df["close"].astype(float)
        ema21 = float(h1_closes.ewm(span=21, adjust=False).mean().iloc[-1])
        ema55 = float(h1_closes.ewm(span=55, adjust=False).mean().iloc[-1])
        ema_bullish = ema21 > ema55 * 1.0002
        ema_bearish = ema21 < ema55 * 0.9998

        # --- Resolve trade direction ---
        direction = self._resolve_direction(bias, ema_bullish, ema_bearish)
        if direction is None:
            _log.debug(
                "BTC: no qualifying direction (phase=%s bias=%s ema21=%.2f ema55=%.2f)",
                phase, bias, ema21, ema55,
            )
            return None

        current_price = float(m15_df.iloc[-1]["close"])

        # --- M15 OB or FVG ---
        obs = detect_order_blocks(m15_df, lookback=5, max_age_bars=50)
        nearest_ob = get_nearest_ob(obs, direction, current_price, max_distance_pct=0.006)

        fvgs = detect_fvgs(m15_df, min_size_pct=0.0003, max_age_bars=50)
        nearest_fvg = get_nearest_fvg(fvgs, direction, current_price, max_distance_pct=0.006)

        if nearest_ob is None and nearest_fvg is None:
            _log.debug("BTC: no OB or FVG in range")
            return None

        # --- Entry / SL / TP levels ---
        entry, sl, tp = self._calc_levels(
            direction, current_price, nearest_ob, nearest_fvg, m15_df
        )
        if entry is None:
            return None

        risk = abs(entry - sl)
        reward = abs(tp - entry)
        if risk <= 0:
            return None
        rr = reward / risk
        if rr < _MIN_RR:
            _log.debug("BTC: RR %.2f < %.1f minimum", rr, _MIN_RR)
            return None

        # --- Confluence score ---
        score_ob  = 1.0 if nearest_ob  else 0.0
        score_fvg = 1.0 if nearest_fvg else 0.0
        score_cycle = self._phase_score(phase, direction)
        score_ema = 1.0 if (
            (direction == "bullish" and ema_bullish) or
            (direction == "bearish" and ema_bearish)
        ) else 0.5
        confluence = round(
            0.30 * score_cycle + 0.25 * score_ob + 0.25 * score_fvg + 0.20 * score_ema,
            3,
        )

        # For non-preferred direction, require higher confluence
        is_preferred = self._is_preferred_direction(bias, direction)
        if not is_preferred and confluence < _NON_PREFERRED_MIN_CONFLUENCE:
            _log.debug(
                "BTC: non-preferred direction; confluence %.3f < %.2f threshold",
                confluence, _NON_PREFERRED_MIN_CONFLUENCE,
            )
            return None

        return TradeSignal(
            symbol=self.symbol,
            direction=direction,
            entry_price=round(entry, 2),
            stop_loss=round(sl, 2),
            take_profit=round(tp, 2),
            rr_ratio=round(rr, 2),
            score_order_block=score_ob,
            score_fvg=score_fvg,
            confluence_score=confluence,
            h1_bias=1 if direction == "bullish" else -1,
            fvg_ref=nearest_fvg,
            ob_ref=nearest_ob,
            bar_time=pd.Timestamp(current_dt),
            entry_type="limit",
            notes=(
                f"size_mult={size_mult:.2f} | phase={phase} | bias={bias} | "
                f"ema21={ema21:.2f} ema55={ema55:.2f}"
            ),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _has_sufficient_volatility(self, h1_df: pd.DataFrame) -> bool:
        """Return True if 24h price range > 2% of current price."""
        bars_24h = min(24, len(h1_df))
        last24 = h1_df.iloc[-bars_24h:]
        high24 = float(last24["high"].max())
        low24  = float(last24["low"].min())
        last_close = float(h1_df.iloc[-1]["close"])
        if last_close <= 0:
            return False
        range_pct = (high24 - low24) / last_close
        return range_pct >= _MIN_24H_RANGE_PCT

    def _resolve_direction(
        self,
        bias: str,
        ema_bullish: bool,
        ema_bearish: bool,
    ) -> Optional[str]:
        """
        Determine the trade direction from cycle bias and EMA trend.

        long_only / short_only: both filters must agree.
        long_preferred / short_preferred: preferred direction if EMA agrees;
        opposite direction allowed if EMA agrees but checked at higher confluence.
        """
        if bias == "long_only":
            return "bullish" if ema_bullish else None
        if bias == "short_only":
            return "bearish" if ema_bearish else None
        if bias == "long_preferred":
            if ema_bullish:
                return "bullish"
            if ema_bearish:
                return "bearish"   # non-preferred, higher confluence required later
        if bias == "short_preferred":
            if ema_bearish:
                return "bearish"
            if ema_bullish:
                return "bullish"   # non-preferred
        return None

    def _is_preferred_direction(self, bias: str, direction: str) -> bool:
        """Return True if direction matches the cycle bias preference."""
        if bias in ("long_only", "long_preferred") and direction == "bullish":
            return True
        if bias in ("short_only", "short_preferred") and direction == "bearish":
            return True
        return False

    def _phase_score(self, phase: str, direction: str) -> float:
        """Score how well the direction aligns with the current cycle phase."""
        alignment = {
            ("bull",         "bullish"):  1.0,
            ("accumulation", "bullish"):  0.8,
            ("distribution", "bearish"):  0.8,
            ("bear",         "bearish"):  1.0,
            ("accumulation", "bearish"):  0.4,
            ("distribution", "bullish"):  0.4,
            ("bull",         "bearish"):  0.2,
            ("bear",         "bullish"):  0.2,
        }
        return alignment.get((phase, direction), 0.5)

    def _calc_levels(
        self,
        direction: str,
        current_price: float,
        ob,
        fvg,
        df: pd.DataFrame,
    ):
        """Calculate entry, stop-loss, and take-profit."""
        if ob is not None:
            entry = (ob.top + ob.bottom) / 2.0
            if direction == "bullish":
                sl = ob.ob_extreme * (1.0 - _SL_BUFFER)
            else:
                sl = ob.ob_extreme * (1.0 + _SL_BUFFER)
        else:
            entry = (fvg.top + fvg.bottom) / 2.0
            if direction == "bullish":
                sl = fvg.bottom * (1.0 - _SL_BUFFER)
            else:
                sl = fvg.top * (1.0 + _SL_BUFFER)

        risk = abs(entry - sl)
        if risk <= 0:
            return None, None, None

        # Target at minimum 3R
        if direction == "bullish":
            sh_mask = find_swing_highs(df, lookback=5)
            candidates = df.loc[sh_mask, "high"]
            above = candidates[candidates > entry + risk * _MIN_RR]
            if not above.empty:
                tp = float(above.iloc[0])
            else:
                tp = entry + risk * _MIN_RR
        else:
            sl_mask = find_swing_lows(df, lookback=5)
            candidates = df.loc[sl_mask, "low"]
            below = candidates[candidates < entry - risk * _MIN_RR]
            if not below.empty:
                tp = float(below.iloc[-1])
            else:
                tp = entry - risk * _MIN_RR

        return entry, sl, tp
