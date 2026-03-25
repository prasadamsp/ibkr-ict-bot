"""
oil.py — OIL (Crude Oil) Instrument Strategy

Two modes:

EIA Mode (Wednesday 14:30–16:00 UTC):
  News-momentum trade — market order in the direction of the last 2 M15
  closes after the report. SL/TP based on ATR multiples.
  Bypasses ICTStrategy (event-driven, not pattern-driven).

Normal Day Mode:
  Pre-filter:  ATR% >= 0.8% on H1 (avoid dead oil sessions).
  Post-filter: Seasonal direction bias (Feb–Aug bull window).
  ICT signal logic handled by ICTStrategy.on_bar().
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from strategy.instruments.base import BaseInstrumentStrategy
from strategy.strategy import TradeSignal

_log = logging.getLogger("strategy")

_MIN_ATR_PCT    = 0.008
_EIA_HOUR_START = 14
_EIA_MIN_START  = 30
_EIA_HOUR_END   = 16
_EIA_BASE_CONF  = 0.70
_MIN_RR         = 2.0
_BULL_MONTHS    = {2, 3, 4, 5, 6, 7, 8}   # Feb–Aug bullish seasonal window


class OILStrategy(BaseInstrumentStrategy):
    """EIA event + ICT strategy for WTI Crude Oil."""

    def generate_signal(
        self,
        m15_df: pd.DataFrame,
        h1_df: pd.DataFrame,
        current_dt: datetime,
        d1_df: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> Optional[TradeSignal]:
        try:
            if len(m15_df) < 50 or len(h1_df) < 30:
                return None

            # EIA Wednesday mode — bypass ICTStrategy
            if current_dt.weekday() == 2 and self._in_eia_window(current_dt):
                regime = self.get_regime(h1_df)
                if regime.atr_pct < _MIN_ATR_PCT:
                    return None
                return self._eia_signal(m15_df, regime, current_dt)

            # Normal day — use base class (pre_filter → ICTStrategy → post_filter)
            return super().generate_signal(m15_df, h1_df, current_dt, d1_df, **kwargs)

        except Exception as exc:
            _log.error("OIL generate_signal error: %s", exc, exc_info=True)
            return None

    def _pre_filter(self, m15_df, h1_df, current_dt, **kwargs) -> bool:
        # ATR volatility gate
        regime = self.get_regime(h1_df)
        if regime.atr_pct < _MIN_ATR_PCT:
            _log.debug("OIL: ATR%% %.4f < %.4f minimum, skipping", regime.atr_pct, _MIN_ATR_PCT)
            return False
        return True

    def _post_filter(self, signal, m15_df, h1_df, current_dt, **kwargs) -> bool:
        # Seasonal bear window (Sep–Jan): block longs
        if current_dt.month not in _BULL_MONTHS and signal.direction == "bullish":
            _log.debug("OIL: outside bull seasonal window, blocking long")
            return False
        return True

    # ------------------------------------------------------------------
    # EIA momentum signal (news-driven, market order)
    # ------------------------------------------------------------------

    def _in_eia_window(self, dt: datetime) -> bool:
        h, m = dt.hour, dt.minute
        after_start = (h > _EIA_HOUR_START) or (h == _EIA_HOUR_START and m >= _EIA_MIN_START)
        before_end  = h < _EIA_HOUR_END
        return after_start and before_end

    def _eia_signal(self, m15_df: pd.DataFrame, regime, current_dt: datetime) -> Optional[TradeSignal]:
        if len(m15_df) < 3:
            return None

        c_curr = float(m15_df.iloc[-1]["close"])
        c_prev = float(m15_df.iloc[-2]["close"])
        direction = "bullish" if c_curr > c_prev else "bearish"

        atr = regime.atr if regime.atr > 0 else c_curr * regime.atr_pct
        if direction == "bullish":
            sl = c_curr - 2.0 * atr
            tp = c_curr + 3.0 * atr
        else:
            sl = c_curr + 2.0 * atr
            tp = c_curr - 3.0 * atr

        risk = abs(c_curr - sl)
        reward = abs(tp - c_curr)
        if risk <= 0:
            return None
        rr = reward / risk
        if rr < _MIN_RR:
            return None

        size_mult = self.get_size_multiplier(current_dt)
        season_note = self._seasonality.get_note(self.symbol, current_dt)

        return TradeSignal(
            symbol=self.symbol,
            direction=direction,
            entry_price=round(c_curr, 3),
            stop_loss=round(sl, 3),
            take_profit=round(tp, 3),
            rr_ratio=round(rr, 2),
            confluence_score=round(min(_EIA_BASE_CONF * min(size_mult, 1.5), 1.0), 3),
            h1_bias=1 if direction == "bullish" else -1,
            bar_time=pd.Timestamp(current_dt),
            entry_type="market",
            notes=f"EIA_momentum | size_mult={size_mult:.2f} | {season_note}",
        )
