"""
btc.py — BTC Instrument Strategy

Pre-filters:
  1. 24h range gate: H1 range over 24 bars must be >= 2% (avoid tight consolidation).
  2. Dead market guard.

Post-filters:
  3. Halving cycle direction gate:
       bull phase         → long only
       bear phase         → short only
       accumulation phase → prefer longs (shorts require confluence >= 0.70)
       distribution phase → prefer shorts (longs require confluence >= 0.70)
  4. EMA trend alignment: EMA21 vs EMA55 on H1 must agree with signal direction.

ICT signal logic handled by ICTStrategy.on_bar().
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from strategy.btc_cycle import HalvingClock
from strategy.instruments.base import BaseInstrumentStrategy
from strategy.strategy import TradeSignal

_log = logging.getLogger("strategy")

_MIN_24H_RANGE_PCT          = 0.02
_NON_PREFERRED_MIN_CONF     = 0.70


class BTCStrategy(BaseInstrumentStrategy):
    """Halving-cycle + EMA + ICT strategy for BTC."""

    def __init__(self, symbol, strategy_cfg, regime_detector, seasonality):
        super().__init__(symbol, strategy_cfg, regime_detector, seasonality)
        self._halving_clock = HalvingClock()

    def _pre_filter(self, m15_df, h1_df, current_dt, **kwargs) -> bool:
        # 24h range gate
        if len(h1_df) < 24:
            return False
        last_24h = h1_df.iloc[-24:]
        price_range = float(last_24h["high"].max() - last_24h["low"].min())
        current_price = float(h1_df.iloc[-1]["close"])
        if current_price > 0 and (price_range / current_price) < _MIN_24H_RANGE_PCT:
            _log.debug(
                "BTC: 24h range %.2f%% < %.2f%% — dead market, skipping",
                (price_range / current_price) * 100, _MIN_24H_RANGE_PCT * 100,
            )
            return False
        return True

    def _post_filter(self, signal, m15_df, h1_df, current_dt, **kwargs) -> bool:
        # Halving cycle direction gate
        cycle = self._halving_clock.get_phase(current_dt)
        phase = cycle.phase if hasattr(cycle, "phase") else str(cycle)

        direction = signal.direction
        if phase == "bull" and direction == "bearish":
            _log.debug("BTC: halving bull phase — blocking short")
            return False
        if phase == "bear" and direction == "bullish":
            _log.debug("BTC: halving bear phase — blocking long")
            return False
        if phase == "accumulation" and direction == "bearish":
            if signal.confluence_score < _NON_PREFERRED_MIN_CONF:
                _log.debug(
                    "BTC: accumulation — short needs conf >= %.2f, got %.3f",
                    _NON_PREFERRED_MIN_CONF, signal.confluence_score,
                )
                return False
        if phase == "distribution" and direction == "bullish":
            if signal.confluence_score < _NON_PREFERRED_MIN_CONF:
                _log.debug(
                    "BTC: distribution — long needs conf >= %.2f, got %.3f",
                    _NON_PREFERRED_MIN_CONF, signal.confluence_score,
                )
                return False

        # EMA trend alignment: EMA21 vs EMA55 on H1
        h1_closes = h1_df["close"].astype(float)
        ema21 = float(h1_closes.ewm(span=21, adjust=False).mean().iloc[-1])
        ema55 = float(h1_closes.ewm(span=55, adjust=False).mean().iloc[-1])
        ema_bullish = ema21 > ema55
        if ema_bullish and direction == "bearish":
            _log.debug("BTC: EMA21 > EMA55 (bullish trend), blocking short")
            return False
        if not ema_bullish and direction == "bullish":
            _log.debug("BTC: EMA21 < EMA55 (bearish trend), blocking long")
            return False

        return True
