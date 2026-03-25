"""
xauusd.py — XAUUSD (Gold) Instrument Strategy

Pre-filters (gating entry into ICTStrategy):
  1. ADX gate: H1 ADX must be >= 20 (trending market required).
  2. Session: London or NY sessions only.

Post-filters (applied after ICTStrategy generates a signal):
  3. Seasonal direction block: Sep–Feb bull window → no shorts.

All ICT signal logic (BOS/MSS, FVG, OB, Breaker, AMD, CBDR, IPDA,
Daily Bias, Displacement) is handled by ICTStrategy.on_bar().
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from strategy.instruments.base import BaseInstrumentStrategy
from strategy.sessions import get_session
from strategy.strategy import TradeSignal

_log = logging.getLogger("strategy")

_BULL_MONTHS = {9, 10, 11, 12, 1, 2}
_MIN_ADX = 20.0


class XAUUSDStrategy(BaseInstrumentStrategy):
    """ICT strategy for XAUUSD (Gold CFD).

    Pre-filter:  ADX >= 20 on H1, London/NY session only.
    Post-filter: No short signals during seasonal bull window (Sep–Feb).
    """

    def _pre_filter(self, m15_df, h1_df, current_dt, **kwargs) -> bool:
        # ADX gate — require trending market
        regime = self.get_regime(h1_df)
        if regime.adx < _MIN_ADX:
            _log.debug("XAUUSD: ADX %.1f < %.1f — skipping", regime.adx, _MIN_ADX)
            return False

        # Session gate — London or NY only
        session = get_session(pd.Timestamp(current_dt))
        if session not in {"london_kill", "london", "london_close", "ny_kill", "ny"}:
            _log.debug("XAUUSD: outside London/NY session (%s)", session)
            return False

        return True

    def _post_filter(self, signal, m15_df, h1_df, current_dt, **kwargs) -> bool:
        # Seasonal bull window — block shorts
        if current_dt.month in _BULL_MONTHS and signal.direction == "bearish":
            _log.debug("XAUUSD: seasonal bull window, blocking short")
            return False
        return True
