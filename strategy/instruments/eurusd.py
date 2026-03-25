"""
eurusd.py — EURUSD Instrument Strategy

Pre-filters:
  1. Kill zone only: London (07:00–10:00 UTC) or NY (12:00–15:00 UTC).
  2. Asian range check: require a valid Asian session range to be defined.

Post-filters:
  3. Seasonal direction block: Q1 (Jan–Mar) → short bias only.
  4. Low-vol months (Jul–Sep): require higher confluence (>= 0.60).

ICT signal logic (BOS/MSS, FVG, OB, Breaker, AMD, CBDR, IPDA, liquidity
sweep, etc.) is handled by ICTStrategy.on_bar().
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from strategy.instruments.base import BaseInstrumentStrategy
from strategy.sessions import get_asian_range, get_session
from strategy.strategy import TradeSignal

_log = logging.getLogger("strategy")

_LONDON_KZ_START = 7
_LONDON_KZ_END   = 10
_NY_KZ_START     = 12
_NY_KZ_END       = 15

_SHORT_BIAS_MONTHS   = {1, 2, 3}
_LOW_VOL_MONTHS      = {7, 8, 9}
_LOW_VOL_MIN_CONF    = 0.60


class EURUSDStrategy(BaseInstrumentStrategy):
    """Kill-zone ICT strategy for EURUSD.

    Pre-filter:  Kill zone + Asian range presence.
    Post-filter: Q1 short-bias; summer low-vol confluence gate.
    """

    def _pre_filter(self, m15_df, h1_df, current_dt, **kwargs) -> bool:
        # Kill zone only
        h = current_dt.hour
        in_kz = (
            (_LONDON_KZ_START <= h < _LONDON_KZ_END) or
            (_NY_KZ_START <= h < _NY_KZ_END)
        )
        if not in_kz:
            _log.debug("EURUSD: outside kill zone (hour=%d UTC)", h)
            return False

        # Require a valid Asian range
        if get_asian_range(m15_df) is None:
            _log.debug("EURUSD: no Asian range available")
            return False

        return True

    def _post_filter(self, signal, m15_df, h1_df, current_dt, **kwargs) -> bool:
        month = current_dt.month

        # Q1 short-bias: block longs
        if month in _SHORT_BIAS_MONTHS and signal.direction == "bullish":
            _log.debug("EURUSD: Q1 short-bias, blocking long")
            return False

        # Summer low-vol: require higher confluence
        if month in _LOW_VOL_MONTHS and signal.confluence_score < _LOW_VOL_MIN_CONF:
            _log.debug(
                "EURUSD: summer low-vol confluence %.3f < %.3f",
                signal.confluence_score, _LOW_VOL_MIN_CONF,
            )
            return False

        return True
