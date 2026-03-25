"""
gbpusd.py — GBPUSD Instrument Strategy

Pre-filters:
  1. London open window: 07:00–09:00 UTC only (Judas Swing window).
  2. Volatility gate: M15 ATR(14) × 4 must exceed 40 pips (0.0040).

The Judas Swing pattern (false move at London open → snap-back) is already
captured by ICTStrategy's AMD Power of 3 (swept_lows/swept_highs) and
liquidity sweep detection. CBDR breakout direction confirms the true move.

ICT signal logic handled by ICTStrategy.on_bar().
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from strategy.instruments.base import BaseInstrumentStrategy
from strategy.structure import calculate_atr

_log = logging.getLogger("strategy")

_LONDON_OPEN_START = 7
_LONDON_OPEN_END   = 9
_MIN_HOURLY_ATR    = 0.0040   # ~40 pips proxy


class GBPUSDStrategy(BaseInstrumentStrategy):
    """Judas Swing / London open ICT strategy for GBPUSD.

    Pre-filter: London open window (07–09 UTC) + ATR volatility gate.
    """

    def _pre_filter(self, m15_df, h1_df, current_dt, **kwargs) -> bool:
        # London open window only
        h = current_dt.hour
        if not (_LONDON_OPEN_START <= h < _LONDON_OPEN_END):
            _log.debug("GBPUSD: outside Judas window (hour=%d UTC)", h)
            return False

        # Volatility gate: M15 ATR × 4 as hourly proxy
        atr_series = calculate_atr(m15_df, period=14)
        if len(atr_series) == 0:
            return False
        hourly_atr_proxy = float(atr_series.iloc[-1]) * 4.0
        if hourly_atr_proxy < _MIN_HOURLY_ATR:
            _log.debug(
                "GBPUSD: hourly ATR proxy %.5f < %.5f minimum, skipping",
                hourly_atr_proxy, _MIN_HOURLY_ATR,
            )
            return False

        return True
