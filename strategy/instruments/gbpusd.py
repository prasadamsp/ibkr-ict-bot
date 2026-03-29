"""
gbpusd.py — GBPUSD Instrument Strategy

Pre-filters:
  1. London open window: 07:00–09:00 UTC only (Judas Swing window).
  2. Volatility gate: M15 ATR(14) × 4 must exceed 40 pips (0.0040).
  3. Seasonal gate: Jan 1 – Jun 30 only. No new entries Jul–Dec.
     Evidence: validation Sharpe 22.6 vs test Sharpe -32.3 = overfitting.
     Jul–Sep vol collapse + seasonal regime change causes strategy to fail.
     Any open positions on Jun 30 EOD should be closed by execution engine.

Post-filters:
  4. Confluence gate: minimum 0.70 (raised from 0.55).
     Rationale: GBPUSD false breakouts are common; only high-quality
     setups are worth the risk. Lower confluence trades were the primary
     cause of the negative test return.

The Judas Swing pattern (false move at London open → snap-back) is already
captured by ICTStrategy's AMD Power of 3 (swept_lows/swept_highs) and
liquidity sweep detection. CBDR breakout direction confirms the true move.

ICT signal logic handled by ICTStrategy.on_bar() via BaseInstrumentStrategy.
Position sizing: 0.3x normal size (risk = 0.15% per trade) due to overfitting
uncertainty until 12 months of live validation confirms the fix.
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from strategy.instruments.base import BaseInstrumentStrategy
from strategy.structure import calculate_atr

_log = logging.getLogger("strategy")

_LONDON_OPEN_START  = 7
_LONDON_OPEN_END    = 9
_MIN_HOURLY_ATR     = 0.0040   # ~40 pips proxy
_MIN_CONFLUENCE     = 0.70     # raised from 0.55 — only high-quality setups
_ACTIVE_MONTHS      = {1, 2, 3, 4, 5, 6}   # Jan–Jun only


class GBPUSDStrategy(BaseInstrumentStrategy):
    """
    Judas Swing / London open ICT strategy for GBPUSD.

    Pre-filters:  London open window (07–09 UTC) + ATR volatility gate
                  + Jan–Jun seasonal gate.
    Post-filters: Confluence ≥ 0.70.
    Sizing:       0.3x (via seasonality multiplier override in notes).
    """

    def _pre_filter(self, m15_df, h1_df, current_dt, **kwargs) -> bool:
        # Seasonal gate: Jan–Jun only
        if current_dt.month not in _ACTIVE_MONTHS:
            _log.debug("GBPUSD: outside active season (month=%d)", current_dt.month)
            return False

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

    def _post_filter(self, signal, m15_df, h1_df, current_dt, **kwargs) -> bool:
        # Raised confluence threshold — only high-quality setups
        if signal.confluence_score < _MIN_CONFLUENCE:
            _log.debug(
                "GBPUSD: confluence %.3f < %.3f — skipping low-quality setup",
                signal.confluence_score, _MIN_CONFLUENCE,
            )
            return False

        return True
