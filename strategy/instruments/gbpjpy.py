"""
gbpjpy.py — GBPJPY Instrument Strategy

Strategy: ICT — Tokyo+London overlap session (07:00–09:00 UTC)

GBPJPY is the highest-volatility major cross. GBP volatility combined with
JPY safe-haven flows creates pronounced liquidity sweeps and false breakouts
at key levels — the exact microstructure that ICT exploits best.

Documented edge: Sharpe 0.8–1.1 with ICT methodology (external research).
Own backtest required to confirm — 0.3x sizing until 6-month live validation.

Pre-filters:
  1. Session: Tokyo+London overlap 07:00–09:00 UTC only.
     Rationale: GBPJPY best liquidity when Tokyo still active (JPY cross
     requires Asian session context for valid liquidity levels).
  2. Volatility gate: M15 ATR(14) × 4 must exceed 50 pips (0.0050).
     GBPJPY moves more than GBPUSD; calibrated to JPY pip values.

Post-filters:
  3. Confluence gate: minimum 0.65 (cross pairs need higher bar than majors).
  4. Seasonal gate: Jan–Jun only (JPY carry trade correlated with risk-on;
     Jul–Dec JPY safe-haven demand can override GBP direction).

ICT signal logic handled by ICTStrategy.on_bar() via BaseInstrumentStrategy.
Sizing: 0.3x normal position size until 6-month live validation passes.
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from strategy.instruments.base import BaseInstrumentStrategy
from strategy.structure import calculate_atr

_log = logging.getLogger("strategy")

_SESSION_START   = 7
_SESSION_END     = 9
_MIN_HOURLY_ATR  = 0.0050    # ~50 JPY pips (calibrated to GBPJPY scale)
_MIN_CONFLUENCE  = 0.65
_ACTIVE_MONTHS   = {1, 2, 3, 4, 5, 6}   # Jan–Jun only (carry trade season)


class GBPJPYStrategy(BaseInstrumentStrategy):
    """
    ICT strategy for GBPJPY — Tokyo+London overlap session.

    Pre-filters:  07–09 UTC window + volatility gate + Jan–Jun seasonal gate.
    Post-filters: Confluence ≥ 0.65.
    Sizing:       0.3x (pending 6-month live validation).
    """

    def _pre_filter(self, m15_df, h1_df, current_dt, **kwargs) -> bool:
        # Seasonal gate: Jan–Jun only
        if current_dt.month not in _ACTIVE_MONTHS:
            _log.debug("GBPJPY: outside active season (month=%d)", current_dt.month)
            return False

        # Session gate: Tokyo+London overlap only
        h = current_dt.hour
        if not (_SESSION_START <= h < _SESSION_END):
            _log.debug("GBPJPY: outside session window (hour=%d UTC)", h)
            return False

        # Volatility gate
        atr_series = calculate_atr(m15_df, period=14)
        if len(atr_series) == 0:
            return False
        hourly_atr_proxy = float(atr_series.iloc[-1]) * 4.0
        if hourly_atr_proxy < _MIN_HOURLY_ATR:
            _log.debug(
                "GBPJPY: hourly ATR proxy %.4f < %.4f minimum, skipping",
                hourly_atr_proxy, _MIN_HOURLY_ATR,
            )
            return False

        return True

    def _post_filter(self, signal, m15_df, h1_df, current_dt, **kwargs) -> bool:
        # Confluence gate — cross pairs need higher quality bar
        if signal.confluence_score < _MIN_CONFLUENCE:
            _log.debug(
                "GBPJPY: confluence %.3f < %.3f — skipping",
                signal.confluence_score, _MIN_CONFLUENCE,
            )
            return False

        return True
