"""
xagusd.py — XAGUSD (Silver) Instrument Strategy

Pre-filters:
  1. GSR (Gold-Silver Ratio) gate:
       ratio > 85 → long bias (silver cheap vs gold)
       ratio < 70 → short bias (silver expensive vs gold)
       70–85      → neutral, both directions allowed
  2. Gold direction alignment: signal direction must match gold's direction
     (or gold must be neutral).

Post-filters:
  3. Seasonal direction block: Sep–Feb bull window → no shorts.

generate_signal() accepts extra kwargs: ratio=80.0, gold_direction="neutral"
These are passed from StrategyRouter which reads XAU/XAG prices.

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

_GSR_STRONG_LONG  = 85.0
_GSR_STRONG_SHORT = 70.0
_BULL_MONTHS = {9, 10, 11, 12, 1, 2}


class XAGUSDStrategy(BaseInstrumentStrategy):
    """Gold-Silver ratio + ICT strategy for XAGUSD (Silver CFD)."""

    def _pre_filter(self, m15_df, h1_df, current_dt,
                    ratio: float = 80.0,
                    gold_direction: str = "neutral",
                    **kwargs) -> bool:
        # GSR bias check — determine allowed directions
        if ratio > _GSR_STRONG_LONG:
            allowed = {"bullish"}
        elif ratio < _GSR_STRONG_SHORT:
            allowed = {"bearish"}
        else:
            allowed = {"bullish", "bearish"}   # neutral range

        # Gold direction alignment
        if gold_direction == "bullish":
            allowed = allowed & {"bullish"}
        elif gold_direction == "bearish":
            allowed = allowed & {"bearish"}
        # neutral → no restriction

        if not allowed:
            _log.debug(
                "XAGUSD: GSR=%.1f gold_dir=%s — no valid directions",
                ratio, gold_direction,
            )
            return False

        # Store allowed directions for post-filter use
        self._allowed_directions = allowed
        _log.debug(
            "XAGUSD: GSR=%.1f gold_dir=%s allowed=%s",
            ratio, gold_direction, allowed,
        )
        return True

    def _post_filter(self, signal, m15_df, h1_df, current_dt,
                     ratio: float = 80.0,
                     gold_direction: str = "neutral",
                     **kwargs) -> bool:
        # GSR / gold direction restriction
        allowed = getattr(self, "_allowed_directions", {"bullish", "bearish"})
        if signal.direction not in allowed:
            _log.debug(
                "XAGUSD: signal direction %s not in allowed %s",
                signal.direction, allowed,
            )
            return False

        # Seasonal bull window — no shorts
        if current_dt.month in _BULL_MONTHS and signal.direction == "bearish":
            _log.debug("XAGUSD: seasonal bull window, blocking short")
            return False

        return True
