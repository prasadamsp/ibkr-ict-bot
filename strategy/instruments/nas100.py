"""
nas100.py — NAS100 (NASDAQ-100) Instrument Strategy

Pre-filters:
  1. VIX proxy (panic) gate: H1 ATR% must be <= 2%.
  2. August confluence threshold raised internally via post-filter.

Post-filters:
  3. EMA200 direction gate: above EMA200 → only longs.
  4. Momentum confirmation: last 3 M15 closes aligned with signal direction.
  5. August: reject signals with confluence < 0.75 in August.

ICT signal logic (BOS/MSS, FVG, OB, AMD, CBDR, IPDA, etc.) handled by
ICTStrategy.on_bar().
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from strategy.instruments.base import BaseInstrumentStrategy
from strategy.strategy import TradeSignal

_log = logging.getLogger("strategy")

_VIX_PROXY_ATR_PCT      = 0.02
_AUGUST_MIN_CONFLUENCE  = 0.75
_DEFAULT_MIN_CONFLUENCE = 0.55


class NAS100Strategy(BaseInstrumentStrategy):
    """EMA-trend + ICT strategy for NAS100.

    Pre-filter:  VIX/panic guard (ATR% <= 2%).
    Post-filter: EMA200 direction restriction; momentum alignment; August gate.
    """

    def _pre_filter(self, m15_df, h1_df, current_dt, **kwargs) -> bool:
        # Panic guard
        regime = self.get_regime(h1_df)
        if regime.atr_pct > _VIX_PROXY_ATR_PCT:
            _log.debug(
                "NAS100: ATR%% %.4f > %.4f panic threshold, skipping",
                regime.atr_pct, _VIX_PROXY_ATR_PCT,
            )
            return False
        return True

    def _post_filter(self, signal, m15_df, h1_df, current_dt, **kwargs) -> bool:
        current_price = float(m15_df.iloc[-1]["close"])
        h1_closes = h1_df["close"].astype(float)
        ema200 = float(h1_closes.ewm(span=200, adjust=False).mean().iloc[-1])

        # Above EMA200 → long only
        if current_price > ema200 and signal.direction == "bearish":
            _log.debug("NAS100: price above EMA200, blocking short")
            return False

        # Momentum: last 3 M15 closes must align with direction
        if not self._momentum_aligned(m15_df, signal.direction):
            _log.debug("NAS100: last 3 M15 closes not aligned with %s", signal.direction)
            return False

        # August: raise confluence threshold
        if current_dt.month == 8 and signal.confluence_score < _AUGUST_MIN_CONFLUENCE:
            _log.debug(
                "NAS100: August confluence %.3f < %.3f",
                signal.confluence_score, _AUGUST_MIN_CONFLUENCE,
            )
            return False

        return True

    def _momentum_aligned(self, m15_df: pd.DataFrame, direction: str) -> bool:
        if len(m15_df) < 4:
            return False
        closes = m15_df["close"].values
        c1, c2, c3 = closes[-3], closes[-2], closes[-1]
        return (c2 > c1 and c3 > c2) if direction == "bullish" else (c2 < c1 and c3 < c2)
