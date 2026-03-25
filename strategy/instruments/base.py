"""
base.py — Abstract base class for all instrument-specific strategy modules.

Architecture
────────────
Each instrument strategy (XAUUSD, NAS100, EURUSD, etc.) inherits from
BaseInstrumentStrategy and:

1. Applies instrument-specific PRE-FILTERS (ADX gate, seasonal window, etc.)
   in `_pre_filter()` — return False to skip signal generation.

2. Delegates CORE SIGNAL LOGIC to `ICTStrategy.on_bar()` which runs the full
   ICT stack: H1 bias → D1 gate → session → AMD (Power of 3) → CBDR →
   BOS/MSS + displacement → FVG → OB → Breaker → liquidity sweep →
   weighted confluence → IPDA TP → CBDR TP projection → TradeSignal.

3. Applies instrument-specific POST-FILTERS (e.g. seasonal direction block,
   special ratio logic for XAGUSD) in `_post_filter()` — return False to
   discard the signal.

This ensures live trading and backtesting run identical logic.
Previously, instrument strategies had their own ad-hoc signal generation
that bypassed the advanced ICT concepts added to ICTStrategy.

Concrete helpers provided:
  - get_size_multiplier(): delegates to SeasonalityCalendar
  - get_regime():          delegates to RegimeDetector
"""

from __future__ import annotations

import logging
from abc import ABC
from datetime import datetime
from typing import Optional

import pandas as pd

from strategy.regime import RegimeDetector, RegimeState
from strategy.seasonality import SeasonalityCalendar
from strategy.strategy import ICTStrategy, TradeSignal

_log = logging.getLogger("strategy")


class BaseInstrumentStrategy(ABC):
    """
    Abstract base for a single-instrument ICT strategy module.

    Parameters
    ----------
    symbol : str
        Instrument identifier (e.g. "XAUUSD").
    strategy_cfg : StrategyConfig
        Config dataclass from config.settings.
    regime_detector : RegimeDetector
        Shared RegimeDetector instance.
    seasonality : SeasonalityCalendar
        Shared SeasonalityCalendar instance.
    """

    def __init__(
        self,
        symbol: str,
        strategy_cfg,
        regime_detector: RegimeDetector,
        seasonality: SeasonalityCalendar,
    ) -> None:
        self.symbol = symbol
        self.cfg = strategy_cfg
        self._regime_detector = regime_detector
        self._seasonality = seasonality
        # Core ICT signal engine — shared across all instruments
        self._ict = ICTStrategy(strategy_cfg)

    # ------------------------------------------------------------------
    # Public interface — called by StrategyRouter
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        m15_df: pd.DataFrame,
        h1_df: pd.DataFrame,
        current_dt: datetime,
        d1_df: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> Optional[TradeSignal]:
        """
        Generate a trade signal for this instrument.

        Pipeline:
          1. Minimum bar guard
          2. _pre_filter()  — instrument-specific gates (ADX, seasonal, etc.)
          3. ICTStrategy.on_bar() — full ICT signal engine
          4. _post_filter() — instrument-specific post-processing

        Returns TradeSignal or None.  Never raises.
        """
        try:
            if len(m15_df) < 50 or len(h1_df) < 30:
                return None

            # Instrument-specific pre-filters
            if not self._pre_filter(m15_df, h1_df, current_dt, **kwargs):
                return None

            # Core ICT engine
            ts = pd.Timestamp(current_dt) if not isinstance(current_dt, pd.Timestamp) else current_dt
            signal = self._ict.on_bar(self.symbol, "M15", m15_df, h1_df, d1_df)

            if signal is None:
                return None

            # Instrument-specific post-filter
            if not self._post_filter(signal, m15_df, h1_df, current_dt, **kwargs):
                return None

            # Attach seasonality multiplier to notes
            size_mult = self.get_size_multiplier(current_dt)
            season_note = self._seasonality.get_note(self.symbol, current_dt)
            signal.notes = f"size_mult={size_mult:.2f} ({season_note})"

            return signal

        except Exception as exc:
            _log.error(
                "[%s] generate_signal error: %s", self.symbol, exc, exc_info=True
            )
            return None

    # ------------------------------------------------------------------
    # Override hooks — subclasses implement these
    # ------------------------------------------------------------------

    def _pre_filter(
        self,
        m15_df: pd.DataFrame,
        h1_df: pd.DataFrame,
        current_dt: datetime,
        **kwargs,
    ) -> bool:
        """
        Return True to allow signal generation, False to skip.
        Default: always allow.  Override for instrument-specific gates.
        """
        return True

    def _post_filter(
        self,
        signal: TradeSignal,
        m15_df: pd.DataFrame,
        h1_df: pd.DataFrame,
        current_dt: datetime,
        **kwargs,
    ) -> bool:
        """
        Return True to keep the signal, False to discard it.
        Default: always keep.  Override for instrument-specific rejections.
        """
        return True

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def get_size_multiplier(self, current_dt: datetime) -> float:
        """Return the seasonality position-size multiplier for this instrument."""
        try:
            return self._seasonality.get_multiplier(self.symbol, current_dt)
        except Exception:
            return 1.0

    def get_regime(self, df: pd.DataFrame) -> RegimeState:
        """Classify the current market regime from the supplied OHLCV dataframe."""
        try:
            return self._regime_detector.detect(df)
        except Exception:
            from strategy.regime import RegimeState
            return RegimeState(
                trend="sideways",
                volatility="normal",
                adx=0.0,
                atr=0.0,
                atr_pct=0.0,
            )
