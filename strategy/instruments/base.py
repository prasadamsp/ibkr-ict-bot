"""
base.py — Abstract base class for all instrument-specific strategy modules.

Each instrument strategy (XAUUSD, NAS100, EURUSD, etc.) inherits from
BaseInstrumentStrategy and implements generate_signal().

The base class provides two concrete helpers:
  - get_size_multiplier(): delegates to SeasonalityCalendar
  - get_regime():          delegates to RegimeDetector

All generate_signal() implementations must:
  - Return None rather than raise on any error
  - Guard against insufficient bar count (< 50 bars)
  - Never block the calling thread longer than a few milliseconds
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import pandas as pd

from strategy.regime import RegimeDetector, RegimeState
from strategy.seasonality import SeasonalityCalendar
from strategy.strategy import TradeSignal


class BaseInstrumentStrategy(ABC):
    """
    Abstract base for a single-instrument ICT strategy module.

    Parameters
    ----------
    symbol : str
        Instrument identifier (e.g. "XAUUSD").  Used for logging and
        as the symbol field on emitted TradeSignal objects.
    strategy_cfg : StrategyConfig
        Config dataclass from config.settings (passed through so each
        instrument can read thresholds like min_confluence_score).
    regime_detector : RegimeDetector
        Shared RegimeDetector instance — allows one set of EMA/ADX
        computations to serve multiple instruments.
    seasonality : SeasonalityCalendar
        Shared SeasonalityCalendar instance — returns monthly multipliers
        and EIA-day flags.
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

    # ------------------------------------------------------------------
    # Abstract interface — subclasses MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    def generate_signal(
        self,
        m15_df: pd.DataFrame,
        h1_df: pd.DataFrame,
        current_dt: datetime,
    ) -> Optional[TradeSignal]:
        """
        Analyse the provided dataframes and return a TradeSignal or None.

        Contract (all implementations must honour):
          - Return None if len(m15_df) < 50 or len(h1_df) < 30.
          - Catch all exceptions internally; never raise.
          - Never perform I/O or sleep inside this method.
          - Entry, stop_loss, and take_profit must be finite positive floats.
          - rr_ratio must be >= 2.0 (instruments may override to 3.0).

        Parameters
        ----------
        m15_df : pd.DataFrame
            15-minute OHLCV bars (closed bars only, ascending datetime index).
        h1_df : pd.DataFrame
            1-hour OHLCV bars (closed bars only, ascending datetime index).
        current_dt : datetime
            The timestamp of the most recently closed M15 bar.

        Returns
        -------
        TradeSignal or None
        """

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def get_size_multiplier(self, current_dt: datetime) -> float:
        """
        Return the seasonality position-size multiplier for this instrument.

        Delegates to SeasonalityCalendar.get_multiplier(symbol, dt).
        Returns 1.0 (neutral) if an error occurs.

        Parameters
        ----------
        current_dt : datetime
            Reference date for the seasonal lookup.

        Returns
        -------
        float   Multiplier in [0.5, 1.5]; 1.0 = no seasonal adjustment.
        """
        try:
            return self._seasonality.get_multiplier(self.symbol, current_dt)
        except Exception:
            return 1.0

    def get_regime(self, df: pd.DataFrame) -> RegimeState:
        """
        Classify the current market regime from the supplied OHLCV dataframe.

        Delegates to RegimeDetector.detect(df).  If detection fails (e.g.
        insufficient bars), returns a neutral RegimeState with adx=0,
        trend="sideways", volatility="normal".

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV bars (any timeframe).

        Returns
        -------
        RegimeState
        """
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
