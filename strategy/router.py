"""
router.py — StrategyRouter

The StrategyRouter is the single entry point for signal generation across all
7 instruments.  The calling code (DataHandler / main loop) calls:

    signal = router.route(symbol, m15_df, h1_df, current_dt, open_positions)

and receives either a TradeSignal or None.

Routing pipeline (per call)
---------------------------
1. CorrelationGuard.can_open()
   Check whether opening a new position in this instrument/direction is
   permitted given current portfolio exposure.  This is checked BEFORE any
   expensive signal computation.  If blocked, log and return None immediately.

   Note: because the direction is not yet known at gate time, we check
   CorrelationGuard *after* signal generation using the signal's direction.
   The pre-generation check is skipped here to avoid double-work; the router
   performs a post-signal guard check to reject correlated setups.

2. Instrument dispatch
   Route to the correct BaseInstrumentStrategy subclass by symbol.

3. Seasonality multiplier confirmation
   The instrument strategy already embeds size_mult in its signal notes.
   The router re-reads it from SeasonalityCalendar for the outer caller's
   convenience and appends it to a final log line.

4. Return signal or None

extra_data dictionary
---------------------
The `extra_data` dict passes instrument-specific context:
  {
    "xau_price":      float   — current XAU spot price (for GSR calc)
    "xag_price":      float   — current XAG spot price (for GSR calc)
    "gold_direction": str     — XAUUSD signal direction ("bullish"/"bearish"/"neutral")
  }

If extra_data is None or keys are missing, sensible defaults are used
(ratio = 80.0, gold_direction = "neutral").

Supported symbols
-----------------
  XAUUSD, NAS100, EURUSD, GBPUSD, BTC, XAGUSD, OIL, GBPJPY

Unknown symbols are logged at WARNING and return None.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from risk.correlation_guard import CorrelationGuard
from risk.news_blackout import NewsBlackout
from strategy.regime import RegimeDetector
from strategy.seasonality import SeasonalityCalendar
from strategy.strategy import TradeSignal

from strategy.instruments.btc    import BTCStrategy
from strategy.instruments.eurusd import EURUSDStrategy
from strategy.instruments.gbpjpy import GBPJPYStrategy
from strategy.instruments.gbpusd import GBPUSDStrategy
from strategy.instruments.nas100 import NAS100Strategy
from strategy.instruments.oil    import OILStrategy
from strategy.instruments.xagusd import XAGUSDStrategy
from strategy.instruments.xauusd import XAUUSDStrategy

_log = logging.getLogger("strategy")


class StrategyRouter:
    """
    Instantiates all instrument strategies and routes signal requests.

    Parameters
    ----------
    strategy_cfg :
        StrategyConfig dataclass from config.settings.  Passed to each
        instrument strategy for threshold access.
    risk_cfg :
        RiskConfig dataclass from config.settings.  Currently stored for
        future use (e.g. dynamic RR thresholds based on account risk).

    Usage
    -----
        router = StrategyRouter(CONFIG.strategy, CONFIG.risk)
        signal = router.route("XAUUSD", m15, h1, now, open_positions)
    """

    def __init__(self, strategy_cfg, risk_cfg) -> None:
        self._strategy_cfg = strategy_cfg
        self._risk_cfg = risk_cfg

        # Shared infrastructure instances
        self._regime_detector = RegimeDetector()
        self._seasonality = SeasonalityCalendar()
        self._correlation_guard = CorrelationGuard()
        self._news_blackout = NewsBlackout()

        # Instantiate all 8 instrument strategies
        _kwargs = dict(
            strategy_cfg=strategy_cfg,
            regime_detector=self._regime_detector,
            seasonality=self._seasonality,
        )
        self._strategies: Dict[str, object] = {
            "XAUUSD": XAUUSDStrategy(symbol="XAUUSD", **_kwargs),
            "NAS100": NAS100Strategy(symbol="NAS100", **_kwargs),
            "EURUSD": EURUSDStrategy(symbol="EURUSD", **_kwargs),
            "GBPUSD": GBPUSDStrategy(symbol="GBPUSD", **_kwargs),
            "BTC":    BTCStrategy(symbol="BTC",    **_kwargs),
            "XAGUSD": XAGUSDStrategy(symbol="XAGUSD", **_kwargs),
            "OIL":    OILStrategy(symbol="OIL",    **_kwargs),
            "GBPJPY": GBPJPYStrategy(symbol="GBPJPY", **_kwargs),
        }

        _log.info(
            "StrategyRouter initialised — %d instruments: %s",
            len(self._strategies),
            ", ".join(self._strategies.keys()),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(
        self,
        symbol: str,
        m15_df: pd.DataFrame,
        h1_df: pd.DataFrame,
        current_dt: datetime,
        open_positions: List,
        extra_data: Optional[Dict] = None,
        d1_df: Optional[pd.DataFrame] = None,
    ) -> Optional[TradeSignal]:
        """
        Generate a trade signal for the given symbol, or return None.

        Parameters
        ----------
        symbol : str
            Instrument identifier (e.g. "XAUUSD").
        m15_df : pd.DataFrame
            15-minute OHLCV bars (closed, ascending).
        h1_df : pd.DataFrame
            1-hour OHLCV bars (closed, ascending).
        current_dt : datetime
            Timestamp of the most recently closed M15 bar.
        open_positions : List[OpenPosition]
            All currently open positions; used by CorrelationGuard.
        extra_data : dict, optional
            Instrument-specific context.  Supported keys:
              "xau_price"      → float (XAU spot for GSR)
              "xag_price"      → float (XAG spot for GSR)
              "gold_direction" → str   (from XAUUSD signal)

        Returns
        -------
        TradeSignal or None
        """
        sym = symbol.upper().strip()
        extra = extra_data or {}

        # --- Instrument dispatch ---
        strategy = self._strategies.get(sym)
        if strategy is None:
            _log.warning("StrategyRouter: unknown symbol '%s', skipping", sym)
            return None

        # --- News blackout check (before any expensive computation) ---
        blocked, blackout_reason = self._news_blackout.is_blocked(sym, current_dt)
        if blocked:
            _log.info("StrategyRouter: %s NEWS BLACKOUT — %s", sym, blackout_reason)
            return None

        _log.debug("StrategyRouter: routing %s at %s", sym, current_dt)

        # --- Generate signal ---
        try:
            signal = self._call_strategy(strategy, sym, m15_df, h1_df, current_dt, extra, d1_df)
        except Exception as exc:
            _log.error("StrategyRouter: unexpected error routing %s: %s", sym, exc, exc_info=True)
            return None

        if signal is None:
            _log.debug("StrategyRouter: %s → no signal", sym)
            return None

        # --- Correlation guard (post-signal: direction is now known) ---
        direction_str = "long" if signal.direction == "bullish" else "short"
        can_open, reason = self._correlation_guard.can_open(sym, direction_str, open_positions)
        if not can_open:
            _log.debug("StrategyRouter: %s correlation guard blocked — %s", sym, reason)
            return None

        # --- Seasonality multiplier confirmation log ---
        season_mult = self._seasonality.get_multiplier(sym, current_dt)
        season_note = self._seasonality.get_note(sym, current_dt)
        _log.info(
            "StrategyRouter: SIGNAL [%s] %s | Entry=%.5f SL=%.5f TP=%.5f "
            "RR=%.2f Score=%.3f | season_mult=%.2f (%s)",
            sym,
            signal.direction.upper(),
            signal.entry_price,
            signal.stop_loss,
            signal.take_profit,
            signal.rr_ratio,
            signal.confluence_score,
            season_mult,
            season_note,
        )

        return signal

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_strategy(
        self,
        strategy,
        symbol: str,
        m15_df: pd.DataFrame,
        h1_df: pd.DataFrame,
        current_dt: datetime,
        extra: Dict,
        d1_df=None,
    ) -> Optional[TradeSignal]:
        """
        Dispatch to the correct strategy, passing extra_data where needed.
        """
        if symbol == "XAGUSD":
            # Calculate Gold-Silver ratio if price data available
            xau = extra.get("xau_price")
            xag = extra.get("xag_price")
            if xau and xag and xag > 0:
                ratio = xau / xag
            else:
                ratio = 80.0   # neutral default
            gold_dir = extra.get("gold_direction", "neutral")
            _log.debug(
                "StrategyRouter: XAGUSD extra — ratio=%.2f gold_dir=%s", ratio, gold_dir
            )
            return strategy.generate_signal(
                m15_df, h1_df, current_dt,
                ratio=ratio,
                gold_direction=gold_dir,
                d1_df=d1_df,
            )
        else:
            return strategy.generate_signal(m15_df, h1_df, current_dt, d1_df=d1_df)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_strategy(self, symbol: str):
        """Return the instrument strategy instance for the given symbol, or None."""
        return self._strategies.get(symbol.upper().strip())

    def symbols(self) -> List[str]:
        """Return list of all supported symbol identifiers."""
        return list(self._strategies.keys())
