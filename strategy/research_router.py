"""
research_router.py — ResearchRouter

Drop-in replacement for StrategyRouter that routes signal generation through
the validated research algorithms (from data/research/best_algos.json) rather
than the hand-crafted ICT strategies.

This enables a direct A/B comparison:

    main branch   → ICT strategies (StrategyRouter / AdaptiveRouter)
    algo-bridge   → Validated research algos (ResearchRouter)   ← this file

Both run simultaneously on the same paper account (different IBKR clientIds)
writing to separate log files so results can be compared over 2+ weeks.

Architecture
------------
On startup:
  1. Reads best_algos.json → { symbol: {algo, params, ...} }
  2. Instantiates the correct BaseAlgo subclass per symbol with stored params
  3. Falls back to a sensible default algo if a symbol is missing from the file

On each bar (route() call):
  1. News blackout check (NewsBlackout)
  2. Macro gate (macro_allows_signal) — same as ICT system
  3. Run algo.generate(m15_df, h1_df, idx=last_bar)
  4. Convert AlgoSignal → TradeSignal
  5. Macro confluence delta (raises effective threshold against macro headwinds)
  6. Correlation guard (CorrelationGuard.can_open)
  7. Concurrent FX gate (should_skip_concurrent_fx / record_signal)
  8. Return TradeSignal or None

AlgoSignal → TradeSignal conversion
-------------------------------------
AlgoSignal already contains entry, sl, tp, rr computed by the algo itself.
The mapping is direct:
  direction:        "long" → "bullish",  "short" → "bearish"
  confluence_score: derived from RR ratio (capped at 0.95)
                    rr=2.0 → 0.57,  rr=2.5 → 0.67,  rr=3.0 → 0.75
  score_* fields:   not applicable (algo-based, not ICT), left at 0

Confluence threshold
---------------------
MIN_RESEARCH_CONFLUENCE = 0.50  (lower than ICT 0.55 — algo signals are binary;
either the mechanical condition is met or not, so we trust the algo's own
SL/TP math rather than a multi-component confluence score)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from research.algos.base       import BaseAlgo, AlgoSignal
from research.algos.donchian   import DonchianBreakoutAlgo
from research.algos.ema_pullback import EMAPullbackAlgo
from research.algos.macd_momentum import MACDMomentumAlgo
from research.algos.bb_rsi     import BBRSIAlgo
from research.algos.rsi_extreme import RSIExtremeAlgo
from research.algos.zscore     import ZScoreReversionAlgo
from research.algos.ma_crossover import MACrossoverAlgo
from research.algos.ict_fvg    import ICTFVGAlgo

from research.macro_filters import macro_allows_signal, macro_confluence_delta
from risk.correlation_guard  import CorrelationGuard, OpenPosition
from risk.news_blackout      import NewsBlackout
from strategy.strategy       import TradeSignal

_log = logging.getLogger("strategy.research")

# ---------------------------------------------------------------------------
# Algo registry — maps research algo name → class
# ---------------------------------------------------------------------------

ALGO_REGISTRY: Dict[str, type] = {
    "donchian_breakout": DonchianBreakoutAlgo,
    "ema_pullback":      EMAPullbackAlgo,
    "macd_momentum":     MACDMomentumAlgo,
    "bb_rsi":            BBRSIAlgo,
    "rsi_extreme":       RSIExtremeAlgo,
    "zscore_reversion":  ZScoreReversionAlgo,
    "ma_crossover":      MACrossoverAlgo,
    "ict_fvg":           ICTFVGAlgo,
}

# Minimum RR-derived confluence to emit a signal (algo signals are binary —
# if it fires, it met its mechanical conditions, so threshold is deliberately
# lower than ICT's multi-component 0.55)
MIN_RESEARCH_CONFLUENCE = 0.50

# Default algo per symbol if best_algos.json is missing an entry
_DEFAULT_ALGO = "macd_momentum"
_DEFAULT_PARAMS: Dict = {}


# ---------------------------------------------------------------------------
# ResearchRouter
# ---------------------------------------------------------------------------

class ResearchRouter:
    """
    Routes signal generation through validated research algorithms.

    Identical public interface to StrategyRouter — route() accepts the same
    arguments and returns TradeSignal or None.

    Parameters
    ----------
    strategy_cfg :
        StrategyConfig from config.settings (used for session/threshold access).
    risk_cfg :
        RiskConfig from config.settings.
    best_algos_path : str or Path, optional
        Path to best_algos.json. Defaults to data/research/best_algos.json.
    """

    def __init__(self, strategy_cfg, risk_cfg, best_algos_path=None) -> None:
        self._strategy_cfg = strategy_cfg
        self._risk_cfg = risk_cfg

        self._correlation_guard = CorrelationGuard()
        self._news_blackout = NewsBlackout()

        path = Path(best_algos_path or "data/research/best_algos.json")
        self._algos: Dict[str, BaseAlgo] = self._load_algos(path)

        _log.info(
            "ResearchRouter initialised — %d instruments: %s",
            len(self._algos),
            ", ".join(self._algos.keys()),
        )

    # ------------------------------------------------------------------
    # Public API (identical signature to StrategyRouter.route)
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
        Generate a trade signal using the validated research algo for symbol.

        Returns TradeSignal or None.
        """
        sym = symbol.upper().strip()

        if sym not in self._algos:
            _log.warning("ResearchRouter: no algo configured for '%s', skipping", sym)
            return None

        # 1. News blackout
        blocked, reason = self._news_blackout.is_blocked(sym, current_dt)
        if blocked:
            _log.info("ResearchRouter: %s NEWS BLACKOUT — %s", sym, reason)
            return None

        # 2. Macro gate — same rules as ICT system
        if not macro_allows_signal(sym, "long", current_dt, h1_df, eurusd_h1=self._get_eurusd(sym, h1_df)):
            # Check short direction too before blocking entirely
            if not macro_allows_signal(sym, "short", current_dt, h1_df, eurusd_h1=self._get_eurusd(sym, h1_df)):
                _log.debug("ResearchRouter: %s macro blocked both directions", sym)
                return None

        # 3. Run the research algo on the latest bar
        algo = self._algos[sym]
        idx = len(m15_df) - 1

        if idx < 30:
            _log.debug("ResearchRouter: %s insufficient bars (%d)", sym, idx)
            return None

        try:
            algo_signal: Optional[AlgoSignal] = algo.generate(m15_df, h1_df, idx)
        except Exception as e:
            _log.error("ResearchRouter: %s algo error — %s", sym, e, exc_info=True)
            return None

        if algo_signal is None:
            _log.debug("ResearchRouter: %s → no signal", sym)
            return None

        # 4. Macro direction gate — now we know the direction, re-check
        direction_str = algo_signal.direction  # "long" or "short"
        if not macro_allows_signal(sym, direction_str, current_dt, h1_df,
                                   eurusd_h1=self._get_eurusd(sym, h1_df)):
            _log.debug("ResearchRouter: %s macro blocked direction=%s", sym, direction_str)
            return None

        # 5. Macro confluence delta — raise bar if macro is a headwind
        macro_delta = macro_confluence_delta(
            sym, direction_str, h1_df,
            eurusd_h1=self._get_eurusd(sym, h1_df),
        )
        confluence = self._rr_to_confluence(algo_signal.rr)
        effective_threshold = MIN_RESEARCH_CONFLUENCE + macro_delta

        if confluence < effective_threshold:
            _log.debug(
                "ResearchRouter: %s score %.2f below threshold %.2f (macro_delta=+%.2f)",
                sym, confluence, effective_threshold, macro_delta,
            )
            return None

        # 6. Convert AlgoSignal → TradeSignal
        signal = self._to_trade_signal(sym, algo_signal, confluence)

        # 7. Correlation guard
        can_open, cg_reason = self._correlation_guard.can_open(
            sym, direction_str, open_positions
        )
        if not can_open:
            _log.debug("ResearchRouter: %s correlation guard blocked — %s", sym, cg_reason)
            return None

        # 8. Concurrent FX gate
        skip, fx_reason = self._correlation_guard.should_skip_concurrent_fx(
            sym, direction_str, confluence, current_dt
        )
        if skip:
            _log.info("ResearchRouter: %s CONCURRENT FX GATE — %s", sym, fx_reason)
            return None

        # Signal approved — record for FX gate
        self._correlation_guard.record_signal(sym, direction_str, confluence, current_dt)

        _log.info(
            "ResearchRouter: SIGNAL [%s] %s | algo=%s Entry=%.5f SL=%.5f TP=%.5f "
            "RR=%.2f confluence=%.2f",
            sym,
            signal.direction.upper(),
            algo.name,
            signal.entry_price,
            signal.stop_loss,
            signal.take_profit,
            signal.rr_ratio,
            confluence,
        )
        return signal

    def symbols(self) -> List[str]:
        """Return list of supported symbol identifiers."""
        return list(self._algos.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_algos(self, path: Path) -> Dict[str, BaseAlgo]:
        """Read best_algos.json and instantiate algo per symbol."""
        algos: Dict[str, BaseAlgo] = {}

        if not path.exists():
            _log.warning(
                "ResearchRouter: best_algos.json not found at %s — "
                "all symbols will use default algo (%s)", path, _DEFAULT_ALGO
            )
            return algos

        with open(path) as f:
            best: dict = json.load(f)

        for symbol, info in best.items():
            algo_name = info.get("algo", _DEFAULT_ALGO)
            params = info.get("params", _DEFAULT_PARAMS)

            cls = ALGO_REGISTRY.get(algo_name)
            if cls is None:
                _log.warning(
                    "ResearchRouter: unknown algo '%s' for %s, using %s",
                    algo_name, symbol, _DEFAULT_ALGO
                )
                cls = ALGO_REGISTRY[_DEFAULT_ALGO]
                params = {}

            try:
                instance = cls(**params) if params else cls()
            except TypeError as e:
                _log.warning(
                    "ResearchRouter: param mismatch for %s(%s): %s — using defaults",
                    algo_name, params, e
                )
                instance = cls()

            algos[symbol] = instance
            _log.info(
                "ResearchRouter: %s → %s (val_sharpe=%.2f test_sharpe=%.2f)",
                symbol, algo_name,
                info.get("val_sharpe", 0.0),
                info.get("test_sharpe", 0.0),
            )

        return algos

    def _to_trade_signal(
        self, symbol: str, algo_signal: AlgoSignal, confluence: float
    ) -> TradeSignal:
        """Convert AlgoSignal to TradeSignal. Entry/SL/TP come directly from the algo."""
        direction = "bullish" if algo_signal.direction == "long" else "bearish"
        return TradeSignal(
            symbol=symbol,
            direction=direction,
            entry_price=algo_signal.entry,
            stop_loss=algo_signal.sl,
            take_profit=algo_signal.tp,
            rr_ratio=algo_signal.rr,
            confluence_score=confluence,
            # ICT sub-scores not applicable for research algos
            score_bos=0.0,
            score_fvg=0.0,
            score_order_block=0.0,
            score_liquidity=0.0,
            score_session=0.0,
            score_pd_zone=0.0,
        )

    @staticmethod
    def _rr_to_confluence(rr: float) -> float:
        """
        Derive a confluence proxy from the RR ratio.

        RR=2.0 → 0.57,  RR=2.5 → 0.67,  RR=3.0 → 0.75,  RR=5.0 → 0.95
        Formula: rr / (rr + 1.5), capped at 0.95.
        """
        if rr <= 0:
            return 0.0
        return min(rr / (rr + 1.5), 0.95)

    def _get_eurusd(self, symbol: str, h1_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Return EURUSD h1_df for macro_allows_signal if symbol IS EURUSD, else None."""
        return h1_df if symbol == "EURUSD" else None
