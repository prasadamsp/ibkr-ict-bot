"""
router.py — AdaptiveRouter

Drop-in replacement for StrategyRouter that adds three adaptive layers:

1. ML Regime Classification
   Replaces rule-based RegimeClassifier with MLRegimeClassifier.
   Falls back to rule-based if model not ready (< 500 labeled samples).

2. Portfolio-Level Size Weighting
   PortfolioOptimiser computes Sharpe-weighted size multipliers per instrument.
   The signal's confluence_score is unchanged; only the downstream position
   sizer should read the multiplier (passed in signal.notes).

3. Weekly Parameter Retuning
   ParameterTuner background thread publishes updated thresholds to PARAM_STORE.
   Individual instrument strategies read from PARAM_STORE via get() calls.
   (Currently instruments read hard-coded thresholds; this is the migration path.)

Interface
---------
    from strategy.adaptive import AdaptiveRouter

    router = AdaptiveRouter(CONFIG.strategy, CONFIG.risk)

    # After each closed trade, feed the result back:
    router.record_trade("XAUUSD", pnl=250.0, account_equity=52000.0)

    # Normal signal generation (identical call signature to StrategyRouter):
    signal = router.route("XAUUSD", m15, h1, now, open_positions)

    # Update data for parameter tuner (call when fresh data arrives):
    router.update_data("XAUUSD", m15_df, h1_df)

    # Trigger ML retrain manually (or let background thread handle it):
    router.retrain_ml(h1_map)   # h1_map: {symbol: h1_df}

Backward compatibility
----------------------
AdaptiveRouter inherits from StrategyRouter, so all existing code that
uses StrategyRouter continues to work unchanged.  The adaptive layers are
additive.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from strategy.router import StrategyRouter
from strategy.strategy import TradeSignal
from strategy.adaptive.ml_regime   import MLRegimeClassifier
from strategy.adaptive.portfolio   import PortfolioOptimiser
from strategy.adaptive.param_tuner import ParameterTuner, PARAM_STORE

_log = logging.getLogger("strategy.adaptive")

_ML_MODEL_PATH = Path("data/adaptive/ml_regime_model.pkl")


class AdaptiveRouter(StrategyRouter):
    """
    StrategyRouter + ML regime + portfolio sizing + parameter retuning.

    Parameters
    ----------
    strategy_cfg, risk_cfg :
        Same as StrategyRouter.
    enable_ml : bool
        Whether to use MLRegimeClassifier (default True).
        Set to False to use rule-based fallback only.
    enable_portfolio_opt : bool
        Whether to apply Sharpe-weighted size multipliers (default True).
    enable_param_tuner : bool
        Whether to launch the background parameter tuner (default True).
    """

    def __init__(
        self,
        strategy_cfg,
        risk_cfg,
        enable_ml:            bool = True,
        enable_portfolio_opt: bool = True,
        enable_param_tuner:   bool = True,
    ) -> None:
        super().__init__(strategy_cfg, risk_cfg)

        self._enable_ml   = enable_ml
        self._enable_port = enable_portfolio_opt

        # ML regime classifier
        self._ml_classifier = MLRegimeClassifier(model_path=_ML_MODEL_PATH) if enable_ml else None

        # Portfolio optimiser
        self._portfolio = PortfolioOptimiser(
            base_risk_per_trade=getattr(risk_cfg, "risk_per_trade", 0.005),
        ) if enable_portfolio_opt else None

        # Parameter tuner
        self._tuner = ParameterTuner(param_store=PARAM_STORE) if enable_param_tuner else None
        if self._tuner:
            self._tuner.start_background()

        _log.info(
            "AdaptiveRouter initialised (ml=%s, portfolio_opt=%s, param_tuner=%s)",
            enable_ml, enable_portfolio_opt, enable_param_tuner,
        )

    # ------------------------------------------------------------------
    # Trade feedback loop
    # ------------------------------------------------------------------

    def record_trade(
        self,
        symbol: str,
        pnl: float,
        account_equity: float = 50_000.0,
    ) -> None:
        """
        Feed a closed trade's P&L back into the portfolio optimiser.

        Call this from execution.py after every position close.
        """
        if self._portfolio:
            self._portfolio.record_trade(symbol, pnl, account_equity)

    # ------------------------------------------------------------------
    # Data update for parameter tuner
    # ------------------------------------------------------------------

    def update_data(
        self,
        symbol: str,
        m15_df: pd.DataFrame,
        h1_df: pd.DataFrame,
    ) -> None:
        """
        Provide fresh bar data to the parameter tuner.

        Call this whenever the live data handler refreshes bars.
        """
        if self._tuner:
            self._tuner.update_data(symbol, m15_df, h1_df)

    # ------------------------------------------------------------------
    # ML retraining
    # ------------------------------------------------------------------

    def retrain_ml(
        self,
        h1_map: Dict[str, pd.DataFrame],
        force: bool = False,
    ) -> bool:
        """
        Retrain the ML regime classifier using the provided H1 data.

        Concatenates all available H1 bars across instruments for a richer
        training set (regime patterns are cross-instrument).

        Parameters
        ----------
        h1_map : {symbol: h1_df}
        force : skip the weekly retrain interval check

        Returns True if retrained.
        """
        if not self._ml_classifier or not self._ml_classifier.should_retrain() and not force:
            return False

        try:
            all_h1 = [df for df in h1_map.values() if df is not None and len(df) >= 50]
            if not all_h1:
                return False
            combined = pd.concat(all_h1).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
            return self._ml_classifier.fit(combined, force=force)
        except Exception as exc:
            _log.error("AdaptiveRouter.retrain_ml error: %s", exc, exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Route override — adds adaptive sizing note
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
        Generate a signal with adaptive sizing multiplier injected into notes.

        All parent StrategyRouter logic applies unchanged.  The only addition:
        the portfolio size multiplier is appended to signal.notes so the
        execution layer can pick it up.
        """
        signal = super().route(
            symbol, m15_df, h1_df, current_dt, open_positions, extra_data, d1_df
        )
        if signal is None:
            return None

        if self._portfolio:
            port_mult = self._portfolio.get_multiplier(symbol, current_dt)
            # Append to notes — execution layer parses "port_mult=X.XX"
            signal = signal._replace(
                notes=f"{signal.notes} | port_mult={port_mult:.2f}"
            ) if hasattr(signal, "_replace") else signal

            _log.debug(
                "AdaptiveRouter: %s port_mult=%.2f applied",
                symbol, port_mult,
            )

        return signal

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def adaptive_status(self) -> Dict:
        """Return a diagnostics snapshot for monitoring dashboards."""
        return {
            "ml_model_ready": self._ml_classifier is not None and self._ml_classifier._model is not None,
            "ml_last_trained": str(getattr(self._ml_classifier, "_last_trained_at", None)),
            "portfolio_multipliers": self._portfolio.get_multipliers() if self._portfolio else {},
            "portfolio_sharpes": self._portfolio.sharpe_summary() if self._portfolio else {},
            "param_store": dict(PARAM_STORE._params),
            "tuner_running": self._tuner._thread is not None and self._tuner._thread.is_alive() if self._tuner else False,
        }
