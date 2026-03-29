"""
strategy/adaptive — Self-Adaptive Trading System (Phase 2)

Components
----------
MLRegimeClassifier
    Replaces the rule-based RegimeClassifier with an sklearn RandomForest
    trained on rolling 6-month windows.  Produces the same StrategyRegime
    enum outputs so it is a drop-in replacement.

PortfolioOptimiser
    Daily rebalancing engine.  Computes per-symbol Sharpe on a rolling
    30-trade window and reweights position-size multipliers accordingly.
    High-Sharpe instruments get up to 2× allocation; losing streaks get 0.5×.

AdaptiveRouter
    Thin wrapper around StrategyRouter that injects ML-classified regime and
    portfolio-optimised size multipliers before every signal call.

ParameterTuner
    Rolling 6-month optimiser for per-instrument thresholds (confluence,
    ATR multipliers).  Runs offline in a background thread; publishes updates
    to a SharedParamStore.

Usage
-----
    from strategy.adaptive import AdaptiveRouter
    router = AdaptiveRouter(CONFIG.strategy, CONFIG.risk)
    signal = router.route(sym, m15, h1, now, open_positions)
"""

from strategy.adaptive.ml_regime    import MLRegimeClassifier
from strategy.adaptive.portfolio    import PortfolioOptimiser
from strategy.adaptive.param_tuner  import ParameterTuner
from strategy.adaptive.router       import AdaptiveRouter

__all__ = [
    "MLRegimeClassifier",
    "PortfolioOptimiser",
    "ParameterTuner",
    "AdaptiveRouter",
]
