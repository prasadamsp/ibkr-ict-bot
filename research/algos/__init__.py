"""
Algorithm library — all strategies available to the grid search.

Each algo is a stateless class with a generate(m15_df, h1_df, idx) method.
New algos: subclass BaseAlgo, add to ALL_ALGOS list at bottom.
"""

from research.algos.base          import BaseAlgo, AlgoSignal
from research.algos.ma_crossover  import MACrossoverAlgo
from research.algos.bb_rsi        import BBRSIAlgo
from research.algos.rsi_extreme   import RSIExtremeAlgo
from research.algos.donchian      import DonchianBreakoutAlgo
from research.algos.macd_momentum import MACDMomentumAlgo
from research.algos.keltner       import KeltnerReversionAlgo
from research.algos.zscore        import ZScoreReversionAlgo
from research.algos.ema_pullback  import EMAPullbackAlgo
from research.algos.vwap_revert   import VWAPReversionAlgo
from research.algos.ict_fvg       import ICTFVGAlgo

# Master list — grid search iterates over all of these
ALL_ALGOS: list[type[BaseAlgo]] = [
    MACrossoverAlgo,
    BBRSIAlgo,
    RSIExtremeAlgo,
    DonchianBreakoutAlgo,
    MACDMomentumAlgo,
    KeltnerReversionAlgo,
    ZScoreReversionAlgo,
    EMAPullbackAlgo,
    VWAPReversionAlgo,
    ICTFVGAlgo,
]

__all__ = ["BaseAlgo", "AlgoSignal", "ALL_ALGOS"] + [a.__name__ for a in ALL_ALGOS]
