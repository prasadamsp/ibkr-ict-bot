"""MACD histogram momentum — enter when histogram flips sign after extreme."""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from research.algos.base import BaseAlgo, AlgoSignal


class MACDMomentumAlgo(BaseAlgo):
    """
    MACD histogram crosses zero from extreme:
      Long:  histogram was negative (< -threshold), flips positive this bar.
      Short: histogram was positive (> +threshold), flips negative.
    SL: 1.5× ATR.  TP: 2.5× ATR.
    """
    name = "macd_momentum"

    def __init__(
        self,
        fast: int = 12, slow: int = 26, signal: int = 9,
        hist_threshold: float = 0.0,   # 0 = any crossing
        sl_mult: float = 1.5, tp_mult: float = 2.5,
    ):
        self.fast      = fast
        self.slow      = slow
        self.signal    = signal
        self.threshold = hist_threshold
        self.sl_mult   = sl_mult
        self.tp_mult   = tp_mult

    def generate(self, m15_df: pd.DataFrame, h1_df: pd.DataFrame, idx: int) -> Optional[AlgoSignal]:
        if idx < self.slow + self.signal + 5:
            return None

        close  = m15_df["close"].astype(float)
        macd   = self.ema(close, self.fast) - self.ema(close, self.slow)
        sig    = self.ema(macd, self.signal)
        hist   = macd - sig

        h_curr  = float(hist.iloc[idx])
        h_prev  = float(hist.iloc[idx - 1])
        c_curr  = float(close.iloc[idx])
        atr_val = float(self.atr(m15_df).iloc[idx])

        if np.isnan(h_curr) or atr_val <= 0:
            return None

        if h_prev < -self.threshold and h_curr > 0:
            direction = "long"
        elif h_prev > self.threshold and h_curr < 0:
            direction = "short"
        else:
            return None

        entry = c_curr
        if direction == "long":
            sl = entry - self.sl_mult * atr_val
            tp = entry + self.tp_mult * atr_val
        else:
            sl = entry + self.sl_mult * atr_val
            tp = entry - self.tp_mult * atr_val

        risk   = abs(entry - sl)
        reward = abs(tp - entry)
        if risk <= 0:
            return None
        return AlgoSignal(direction, entry, sl, tp, round(reward / risk, 2))
