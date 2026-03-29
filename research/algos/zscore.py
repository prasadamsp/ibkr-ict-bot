"""Z-score mean reversion — statistical deviation from rolling mean."""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from research.algos.base import BaseAlgo, AlgoSignal


class ZScoreReversionAlgo(BaseAlgo):
    """
    Compute rolling Z-score of close price.
    Long:  Z-score < -threshold (price statistically cheap), reverts up.
    Short: Z-score > +threshold (price statistically dear), reverts down.
    TP = rolling mean.  SL = 1.5× ATR.
    Good for: range-bound instruments, FX pairs during low-volatility sessions.
    """
    name = "zscore_reversion"

    def __init__(
        self,
        period: int = 30, threshold: float = 2.0,
        sl_mult: float = 1.5, min_rr: float = 1.2,
    ):
        self.period    = period
        self.threshold = threshold
        self.sl_mult   = sl_mult
        self.min_rr    = min_rr

    def generate(self, m15_df: pd.DataFrame, h1_df: pd.DataFrame, idx: int) -> Optional[AlgoSignal]:
        if idx < self.period + 5:
            return None

        close   = m15_df["close"].astype(float)
        roll_m  = close.rolling(self.period).mean()
        roll_s  = close.rolling(self.period).std()
        zscore  = (close - roll_m) / roll_s.replace(0, np.nan)
        atr_s   = self.atr(m15_df)

        z_curr  = float(zscore.iloc[idx])
        z_prev  = float(zscore.iloc[idx - 1])
        c_curr  = float(close.iloc[idx])
        mean_v  = float(roll_m.iloc[idx])
        atr_val = float(atr_s.iloc[idx])

        if np.isnan(z_curr) or np.isnan(mean_v) or atr_val <= 0:
            return None

        # Require Z-score to be reverting (moving back toward mean)
        if z_prev < -self.threshold and z_curr > z_prev:
            direction = "long"
            entry = c_curr
            sl    = entry - self.sl_mult * atr_val
            tp    = mean_v
        elif z_prev > self.threshold and z_curr < z_prev:
            direction = "short"
            entry = c_curr
            sl    = entry + self.sl_mult * atr_val
            tp    = mean_v
        else:
            return None

        risk   = abs(entry - sl)
        reward = abs(tp - entry)
        if risk <= 0 or reward / risk < self.min_rr:
            return None
        return AlgoSignal(direction, entry, sl, tp, round(reward / risk, 2))
