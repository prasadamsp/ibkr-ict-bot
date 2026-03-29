"""Keltner Channel mean reversion — tighter than BB, uses ATR bands."""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from research.algos.base import BaseAlgo, AlgoSignal


class KeltnerReversionAlgo(BaseAlgo):
    """
    Price closes outside Keltner Channel (EMA ± mult×ATR), then closes back inside.
    More robust than BB in trending markets (ATR adapts to volatility regime).
    TP = EMA midline.  SL = 1.5× ATR beyond outer band.
    """
    name = "keltner_reversion"

    def __init__(
        self,
        ema_period: int = 20, atr_period: int = 14, mult: float = 2.0,
        sl_mult: float = 1.5, min_rr: float = 1.2,
    ):
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.mult       = mult
        self.sl_mult    = sl_mult
        self.min_rr     = min_rr

    def generate(self, m15_df: pd.DataFrame, h1_df: pd.DataFrame, idx: int) -> Optional[AlgoSignal]:
        if idx < max(self.ema_period, self.atr_period) + 5:
            return None

        close   = m15_df["close"].astype(float)
        mid     = self.ema(close, self.ema_period)
        atr_s   = self.atr(m15_df, self.atr_period)
        upper   = mid + self.mult * atr_s
        lower   = mid - self.mult * atr_s

        c_curr  = float(close.iloc[idx])
        c_prev  = float(close.iloc[idx - 1])
        up_curr = float(upper.iloc[idx])
        up_prev = float(upper.iloc[idx - 1])
        lo_curr = float(lower.iloc[idx])
        lo_prev = float(lower.iloc[idx - 1])
        mid_val = float(mid.iloc[idx])
        atr_val = float(atr_s.iloc[idx])

        if np.isnan(mid_val) or atr_val <= 0:
            return None

        if c_prev < lo_prev and c_curr > lo_curr:
            direction = "long"
            entry = c_curr
            sl    = lo_curr - self.sl_mult * atr_val
            tp    = mid_val
        elif c_prev > up_prev and c_curr < up_curr:
            direction = "short"
            entry = c_curr
            sl    = up_curr + self.sl_mult * atr_val
            tp    = mid_val
        else:
            return None

        risk   = abs(entry - sl)
        reward = abs(tp - entry)
        if risk <= 0 or reward / risk < self.min_rr:
            return None
        return AlgoSignal(direction, entry, sl, tp, round(reward / risk, 2))
