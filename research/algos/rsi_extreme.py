"""RSI extreme reversal — pure RSI without Bollinger Bands."""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from research.algos.base import BaseAlgo, AlgoSignal


class RSIExtremeAlgo(BaseAlgo):
    """
    RSI crosses back from extreme → mean reversion entry.
    Long:  RSI was < oversold last bar, now ticking up (RSI curr > RSI prev).
    Short: RSI was > overbought last bar, now ticking down.
    SL:    1.0× ATR.  TP:  1.5× ATR.
    """
    name = "rsi_extreme"

    def __init__(
        self,
        rsi_period: int = 14, oversold: float = 30.0, overbought: float = 70.0,
        sl_mult: float = 1.0, tp_mult: float = 1.5,
    ):
        self.rsi_period  = rsi_period
        self.oversold    = oversold
        self.overbought  = overbought
        self.sl_mult     = sl_mult
        self.tp_mult     = tp_mult

    def generate(self, m15_df: pd.DataFrame, h1_df: pd.DataFrame, idx: int) -> Optional[AlgoSignal]:
        if idx < self.rsi_period + 5:
            return None

        close   = m15_df["close"].astype(float)
        rsi_s   = self.rsi(close, self.rsi_period)
        atr_val = float(self.atr(m15_df).iloc[idx])
        r_curr  = float(rsi_s.iloc[idx])
        r_prev  = float(rsi_s.iloc[idx - 1])
        c_curr  = float(close.iloc[idx])

        if np.isnan(r_curr) or atr_val <= 0:
            return None

        if r_prev < self.oversold and r_curr > r_prev:
            direction = "long"
            entry = c_curr
            sl    = entry - self.sl_mult * atr_val
            tp    = entry + self.tp_mult * atr_val
        elif r_prev > self.overbought and r_curr < r_prev:
            direction = "short"
            entry = c_curr
            sl    = entry + self.sl_mult * atr_val
            tp    = entry - self.tp_mult * atr_val
        else:
            return None

        risk   = abs(entry - sl)
        reward = abs(tp - entry)
        if risk <= 0:
            return None
        return AlgoSignal(direction, entry, sl, tp, round(reward / risk, 2))
