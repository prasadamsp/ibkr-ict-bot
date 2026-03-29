"""EMA crossover — fast EMA crosses above/below slow EMA."""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from research.algos.base import BaseAlgo, AlgoSignal


class MACrossoverAlgo(BaseAlgo):
    """
    EMA(fast) crosses EMA(slow).  Trend-following.

    Long:  fast crossed above slow on previous bar, confirmed by close > slow.
    Short: fast crossed below slow.
    SL:    1.5× ATR beyond entry.
    TP:    2.5× ATR (RR ~1.7).
    """
    name = "ma_crossover"

    def __init__(self, fast: int = 20, slow: int = 50, sl_mult: float = 1.5, tp_mult: float = 2.5):
        self.fast    = fast
        self.slow    = slow
        self.sl_mult = sl_mult
        self.tp_mult = tp_mult

    def generate(self, m15_df: pd.DataFrame, h1_df: pd.DataFrame, idx: int) -> Optional[AlgoSignal]:
        if idx < self.slow + 5:
            return None

        close = m15_df["close"].astype(float)
        ema_f = self.ema(close, self.fast)
        ema_s = self.ema(close, self.slow)
        atr   = self.atr(m15_df).iloc[idx]

        if np.isnan(atr) or atr <= 0:
            return None

        f_curr, f_prev = float(ema_f.iloc[idx]), float(ema_f.iloc[idx - 1])
        s_curr, s_prev = float(ema_s.iloc[idx]), float(ema_s.iloc[idx - 1])
        c_curr = float(close.iloc[idx])

        # Golden cross (prev: fast below slow → curr: fast above slow)
        if f_prev <= s_prev and f_curr > s_curr and c_curr > s_curr:
            direction = "long"
            entry = c_curr
            sl    = entry - self.sl_mult * atr
            tp    = entry + self.tp_mult * atr
        # Death cross
        elif f_prev >= s_prev and f_curr < s_curr and c_curr < s_curr:
            direction = "short"
            entry = c_curr
            sl    = entry + self.sl_mult * atr
            tp    = entry - self.tp_mult * atr
        else:
            return None

        risk   = abs(entry - sl)
        reward = abs(tp - entry)
        if risk <= 0:
            return None
        return AlgoSignal(direction, entry, sl, tp, round(reward / risk, 2))
