"""Donchian Channel breakout — trend-following momentum."""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from research.algos.base import BaseAlgo, AlgoSignal


class DonchianBreakoutAlgo(BaseAlgo):
    """
    Price breaks above N-bar high (or below N-bar low) → enter in breakout direction.
    Requires: close breaks out AND prior bar was inside channel (avoids chasing).
    SL: opposite channel band.  TP: 2× channel width from entry.
    """
    name = "donchian_breakout"

    def __init__(self, period: int = 20, sl_mult: float = 1.0, tp_mult: float = 2.0):
        self.period   = period
        self.sl_mult  = sl_mult
        self.tp_mult  = tp_mult

    def generate(self, m15_df: pd.DataFrame, h1_df: pd.DataFrame, idx: int) -> Optional[AlgoSignal]:
        if idx < self.period + 5:
            return None

        high  = m15_df["high"].astype(float)
        low   = m15_df["low"].astype(float)
        close = m15_df["close"].astype(float)

        # Channel from bars [idx-period .. idx-1] (exclude current)
        ch_high = float(high.iloc[idx - self.period: idx].max())
        ch_low  = float(low.iloc[idx - self.period: idx].min())
        c_curr  = float(close.iloc[idx])
        c_prev  = float(close.iloc[idx - 1])
        atr_val = float(self.atr(m15_df).iloc[idx])

        if atr_val <= 0 or np.isnan(ch_high):
            return None

        ch_width = ch_high - ch_low

        if c_prev <= ch_high and c_curr > ch_high:
            direction = "long"
            entry = c_curr
            sl    = entry - self.sl_mult * ch_width
            tp    = entry + self.tp_mult * ch_width
        elif c_prev >= ch_low and c_curr < ch_low:
            direction = "short"
            entry = c_curr
            sl    = entry + self.sl_mult * ch_width
            tp    = entry - self.tp_mult * ch_width
        else:
            return None

        risk   = abs(entry - sl)
        reward = abs(tp - entry)
        if risk <= 0:
            return None
        return AlgoSignal(direction, entry, sl, tp, round(reward / risk, 2))
