"""Bollinger Band + RSI mean reversion."""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from research.algos.base import BaseAlgo, AlgoSignal


class BBRSIAlgo(BaseAlgo):
    """
    Prior bar closes outside BB(20,2) AND RSI extreme.
    Current bar closes back inside the band → entry.
    TP = BB midline.  SL = 1.5× ATR beyond band.
    """
    name = "bb_rsi"

    def __init__(
        self,
        bb_period: int = 20, bb_std: float = 2.0,
        rsi_period: int = 14, rsi_long: float = 35.0, rsi_short: float = 65.0,
        sl_mult: float = 1.5, min_rr: float = 1.2,
    ):
        self.bb_period  = bb_period
        self.bb_std     = bb_std
        self.rsi_period = rsi_period
        self.rsi_long   = rsi_long
        self.rsi_short  = rsi_short
        self.sl_mult    = sl_mult
        self.min_rr     = min_rr

    def generate(self, m15_df: pd.DataFrame, h1_df: pd.DataFrame, idx: int) -> Optional[AlgoSignal]:
        if idx < self.bb_period + self.rsi_period + 5:
            return None

        close = m15_df["close"].astype(float)
        sma   = close.rolling(self.bb_period).mean()
        std   = close.rolling(self.bb_period).std()
        upper = sma + self.bb_std * std
        lower = sma - self.bb_std * std
        rsi_s = self.rsi(close, self.rsi_period)
        atr_s = self.atr(m15_df)

        c_curr  = float(close.iloc[idx])
        c_prev  = float(close.iloc[idx - 1])
        up_curr = float(upper.iloc[idx])
        up_prev = float(upper.iloc[idx - 1])
        lo_curr = float(lower.iloc[idx])
        lo_prev = float(lower.iloc[idx - 1])
        mid     = float(sma.iloc[idx])
        rsi_val = float(rsi_s.iloc[idx])
        atr_val = float(atr_s.iloc[idx])

        if np.isnan(rsi_val) or np.isnan(mid) or atr_val <= 0:
            return None

        if c_prev < lo_prev and c_curr > lo_curr and rsi_val < self.rsi_long:
            direction = "long"
            entry = c_curr
            sl    = entry - self.sl_mult * atr_val
            tp    = mid
        elif c_prev > up_prev and c_curr < up_curr and rsi_val > self.rsi_short:
            direction = "short"
            entry = c_curr
            sl    = entry + self.sl_mult * atr_val
            tp    = mid
        else:
            return None

        risk   = abs(entry - sl)
        reward = abs(tp - entry)
        if risk <= 0 or reward / risk < self.min_rr:
            return None
        return AlgoSignal(direction, entry, sl, tp, round(reward / risk, 2))
