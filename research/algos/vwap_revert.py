"""VWAP deviation reversion — intraday mean reversion from VWAP extremes."""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from research.algos.base import BaseAlgo, AlgoSignal


class VWAPReversionAlgo(BaseAlgo):
    """
    Rolling VWAP calculated over a daily window (reset each calendar day).
    Long:  price deviates > threshold standard deviations below VWAP.
    Short: price deviates > threshold standard deviations above VWAP.
    TP = VWAP.  SL = 1.5× ATR.

    Works well for: intraday instruments with volume data (NAS100, BTC, FX).
    Uses daily VWAP calculated from M15 bars (96 bars per day).
    """
    name = "vwap_reversion"

    def __init__(
        self,
        daily_bars: int = 96,   # 96 × 15m = 1 trading day
        std_threshold: float = 1.5,
        sl_mult: float = 1.5, min_rr: float = 1.2,
    ):
        self.daily_bars    = daily_bars
        self.threshold     = std_threshold
        self.sl_mult       = sl_mult
        self.min_rr        = min_rr

    def generate(self, m15_df: pd.DataFrame, h1_df: pd.DataFrame, idx: int) -> Optional[AlgoSignal]:
        if idx < self.daily_bars + 5:
            return None

        # Rolling VWAP over daily_bars window
        close  = m15_df["close"].astype(float)
        volume = m15_df["volume"].astype(float).replace(0, 1)  # avoid div/0

        win = slice(idx - self.daily_bars, idx + 1)
        c_win = close.iloc[win]
        v_win = volume.iloc[win]

        tp_win  = (m15_df["high"].astype(float).iloc[win] +
                   m15_df["low"].astype(float).iloc[win] +
                   c_win) / 3.0   # typical price
        vwap    = (tp_win * v_win).sum() / v_win.sum()

        # VWAP standard deviation
        vwap_std = float(np.sqrt(((tp_win - vwap) ** 2 * v_win).sum() / v_win.sum()))
        if vwap_std <= 0:
            return None

        c_curr  = float(close.iloc[idx])
        atr_val = float(self.atr(m15_df).iloc[idx])
        dev     = (c_curr - float(vwap)) / vwap_std

        if atr_val <= 0:
            return None

        if dev < -self.threshold:
            direction = "long"
            entry = c_curr
            sl    = entry - self.sl_mult * atr_val
            tp    = float(vwap)
        elif dev > self.threshold:
            direction = "short"
            entry = c_curr
            sl    = entry + self.sl_mult * atr_val
            tp    = float(vwap)
        else:
            return None

        risk   = abs(entry - sl)
        reward = abs(tp - entry)
        if risk <= 0 or reward / risk < self.min_rr:
            return None
        return AlgoSignal(direction, entry, sl, tp, round(reward / risk, 2))
