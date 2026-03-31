"""MACD histogram momentum — enter when histogram flips sign after extreme.

Filters applied (v2):
  1. H1 EMA trend alignment — only long above H1 EMA, only short below.
     Eliminates flip-flopping against the prevailing trend.
  2. Minimum histogram extreme — prev histogram must exceed hist_threshold
     (as fraction of ATR) before crossing counts.  Filters noise crossings.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from research.algos.base import BaseAlgo, AlgoSignal


class MACDMomentumAlgo(BaseAlgo):
    """
    MACD histogram crosses zero from extreme, aligned with H1 EMA trend:
      Long:  histogram was negative (< -threshold), flips positive, H1 close > H1 EMA.
      Short: histogram was positive (> +threshold), flips negative, H1 close < H1 EMA.
    SL: 1.5× ATR.  TP: 2.5× ATR.
    """
    name = "macd_momentum"

    def __init__(
        self,
        fast: int = 12, slow: int = 26, signal: int = 9,
        hist_threshold: float = 0.0,   # min |histogram| before crossing (0 = any)
        sl_mult: float = 1.5, tp_mult: float = 2.5,
        h1_ema_period: int = 50,       # H1 EMA period for trend filter
    ):
        self.fast          = fast
        self.slow          = slow
        self.signal        = signal
        self.threshold     = hist_threshold
        self.sl_mult       = sl_mult
        self.tp_mult       = tp_mult
        self.h1_ema_period = h1_ema_period

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

        # --- H1 trend filter ---
        # Require at least enough H1 bars to compute the EMA
        if h1_df is not None and len(h1_df) >= self.h1_ema_period:
            h1_close = h1_df["close"].astype(float)
            h1_ema   = float(self.ema(h1_close, self.h1_ema_period).iloc[-1])
            h1_price = float(h1_close.iloc[-1])
            h1_bullish = h1_price > h1_ema
            h1_bearish = h1_price < h1_ema
        else:
            # Not enough H1 data — skip trend filter
            h1_bullish = h1_bearish = True

        # --- MACD crossing with minimum extreme ---
        if h_prev < -self.threshold and h_curr > 0 and h1_bullish:
            direction = "long"
        elif h_prev > self.threshold and h_curr < 0 and h1_bearish:
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
