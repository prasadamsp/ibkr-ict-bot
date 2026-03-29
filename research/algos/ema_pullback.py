"""EMA pullback — enter on retracement to fast EMA in trending market."""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from research.algos.base import BaseAlgo, AlgoSignal


class EMAPullbackAlgo(BaseAlgo):
    """
    Trend confirmed by price above/below slow EMA.
    Enter when price pulls back to touch fast EMA and bounces.

    Long:  close > EMA200, close touches EMA20 (within 0.3×ATR), then bounces up.
    Short: close < EMA200, close touches EMA20, then bounces down.
    SL: just beyond EMA50.  TP: 2× risk.

    Works well for: trending instruments (NAS100, BTC, strong trending FX).
    """
    name = "ema_pullback"

    def __init__(
        self,
        fast: int = 20, slow: int = 200, trend_ema: int = 50,
        touch_atr_mult: float = 0.5, sl_mult: float = 1.0, tp_mult: float = 2.0,
    ):
        self.fast           = fast
        self.slow           = slow
        self.trend_ema      = trend_ema
        self.touch_mult     = touch_atr_mult
        self.sl_mult        = sl_mult
        self.tp_mult        = tp_mult

    def generate(self, m15_df: pd.DataFrame, h1_df: pd.DataFrame, idx: int) -> Optional[AlgoSignal]:
        if idx < self.slow + 5:
            return None

        close    = m15_df["close"].astype(float)
        ema_f    = self.ema(close, self.fast)
        ema_t    = self.ema(close, self.trend_ema)
        ema_s    = self.ema(close, self.slow)
        atr_val  = float(self.atr(m15_df).iloc[idx])
        c_curr   = float(close.iloc[idx])
        c_prev   = float(close.iloc[idx - 1])
        ef_curr  = float(ema_f.iloc[idx])
        et_curr  = float(ema_t.iloc[idx])
        es_curr  = float(ema_s.iloc[idx])

        if np.isnan(es_curr) or atr_val <= 0:
            return None

        touch_zone = self.touch_mult * atr_val

        # Uptrend: close > EMA200, price touches EMA20 from above, bounces
        if c_curr > es_curr and abs(c_curr - ef_curr) < touch_zone and c_curr > c_prev:
            direction = "long"
            entry = c_curr
            sl    = et_curr - self.sl_mult * atr_val
            tp    = entry + self.tp_mult * abs(entry - sl)

        # Downtrend: close < EMA200, price touches EMA20 from below, bounces
        elif c_curr < es_curr and abs(c_curr - ef_curr) < touch_zone and c_curr < c_prev:
            direction = "short"
            entry = c_curr
            sl    = et_curr + self.sl_mult * atr_val
            tp    = entry - self.tp_mult * abs(entry - sl)

        else:
            return None

        risk   = abs(entry - sl)
        reward = abs(tp - entry)
        if risk <= 0:
            return None
        return AlgoSignal(direction, entry, sl, tp, round(reward / risk, 2))
