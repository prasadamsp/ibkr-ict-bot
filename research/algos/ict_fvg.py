"""ICT Fair Value Gap — simplified version for research grid search."""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from research.algos.base import BaseAlgo, AlgoSignal


class ICTFVGAlgo(BaseAlgo):
    """
    Simplified ICT FVG:
      Bullish FVG: low[i] > high[i-2]  (gap between candle[i-2].high and candle[i].low)
      Bearish FVG: high[i] < low[i-2]

    Entry when price retests the FVG (closes inside it).
    SL: beyond FVG by 0.5× ATR.  TP: 2× risk.
    Max FVG age: 20 bars.

    Requires BOS confirmation from H1: if h1 last close direction != signal direction,
    skip (avoid counter-trend FVG entries).
    """
    name = "ict_fvg"

    def __init__(
        self,
        min_fvg_pct: float = 0.0002,  # FVG must be at least 0.02% of price
        max_age_bars: int = 20,
        sl_mult: float = 0.5,
        tp_mult: float = 2.0,
    ):
        self.min_fvg_pct  = min_fvg_pct
        self.max_age_bars = max_age_bars
        self.sl_mult      = sl_mult
        self.tp_mult      = tp_mult

    def generate(self, m15_df: pd.DataFrame, h1_df: pd.DataFrame, idx: int) -> Optional[AlgoSignal]:
        if idx < 30:
            return None

        high  = m15_df["high"].astype(float)
        low   = m15_df["low"].astype(float)
        close = m15_df["close"].astype(float)
        atr_s = self.atr(m15_df)
        c_curr  = float(close.iloc[idx])
        atr_val = float(atr_s.iloc[idx])

        if atr_val <= 0:
            return None

        # H1 bias from last H1 close
        h1_bias = None
        if h1_df is not None and len(h1_df) >= 2:
            h1_close = h1_df["close"].astype(float)
            if float(h1_close.iloc[-1]) > float(h1_close.iloc[-2]):
                h1_bias = "long"
            else:
                h1_bias = "short"

        # Search recent bars for an unfilled FVG
        start = max(3, idx - self.max_age_bars)
        for j in range(idx - 2, start, -1):
            fvg_high = float(high.iloc[j - 2])   # top of candle 2 bars ago
            fvg_low  = float(low.iloc[j])         # bottom of current formation

            # Bullish FVG: gap between high[j-2] and low[j]
            if fvg_low > fvg_high:
                gap_size = fvg_low - fvg_high
                mid_price = (fvg_low + fvg_high) / 2

                if gap_size / mid_price < self.min_fvg_pct:
                    continue
                # Price must be retesting: in the FVG zone
                if fvg_high <= c_curr <= fvg_low:
                    if h1_bias is None or h1_bias == "long":
                        direction = "long"
                        entry = c_curr
                        sl    = fvg_high - self.sl_mult * atr_val
                        tp    = entry + self.tp_mult * abs(entry - sl)
                        risk   = abs(entry - sl)
                        reward = abs(tp - entry)
                        if risk > 0:
                            return AlgoSignal(direction, entry, sl, tp, round(reward / risk, 2))

            # Bearish FVG: gap between low[j-2] and high[j]
            fvg_high2 = float(high.iloc[j])
            fvg_low2  = float(low.iloc[j - 2])

            if fvg_high2 < fvg_low2:
                gap_size = fvg_low2 - fvg_high2
                mid_price = (fvg_low2 + fvg_high2) / 2

                if gap_size / mid_price < self.min_fvg_pct:
                    continue
                if fvg_high2 <= c_curr <= fvg_low2:
                    if h1_bias is None or h1_bias == "short":
                        direction = "short"
                        entry = c_curr
                        sl    = fvg_low2 + self.sl_mult * atr_val
                        tp    = entry - self.tp_mult * abs(entry - sl)
                        risk   = abs(entry - sl)
                        reward = abs(tp - entry)
                        if risk > 0:
                            return AlgoSignal(direction, entry, sl, tp, round(reward / risk, 2))

        return None
