"""
base.py — BaseAlgo

All research algorithms implement this interface. The grid search engine
calls generate() on every bar and scores the resulting trades.

A "signal" is a simple namedtuple — no dependency on TradeSignal or the
live strategy stack.  This keeps the research layer self-contained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class AlgoSignal:
    direction: str        # "long" or "short"
    entry:     float
    sl:        float
    tp:        float
    rr:        float      # reward/risk ratio


class BaseAlgo:
    """
    Abstract base for all research algorithms.

    Subclasses implement generate(m15_df, h1_df, idx) where idx is the
    current bar index into m15_df.  Return AlgoSignal or None.
    """

    name: str = "base"

    def generate(
        self,
        m15_df: pd.DataFrame,
        h1_df:  pd.DataFrame,
        idx:    int,
    ) -> Optional[AlgoSignal]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        close = df["close"].astype(float)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1.0 / period, adjust=False).mean()

    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta    = close.diff()
        avg_gain = delta.clip(lower=0).ewm(alpha=1.0 / period, adjust=False).mean()
        avg_loss = (-delta).clip(lower=0).ewm(alpha=1.0 / period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()
