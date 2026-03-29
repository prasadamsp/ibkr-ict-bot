"""
eurusd.py — EURUSD Instrument Strategy

Strategy: Kill-Zone Mean Reversion (London open 07:00–10:00 UTC)

ICT structure-based signals on EURUSD failed (validation Sharpe -30.5)
because ECB policy and US yield differentials dominate price action,
obscuring FVG / Order Block structure.

Kill-zone mean reversion exploits a well-documented intraday pattern:
at the London open, institutional desks close overnight positions, driving
RSI to extremes. Price typically reverts to neutral RSI (≈50) by 10:00 UTC.

Entry rules:
  Window:    07:00–10:00 UTC only (London kill zone)
  Long:      RSI(14) < 35 on latest M15 close
  Short:     RSI(14) > 65 on latest M15 close
  TP:        1.5 × ATR (RSI reversion target; ~ RSI returning to 50)
  SL:        1.0 × ATR
  Min RR:    1.5
  Macro gate: H1 EMA20 slope — only trade in the direction of the slope
              (slope up → allow longs only; slope down → allow shorts only;
              flat → allow both)

Summer low-vol gate (Jul–Sep): require RSI extreme ≥ 30/70 (tighter threshold).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from strategy.instruments.base import BaseInstrumentStrategy
from strategy.strategy import TradeSignal
from strategy.structure import compute_rsi

_log = logging.getLogger("strategy")

_LONDON_KZ_START = 7
_LONDON_KZ_END   = 10

_RSI_PERIOD      = 14
_RSI_LONG_MAX    = 35.0     # standard threshold
_RSI_SHORT_MIN   = 65.0
_RSI_LONG_MAX_SUMMER  = 30.0    # tighter in Jul–Sep
_RSI_SHORT_MIN_SUMMER = 70.0

_SL_ATR_MULT  = 1.0
_TP_ATR_MULT  = 1.5
_MIN_RR       = 1.5

_LOW_VOL_MONTHS = {7, 8, 9}

# Macro gate slope threshold: EMA change > this fraction of price is "trending"
_SLOPE_THRESHOLD = 0.00005   # 0.005% per bar = ~0.5 pips on EUR/USD


class EURUSDStrategy(BaseInstrumentStrategy):
    """
    London Kill-Zone Mean Reversion strategy for EURUSD.

    Does NOT use ICT (FVG/OB/BOS) — pure RSI mean reversion within
    the London kill-zone window, filtered by H1 EMA20 macro direction.
    """

    def generate_signal(
        self,
        m15_df:     pd.DataFrame,
        h1_df:      pd.DataFrame,
        current_dt: datetime,
        d1_df:      Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> Optional[TradeSignal]:
        try:
            if len(m15_df) < 30 or len(h1_df) < 25:
                return None

            return self._killzone_meanrev(m15_df, h1_df, current_dt)

        except Exception as exc:
            _log.error("EURUSD generate_signal error: %s", exc, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Kill-zone mean reversion core
    # ------------------------------------------------------------------

    def _killzone_meanrev(
        self,
        m15_df:     pd.DataFrame,
        h1_df:      pd.DataFrame,
        current_dt: datetime,
    ) -> Optional[TradeSignal]:

        # London kill-zone window only
        h = current_dt.hour
        if not (_LONDON_KZ_START <= h < _LONDON_KZ_END):
            return None

        close_m15 = m15_df["close"].astype(float)
        rsi       = compute_rsi(close_m15, _RSI_PERIOD)
        rsi_curr  = float(rsi.iloc[-1])

        if np.isnan(rsi_curr):
            return None

        # ATR for SL/TP sizing (use M15 ATR)
        regime = self.get_regime(m15_df)
        atr    = regime.atr
        if atr <= 0:
            return None

        # Summer low-vol: tighter RSI thresholds
        month = current_dt.month
        rsi_long_max   = _RSI_LONG_MAX_SUMMER  if month in _LOW_VOL_MONTHS else _RSI_LONG_MAX
        rsi_short_min  = _RSI_SHORT_MIN_SUMMER if month in _LOW_VOL_MONTHS else _RSI_SHORT_MIN

        # Determine trade direction from RSI
        if rsi_curr < rsi_long_max:
            direction = "bullish"
        elif rsi_curr > rsi_short_min:
            direction = "bearish"
        else:
            return None

        # Macro alignment gate: H1 EMA20 slope
        h1_close   = h1_df["close"].astype(float)
        ema20      = h1_close.ewm(span=20, adjust=False).mean()
        slope      = float(ema20.iloc[-1]) - float(ema20.iloc[-2])
        norm_slope = slope / float(h1_close.iloc[-1]) if float(h1_close.iloc[-1]) != 0 else 0.0

        if norm_slope > _SLOPE_THRESHOLD and direction == "bearish":
            _log.debug("EURUSD: EMA20 slope UP → skipping short (macro misaligned)")
            return None
        if norm_slope < -_SLOPE_THRESHOLD and direction == "bullish":
            _log.debug("EURUSD: EMA20 slope DOWN → skipping long (macro misaligned)")
            return None

        c_curr = float(close_m15.iloc[-1])
        if direction == "bullish":
            entry = round(c_curr, 5)
            sl    = round(entry - _SL_ATR_MULT * atr, 5)
            tp    = round(entry + _TP_ATR_MULT * atr, 5)
        else:
            entry = round(c_curr, 5)
            sl    = round(entry + _SL_ATR_MULT * atr, 5)
            tp    = round(entry - _TP_ATR_MULT * atr, 5)

        risk   = abs(entry - sl)
        reward = abs(tp - entry)
        if risk <= 0 or reward <= 0:
            return None
        rr = reward / risk
        if rr < _MIN_RR:
            return None

        size_mult   = self.get_size_multiplier(current_dt)
        season_note = self._seasonality.get_note(self.symbol, current_dt)

        return TradeSignal(
            symbol=self.symbol,
            direction=direction,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            rr_ratio=round(rr, 2),
            confluence_score=0.65,
            h1_bias=1 if direction == "bullish" else -1,
            bar_time=pd.Timestamp(current_dt),
            entry_type="market",
            notes=(
                f"KZ_MeanRev | rsi={rsi_curr:.1f} | slope={norm_slope:.6f} | "
                f"size_mult={size_mult:.2f} | {season_note}"
            ),
        )
