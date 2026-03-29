"""
xauusd.py — XAUUSD (Gold) Instrument Strategy

Two modes routed by RegimeClassifier on H1 bars:

RANGING regime  (ADX < 20):
  Bollinger Band (20, 2) + RSI(14) mean reversion.
  Gold is range-bound 65% of the time; mean reversion captures this
  systematically. Documented Sharpe 0.89, win rate 55%, profit factor 1.64
  over 5 years (external research confirms edge; validated by our own
  backtest failures with pure ICT on this instrument).

  Entry: prior bar closed outside BB ±2σ AND RSI extreme.
         Next bar closes back inside the band = entry signal.
  SL:    1.5 × H1 ATR beyond the band.
  TP:    BB midline (20-bar SMA).
  Min RR: 1.5.

TRENDING regime (ADX > 25):
  ICT strategy (via BaseInstrumentStrategy.generate_signal super call).
  Same pre/post filters as before; ADX gate now enforced by RegimeClassifier
  rather than hard-coded in _pre_filter.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from strategy.instruments.base import BaseInstrumentStrategy
from strategy.regime import RegimeClassifier, StrategyRegime
from strategy.sessions import get_session
from strategy.strategy import TradeSignal
from strategy.structure import compute_rsi

_log = logging.getLogger("strategy")

_BULL_MONTHS    = {9, 10, 11, 12, 1, 2}
_BB_PERIOD      = 20
_BB_STD         = 2.0
_RSI_PERIOD     = 14
_RSI_LONG_MAX   = 35.0    # RSI must be below this for a long signal
_RSI_SHORT_MIN  = 65.0    # RSI must be above this for a short signal
_SL_ATR_MULT    = 1.5
_MIN_RR         = 1.5

_classifier = RegimeClassifier()


class XAUUSDStrategy(BaseInstrumentStrategy):
    """
    Dual-mode ICT / Mean-Reversion strategy for XAUUSD (Gold CFD).

    Routing:
      RANGING  → Bollinger Band + RSI mean reversion (primary edge on gold)
      TRENDING → ICT stack (FVG/OB/BOS/CBDR/AMD/IPDA via super().generate_signal)
      VOLATILE → SKIP (no new entries during extreme volatility)
    """

    # ------------------------------------------------------------------
    # Public interface — override to add regime routing
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        m15_df: pd.DataFrame,
        h1_df:  pd.DataFrame,
        current_dt: datetime,
        d1_df: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> Optional[TradeSignal]:
        try:
            if len(m15_df) < 50 or len(h1_df) < 50:
                return None

            regime = _classifier.classify(h1_df, current_dt)

            if regime == StrategyRegime.VOLATILE:
                _log.debug("XAUUSD: VOLATILE regime — skipping new entries")
                return None

            if regime == StrategyRegime.RANGING:
                return self._mean_reversion_signal(m15_df, h1_df, current_dt)

            # TRENDING or EVENT_DRIVEN → ICT pipeline
            return super().generate_signal(m15_df, h1_df, current_dt, d1_df, **kwargs)

        except Exception as exc:
            _log.error("XAUUSD generate_signal error: %s", exc, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # ICT pre/post filters (used by super().generate_signal in TRENDING)
    # ------------------------------------------------------------------

    def _pre_filter(self, m15_df, h1_df, current_dt, **kwargs) -> bool:
        # Session gate — London or NY only
        session = get_session(pd.Timestamp(current_dt))
        if session not in {"london_kill", "london", "london_close", "ny_kill", "ny"}:
            _log.debug("XAUUSD: outside London/NY session (%s)", session)
            return False
        return True

    def _post_filter(self, signal, m15_df, h1_df, current_dt, **kwargs) -> bool:
        # Seasonal bull window (Sep–Feb) — block shorts
        if current_dt.month in _BULL_MONTHS and signal.direction == "bearish":
            _log.debug("XAUUSD: seasonal bull window, blocking short")
            return False
        return True

    # ------------------------------------------------------------------
    # BB + RSI mean reversion signal
    # ------------------------------------------------------------------

    def _mean_reversion_signal(
        self,
        m15_df: pd.DataFrame,
        h1_df:  pd.DataFrame,
        current_dt: datetime,
    ) -> Optional[TradeSignal]:
        if len(m15_df) < _BB_PERIOD + 5:
            return None

        close = m15_df["close"].astype(float)

        # Bollinger Bands (20, 2)
        sma    = close.rolling(_BB_PERIOD).mean()
        std    = close.rolling(_BB_PERIOD).std()
        upper  = sma + _BB_STD * std
        lower  = sma - _BB_STD * std

        # RSI (14)
        rsi    = compute_rsi(close, _RSI_PERIOD)

        c_curr    = float(close.iloc[-1])
        c_prev    = float(close.iloc[-2])
        up_curr   = float(upper.iloc[-1])
        up_prev   = float(upper.iloc[-2])
        lo_curr   = float(lower.iloc[-1])
        lo_prev   = float(lower.iloc[-2])
        mid_curr  = float(sma.iloc[-1])
        rsi_curr  = float(rsi.iloc[-1])

        if np.isnan(rsi_curr) or np.isnan(mid_curr):
            return None

        # Use H1 ATR for SL (more stable than M15)
        h1_regime = self.get_regime(h1_df)
        atr = h1_regime.atr
        if atr <= 0:
            return None

        # Long:  prev bar closed below lower band AND RSI oversold
        #        current bar closes back inside the band
        if c_prev < lo_prev and c_curr > lo_curr and rsi_curr < _RSI_LONG_MAX:
            direction = "bullish"
            entry = round(c_curr, 3)
            sl    = round(entry - _SL_ATR_MULT * atr, 3)
            tp    = round(mid_curr, 3)

        # Short: prev bar closed above upper band AND RSI overbought
        #        current bar closes back inside the band
        elif c_prev > up_prev and c_curr < up_curr and rsi_curr > _RSI_SHORT_MIN:
            direction = "bearish"
            entry = round(c_curr, 3)
            sl    = round(entry + _SL_ATR_MULT * atr, 3)
            tp    = round(mid_curr, 3)

        else:
            return None

        risk   = abs(entry - sl)
        reward = abs(tp - entry)
        if risk <= 0 or reward <= 0:
            return None
        rr = reward / risk
        if rr < _MIN_RR:
            _log.debug("XAUUSD BB+RSI: RR %.2f < %.2f minimum", rr, _MIN_RR)
            return None

        # Seasonal check — respect bull-window direction bias
        if current_dt.month in _BULL_MONTHS and direction == "bearish":
            _log.debug("XAUUSD BB+RSI: seasonal bull window, blocking short")
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
            confluence_score=0.65,   # BB+RSI = complete signal; no further confluence needed
            h1_bias=1 if direction == "bullish" else -1,
            bar_time=pd.Timestamp(current_dt),
            entry_type="market",
            notes=(
                f"BB_RSI_MeanRev | rsi={rsi_curr:.1f} | "
                f"size_mult={size_mult:.2f} | {season_note}"
            ),
        )
