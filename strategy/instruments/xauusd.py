"""
xauusd.py — XAUUSD (Gold) Instrument Strategy

Strategy logic overview
-----------------------
Gold is a macro-driven, seasonally strong instrument with well-defined ICT
footprints.  This module combines four filters before issuing a signal:

1. Regime gate (ADX > 20)
   Gold trending markets produce clean ICT setups; ranging / dead markets
   produce too many false fills.  We require ADX(14) on H1 to be above 20
   before any signal is generated.

2. Seasonal direction bias
   Historical seasonality: Sep–Feb is the gold bull window (central bank
   accumulation, year-end flows).  During this window we only take LONG Order
   Block entries — shorts are skipped.  Outside the window (Mar–Aug) both
   directions are permitted.

3. H4 bias via H1 proxy
   True H4 data is not always available via the streaming feed.  We approximate
   the H4 trend by inspecting the last 4 closed H1 bars: if the mean of their
   closes is above the current close we call the H4 bearish, and vice-versa.
   This gives a coarse but reliable directional filter.

4. Session filter
   Entries are only taken during London (07:00–11:00 UTC) or New York
   (12:00–17:00 UTC) sessions, where gold has the highest volume and the
   tightest spreads.

Entry mechanism
---------------
- Primary: M15 Order Block (detect_order_blocks) in the direction of the H1
  bias, with price pulling back into the OB zone.
- Secondary: M15 Fair Value Gap (detect_fvgs) if no OB is in range.
- Limit order at OB/FVG midpoint.

Risk / Reward
-------------
- Stop: below the OB extreme (or FVG boundary) with a 0.02% buffer.
- Target: nearest confirmed M15 swing high (for longs) / swing low (for shorts).
- Minimum 2R required; signal is discarded if RR < 2.0.

Seasonality multiplier
----------------------
The size_multiplier from SeasonalityCalendar is stored in the signal's notes
field.  The router applies the multiplier to the position size; the signal
itself just communicates the raw trade levels.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from strategy.fvg import detect_fvgs, get_nearest_fvg
from strategy.instruments.base import BaseInstrumentStrategy
from strategy.order_blocks import detect_order_blocks, get_nearest_ob
from strategy.sessions import get_session, is_tradeable_session
from strategy.strategy import TradeSignal
from strategy.structure import find_swing_highs, find_swing_lows

_log = logging.getLogger("strategy")

_BULL_MONTHS = {9, 10, 11, 12, 1, 2}   # seasonal long-only window
_MIN_ADX = 20.0
_MIN_RR = 2.0
_SL_BUFFER = 0.0002   # 0.02% buffer beyond OB/FVG boundary


class XAUUSDStrategy(BaseInstrumentStrategy):
    """ICT strategy for XAUUSD (Gold CFD).

    See module docstring for full strategy description.
    """

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        m15_df: pd.DataFrame,
        h1_df: pd.DataFrame,
        current_dt: datetime,
    ) -> Optional[TradeSignal]:
        """
        Analyse M15 and H1 data and return a TradeSignal for gold, or None.

        Parameters
        ----------
        m15_df : pd.DataFrame
            15-minute OHLCV bars, ascending datetime index, closed bars only.
        h1_df : pd.DataFrame
            1-hour OHLCV bars, ascending datetime index, closed bars only.
        current_dt : datetime
            Timestamp of the most recently closed M15 bar.

        Returns
        -------
        TradeSignal or None
        """
        try:
            return self._generate(m15_df, h1_df, current_dt)
        except Exception as exc:
            _log.error("XAUUSD generate_signal error: %s", exc, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _generate(
        self,
        m15_df: pd.DataFrame,
        h1_df: pd.DataFrame,
        current_dt: datetime,
    ) -> Optional[TradeSignal]:

        # --- Minimum bar guard ---
        if len(m15_df) < 50 or len(h1_df) < 30:
            _log.debug("XAUUSD: insufficient bars (m15=%d h1=%d)", len(m15_df), len(h1_df))
            return None

        # --- Regime gate: require ADX > 20 on H1 ---
        regime = self.get_regime(h1_df)
        if regime.adx < _MIN_ADX:
            _log.debug("XAUUSD: ADX %.1f < %.1f — dead market, skipping", regime.adx, _MIN_ADX)
            return None

        # --- Session filter: London or NY only ---
        session = get_session(pd.Timestamp(current_dt))
        if session not in {"london_kill", "london", "london_close", "ny_kill", "ny"}:
            _log.debug("XAUUSD: outside London/NY session (%s)", session)
            return None

        # --- H1 bias (trend direction) ---
        h1_bias = self._h1_bias(h1_df)  # 1 = bullish, -1 = bearish, 0 = neutral
        if h1_bias == 0:
            _log.debug("XAUUSD: H1 bias neutral, skipping")
            return None
        direction = "bullish" if h1_bias == 1 else "bearish"

        # --- Seasonal direction filter ---
        month = current_dt.month
        if month in _BULL_MONTHS and direction == "bearish":
            _log.debug("XAUUSD: seasonal bull window (month=%d), skipping short", month)
            return None

        current_price = float(m15_df.iloc[-1]["close"])

        # --- Order Block detection (primary entry) ---
        obs = detect_order_blocks(m15_df, lookback=5, max_age_bars=50)
        nearest_ob = get_nearest_ob(obs, direction, current_price, max_distance_pct=0.005)

        # --- FVG detection (secondary entry, used when no OB in range) ---
        fvgs = detect_fvgs(m15_df, min_size_pct=0.0002, max_age_bars=50)
        nearest_fvg = get_nearest_fvg(fvgs, direction, current_price, max_distance_pct=0.005)

        if nearest_ob is None and nearest_fvg is None:
            _log.debug("XAUUSD: no OB or FVG in range, skipping")
            return None

        # --- Build entry, SL, TP levels ---
        entry, sl, tp = self._calc_levels(direction, current_price, nearest_ob, nearest_fvg, m15_df)
        if entry is None:
            return None

        risk = abs(entry - sl)
        reward = abs(tp - entry)
        if risk <= 0:
            return None
        rr = reward / risk
        if rr < _MIN_RR:
            _log.debug("XAUUSD: RR %.2f < %.1f minimum", rr, _MIN_RR)
            return None

        # --- Seasonality size multiplier (stored in notes for router) ---
        size_mult = self.get_size_multiplier(current_dt)
        season_note = self._seasonality.get_note(self.symbol, current_dt)

        # --- Confluence score ---
        score_ob  = 1.0 if nearest_ob  else 0.0
        score_fvg = 1.0 if nearest_fvg else 0.0
        score_regime = min(regime.adx / 50.0, 1.0)   # normalise ADX to [0,1]
        from strategy.sessions import session_score
        score_session = session_score(pd.Timestamp(current_dt))
        confluence = round(
            0.35 * score_ob + 0.25 * score_fvg + 0.20 * score_regime + 0.20 * score_session,
            3,
        )

        return TradeSignal(
            symbol=self.symbol,
            direction=direction,
            entry_price=round(entry, 5),
            stop_loss=round(sl, 5),
            take_profit=round(tp, 5),
            rr_ratio=round(rr, 2),
            score_fvg=score_fvg,
            score_order_block=score_ob,
            score_session=score_session,
            confluence_score=confluence,
            h1_bias=h1_bias,
            fvg_ref=nearest_fvg,
            ob_ref=nearest_ob,
            bar_time=pd.Timestamp(current_dt),
            entry_type="limit",
            notes=(
                f"size_mult={size_mult:.2f} | {season_note} | "
                f"adx={regime.adx:.1f} | session={session}"
            ),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _h1_bias(self, h1_df: pd.DataFrame) -> int:
        """
        Derive H1 trend bias from EMA crossover.

        EMA(21) > EMA(55) on H1 → bullish (+1)
        EMA(21) < EMA(55) on H1 → bearish (-1)
        Otherwise                → neutral (0)
        """
        closes = h1_df["close"].astype(float)
        ema21 = float(closes.ewm(span=21, adjust=False).mean().iloc[-1])
        ema55 = float(closes.ewm(span=55, adjust=False).mean().iloc[-1])
        if ema21 > ema55 * 1.0001:
            return 1
        if ema21 < ema55 * 0.9999:
            return -1
        return 0

    def _h4_bias_from_h1(self, h1_df: pd.DataFrame) -> int:
        """
        Approximate H4 trend using the last 4 closed H1 bars.

        If the close of the last H1 bar is above the mean close of those 4
        bars, the H4 proxy is bullish; below the mean → bearish.
        """
        if len(h1_df) < 4:
            return 0
        last4 = h1_df.iloc[-4:]
        mean_close = float(last4["close"].mean())
        last_close = float(h1_df.iloc[-1]["close"])
        if last_close > mean_close * 1.0005:
            return 1
        if last_close < mean_close * 0.9995:
            return -1
        return 0

    def _calc_levels(
        self,
        direction: str,
        current_price: float,
        ob,
        fvg,
        df: pd.DataFrame,
    ):
        """Calculate entry, stop-loss, and take-profit."""
        # Prefer OB; fall back to FVG
        if ob is not None:
            entry = (ob.top + ob.bottom) / 2.0
            if direction == "bullish":
                sl = ob.ob_extreme * (1.0 - _SL_BUFFER)
            else:
                sl = ob.ob_extreme * (1.0 + _SL_BUFFER)
        else:
            entry = (fvg.top + fvg.bottom) / 2.0
            if direction == "bullish":
                sl = fvg.bottom * (1.0 - _SL_BUFFER)
            else:
                sl = fvg.top * (1.0 + _SL_BUFFER)

        # Take-profit: nearest swing in opposing direction
        if direction == "bullish":
            sh_mask = find_swing_highs(df, lookback=5)
            sh_prices = df.loc[sh_mask, "high"]
            above = sh_prices[sh_prices > entry]
            if above.empty:
                return None, None, None
            tp = float(above.iloc[0])
        else:
            sl_mask = find_swing_lows(df, lookback=5)
            sl_prices = df.loc[sl_mask, "low"]
            below = sl_prices[sl_prices < entry]
            if below.empty:
                return None, None, None
            tp = float(below.iloc[-1])

        return entry, sl, tp
