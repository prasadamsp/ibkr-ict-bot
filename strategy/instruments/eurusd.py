"""
eurusd.py — EURUSD Instrument Strategy

Strategy logic overview
-----------------------
EURUSD is primarily traded through the Kill Zone liquidity-sweep pattern.
The strategy focuses on two high-probability windows each day:

  London Kill Zone:   07:00–10:00 UTC
  New York Kill Zone: 12:00–15:00 UTC

Outside these windows, no signals are generated.

Core pattern — Asian Range Sweep + Reversal
-------------------------------------------
1. Calculate the Asian session range (23:00–07:00 UTC prior day).
2. During the Kill Zone, look for a candle wick that sweeps THROUGH one of
   the Asian range extremes (stop hunt).
3. After the sweep, if an M15 Order Block or Fair Value Gap forms in the
   OPPOSITE direction, enter the snap-back trade.
4. Stop: beyond the sweep wick extreme (with a small buffer).
5. Target: opposite Asian range level, or nearest session liquidity pool
   (previous session high/low as a proxy).

Seasonal adjustments
--------------------
  Q1 (Jan–Mar): short bias (USD strength post year-end rebalancing).
  Jul–Sep:      reduce size 40% (summer low-volatility period). This is
                communicated via size_mult in signal notes; the router
                applies it.

The strategy does not generate signals when outside the kill zones, keeping
the trade frequency low and quality high.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd

from strategy.fvg import detect_fvgs, get_nearest_fvg
from strategy.instruments.base import BaseInstrumentStrategy
from strategy.order_blocks import detect_order_blocks, get_nearest_ob
from strategy.sessions import get_asian_range, get_session, session_score
from strategy.strategy import TradeSignal
from strategy.structure import find_swing_highs, find_swing_lows

_log = logging.getLogger("strategy")

# Kill zone windows (UTC hour bounds, inclusive start exclusive end)
_LONDON_KZ_START = 7
_LONDON_KZ_END   = 10
_NY_KZ_START     = 12
_NY_KZ_END       = 15

# Months with short bias (USD strong)
_SHORT_BIAS_MONTHS = {1, 2, 3}
# Months with reduced size (low vol)
_LOW_VOL_MONTHS = {7, 8, 9}
_LOW_VOL_SIZE_MULT = 0.6   # override: apply 40% size reduction

_MIN_RR = 2.0
_SL_BUFFER = 0.0002   # 0.02% beyond sweep extreme
_SWEEP_TOLERANCE = 0.0001   # wick must pierce extreme by at least 1 pip


class EURUSDStrategy(BaseInstrumentStrategy):
    """Kill-zone liquidity-sweep strategy for EURUSD.

    See module docstring for full strategy description.
    """

    def generate_signal(
        self,
        m15_df: pd.DataFrame,
        h1_df: pd.DataFrame,
        current_dt: datetime,
    ) -> Optional[TradeSignal]:
        try:
            return self._generate(m15_df, h1_df, current_dt)
        except Exception as exc:
            _log.error("EURUSD generate_signal error: %s", exc, exc_info=True)
            return None

    # ------------------------------------------------------------------

    def _generate(
        self,
        m15_df: pd.DataFrame,
        h1_df: pd.DataFrame,
        current_dt: datetime,
    ) -> Optional[TradeSignal]:

        # --- Minimum bar guard ---
        if len(m15_df) < 50 or len(h1_df) < 30:
            _log.debug("EURUSD: insufficient bars (m15=%d h1=%d)", len(m15_df), len(h1_df))
            return None

        # --- Kill Zone filter ---
        if not self._in_kill_zone(current_dt):
            _log.debug("EURUSD: outside kill zone at %s", current_dt)
            return None

        # --- Determine allowed directions from seasonal bias ---
        month = current_dt.month
        if month in _SHORT_BIAS_MONTHS:
            allowed_directions = {"bearish"}
        else:
            allowed_directions = {"bullish", "bearish"}

        # --- Asian range ---
        asian_range = get_asian_range(m15_df)
        if asian_range is None:
            _log.debug("EURUSD: no Asian range available")
            return None
        asian_high, asian_low = asian_range

        current_price = float(m15_df.iloc[-1]["close"])

        # --- Detect Asian range sweep in recent M15 bars ---
        sweep_direction, sweep_extreme = self._detect_asian_sweep(
            m15_df, asian_high, asian_low
        )
        if sweep_direction is None:
            _log.debug("EURUSD: no Asian range sweep detected")
            return None

        # Entry direction is opposite to the sweep direction
        direction = "bullish" if sweep_direction == "bearish_sweep" else "bearish"
        if direction not in allowed_directions:
            _log.debug(
                "EURUSD: direction %s not in allowed directions %s (seasonal)",
                direction, allowed_directions,
            )
            return None

        # --- M15 OB or FVG after sweep ---
        obs = detect_order_blocks(m15_df, lookback=5, max_age_bars=30)
        nearest_ob = get_nearest_ob(obs, direction, current_price, max_distance_pct=0.005)

        fvgs = detect_fvgs(m15_df, min_size_pct=0.0001, max_age_bars=30)
        nearest_fvg = get_nearest_fvg(fvgs, direction, current_price, max_distance_pct=0.005)

        if nearest_ob is None and nearest_fvg is None:
            _log.debug("EURUSD: no OB or FVG after sweep")
            return None

        # --- Entry / SL / TP levels ---
        entry, sl, tp = self._calc_levels(
            direction, current_price, nearest_ob, nearest_fvg,
            sweep_extreme, asian_high, asian_low, m15_df
        )
        if entry is None:
            return None

        risk = abs(entry - sl)
        reward = abs(tp - entry)
        if risk <= 0:
            return None
        rr = reward / risk
        if rr < _MIN_RR:
            _log.debug("EURUSD: RR %.2f < %.1f minimum", rr, _MIN_RR)
            return None

        # --- Seasonality size multiplier ---
        if month in _LOW_VOL_MONTHS:
            size_mult = _LOW_VOL_SIZE_MULT
            season_note = "low_vol_summer size_60pct"
        else:
            size_mult = self.get_size_multiplier(current_dt)
            season_note = self._seasonality.get_note(self.symbol, current_dt)

        # --- Confluence score ---
        score_ob  = 1.0 if nearest_ob  else 0.0
        score_fvg = 1.0 if nearest_fvg else 0.0
        score_session = session_score(pd.Timestamp(current_dt))
        # Asian sweep is a strong ICT signal
        score_sweep = 1.0
        confluence = round(
            0.30 * score_sweep + 0.25 * score_ob + 0.25 * score_fvg + 0.20 * score_session,
            3,
        )

        kz_label = "london_kz" if self._in_london_kz(current_dt) else "ny_kz"
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
            h1_bias=1 if direction == "bullish" else -1,
            fvg_ref=nearest_fvg,
            ob_ref=nearest_ob,
            bar_time=pd.Timestamp(current_dt),
            entry_type="limit",
            notes=(
                f"size_mult={size_mult:.2f} | {season_note} | "
                f"sweep={sweep_direction} at {sweep_extreme:.5f} | {kz_label}"
            ),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _in_kill_zone(self, dt: datetime) -> bool:
        """Return True if dt falls in London KZ (07–10) or NY KZ (12–15) UTC."""
        h = dt.hour
        return (
            (_LONDON_KZ_START <= h < _LONDON_KZ_END) or
            (_NY_KZ_START <= h < _NY_KZ_END)
        )

    def _in_london_kz(self, dt: datetime) -> bool:
        return _LONDON_KZ_START <= dt.hour < _LONDON_KZ_END

    def _detect_asian_sweep(
        self,
        m15_df: pd.DataFrame,
        asian_high: float,
        asian_low: float,
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Scan the last 8 M15 bars for a wick that sweeps through an Asian extreme.

        Returns
        -------
        (sweep_type, sweep_extreme) or (None, None)
          sweep_type: "bullish_sweep" (wick above asian_high) or
                      "bearish_sweep" (wick below asian_low)
          sweep_extreme: the wick high or wick low that pierced the level
        """
        lookback = min(8, len(m15_df))
        recent = m15_df.iloc[-lookback:]

        for _, bar in recent.iterrows():
            if bar["high"] > asian_high + _SWEEP_TOLERANCE * asian_high:
                return "bullish_sweep", float(bar["high"])
            if bar["low"] < asian_low - _SWEEP_TOLERANCE * asian_low:
                return "bearish_sweep", float(bar["low"])

        return None, None

    def _calc_levels(
        self,
        direction: str,
        current_price: float,
        ob,
        fvg,
        sweep_extreme: float,
        asian_high: float,
        asian_low: float,
        df: pd.DataFrame,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculate entry, SL (beyond sweep wick), TP (opposite Asian level)."""
        # Entry from OB or FVG
        if ob is not None:
            entry = (ob.top + ob.bottom) / 2.0
        else:
            entry = (fvg.top + fvg.bottom) / 2.0

        # Stop beyond the sweep extreme
        if direction == "bullish":
            # Sweep was below asian_low; stop beyond the wick low
            sl = sweep_extreme * (1.0 - _SL_BUFFER)
            # Target: asian_high, or nearest swing high above entry
            if asian_high > entry:
                tp = asian_high
            else:
                sh_mask = find_swing_highs(df, lookback=5)
                candidates = df.loc[sh_mask, "high"]
                above = candidates[candidates > entry]
                if above.empty:
                    return None, None, None
                tp = float(above.iloc[0])
        else:
            # Sweep was above asian_high; stop beyond the wick high
            sl = sweep_extreme * (1.0 + _SL_BUFFER)
            # Target: asian_low, or nearest swing low below entry
            if asian_low < entry:
                tp = asian_low
            else:
                sl_mask = find_swing_lows(df, lookback=5)
                candidates = df.loc[sl_mask, "low"]
                below = candidates[candidates < entry]
                if below.empty:
                    return None, None, None
                tp = float(below.iloc[-1])

        return entry, sl, tp
