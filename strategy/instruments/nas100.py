"""
nas100.py — NAS100 (NASDAQ-100) Instrument Strategy

Strategy logic overview
-----------------------
The NASDAQ-100 is a momentum-driven equity index with strong trend-following
characteristics.  This module applies three contextual filters before an
entry signal is generated:

1. Macro regime filter (EMA 200 on H1)
   Price above H1 EMA(200) → long bias only.
   Price below H1 EMA(200) → both directions permitted, but size multiplier
   is set to 50% (stored in notes for the router).

2. VIX proxy — panic regime guard
   If ATR% on H1 > 2.0% (regime.atr_pct > 0.02), the market is in a panic /
   liquidity-crisis state.  Signals are suspended until volatility normalises.
   This is a binary cut, not a size reduction.

3. August filter
   August is historically the lowest-liquidity month for US equities
   ("summer doldrums").  The minimum confluence threshold is raised from the
   default to 0.75, making it harder for a setup to qualify.

Entry mechanism
---------------
- Wait for a Break of Structure (BOS) confirmation on H1 (via detect_bos).
- On M15, look for an unfilled Fair Value Gap in the direction of the H1 BOS.
- Momentum confirmation: the last 3 M15 closes must all move in the same
  direction as the intended trade before the entry limit is placed.
- Limit order at the FVG midpoint.

Risk / Reward
-------------
- Stop: beyond the FVG boundary (0.02% buffer).
- Target: nearest M15 swing high (long) or swing low (short).
- Minimum 2R; signals below this threshold are discarded.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from strategy.fvg import detect_fvgs, get_nearest_fvg
from strategy.instruments.base import BaseInstrumentStrategy
from strategy.sessions import get_session, session_score
from strategy.strategy import TradeSignal
from strategy.structure import detect_bos, find_swing_highs, find_swing_lows

_log = logging.getLogger("strategy")

_VIX_PROXY_ATR_PCT = 0.02     # H1 ATR% above this = panic, skip
_AUGUST_MIN_CONFLUENCE = 0.75
_DEFAULT_MIN_CONFLUENCE = 0.55
_MIN_RR = 2.0
_SL_BUFFER = 0.0002


class NAS100Strategy(BaseInstrumentStrategy):
    """ICT / EMA strategy for NAS100 (NASDAQ-100 CFD).

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
            _log.error("NAS100 generate_signal error: %s", exc, exc_info=True)
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
            _log.debug("NAS100: insufficient bars (m15=%d h1=%d)", len(m15_df), len(h1_df))
            return None

        # --- VIX proxy: panic guard ---
        regime = self.get_regime(h1_df)
        if regime.atr_pct > _VIX_PROXY_ATR_PCT:
            _log.debug(
                "NAS100: ATR%% %.4f > %.4f panic threshold, skipping",
                regime.atr_pct, _VIX_PROXY_ATR_PCT,
            )
            return None

        current_price = float(m15_df.iloc[-1]["close"])
        h1_closes = h1_df["close"].astype(float)

        # --- Macro regime: EMA(200) on H1 ---
        ema200 = float(h1_closes.ewm(span=200, adjust=False).mean().iloc[-1])
        above_ema200 = current_price > ema200
        size_note = ""
        if above_ema200:
            allowed_directions = {"bullish"}
        else:
            allowed_directions = {"bullish", "bearish"}
            size_note = "size_50pct=True"

        # --- H1 BOS confirmation ---
        bos_df = detect_bos(h1_df, lookback=5)
        bullish_bos = bool(bos_df["bos_bullish"].iloc[-5:].any())
        bearish_bos = bool(bos_df["bos_bearish"].iloc[-5:].any())

        if bullish_bos and "bullish" in allowed_directions:
            direction = "bullish"
        elif bearish_bos and "bearish" in allowed_directions:
            direction = "bearish"
        else:
            _log.debug(
                "NAS100: no qualifying BOS in allowed directions %s", allowed_directions
            )
            return None

        # --- Session filter ---
        session = get_session(pd.Timestamp(current_dt))
        if session not in {"london_kill", "london", "ny_kill", "ny", "london_close"}:
            _log.debug("NAS100: outside tradeable session (%s)", session)
            return None

        # --- M15 FVG pullback ---
        fvgs = detect_fvgs(m15_df, min_size_pct=0.0003, max_age_bars=50)
        nearest_fvg = get_nearest_fvg(fvgs, direction, current_price, max_distance_pct=0.006)
        if nearest_fvg is None:
            _log.debug("NAS100: no qualifying FVG in range")
            return None

        # --- Momentum confirmation: last 3 M15 closes all in same direction ---
        if not self._momentum_aligned(m15_df, direction):
            _log.debug("NAS100: last 3 M15 closes not aligned with direction %s", direction)
            return None

        # --- Entry levels ---
        entry = (nearest_fvg.top + nearest_fvg.bottom) / 2.0
        if direction == "bullish":
            sl = nearest_fvg.bottom * (1.0 - _SL_BUFFER)
            sh_mask = find_swing_highs(m15_df, lookback=5)
            tp_candidates = m15_df.loc[sh_mask, "high"]
            tp_candidates = tp_candidates[tp_candidates > entry]
            if tp_candidates.empty:
                return None
            tp = float(tp_candidates.iloc[0])
        else:
            sl = nearest_fvg.top * (1.0 + _SL_BUFFER)
            sl_mask = find_swing_lows(m15_df, lookback=5)
            tp_candidates = m15_df.loc[sl_mask, "low"]
            tp_candidates = tp_candidates[tp_candidates < entry]
            if tp_candidates.empty:
                return None
            tp = float(tp_candidates.iloc[-1])

        risk = abs(entry - sl)
        reward = abs(tp - entry)
        if risk <= 0:
            return None
        rr = reward / risk
        if rr < _MIN_RR:
            _log.debug("NAS100: RR %.2f < %.1f minimum", rr, _MIN_RR)
            return None

        # --- Confluence score ---
        score_bos = 1.0
        score_fvg = 1.0
        score_session = session_score(pd.Timestamp(current_dt))
        score_regime = 0.8 if above_ema200 else 0.5
        confluence = round(
            0.30 * score_bos + 0.35 * score_fvg + 0.20 * score_session + 0.15 * score_regime,
            3,
        )

        # --- August filter: raise confluence threshold ---
        month = current_dt.month
        min_confluence = _AUGUST_MIN_CONFLUENCE if month == 8 else _DEFAULT_MIN_CONFLUENCE
        if confluence < min_confluence:
            _log.debug(
                "NAS100: confluence %.3f below threshold %.3f (month=%d)",
                confluence, min_confluence, month,
            )
            return None

        season_mult = self.get_size_multiplier(current_dt)
        notes_parts = [f"size_mult={season_mult:.2f}"]
        if size_note:
            notes_parts.append(size_note)
        notes_parts.append(f"ema200={'above' if above_ema200 else 'below'}")
        notes_parts.append(f"session={session}")

        return TradeSignal(
            symbol=self.symbol,
            direction=direction,
            entry_price=round(entry, 2),
            stop_loss=round(sl, 2),
            take_profit=round(tp, 2),
            rr_ratio=round(rr, 2),
            score_bos=score_bos,
            score_fvg=score_fvg,
            score_session=score_session,
            confluence_score=confluence,
            h1_bias=1 if direction == "bullish" else -1,
            fvg_ref=nearest_fvg,
            bar_time=pd.Timestamp(current_dt),
            entry_type="limit",
            notes=" | ".join(notes_parts),
        )

    # ------------------------------------------------------------------

    def _momentum_aligned(self, m15_df: pd.DataFrame, direction: str) -> bool:
        """Return True if the last 3 M15 closes all move in the direction."""
        if len(m15_df) < 4:
            return False
        closes = m15_df["close"].values
        c1, c2, c3 = closes[-3], closes[-2], closes[-1]
        if direction == "bullish":
            return c2 > c1 and c3 > c2
        else:
            return c2 < c1 and c3 < c2
