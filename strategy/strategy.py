"""
ICT Signal Engine — combines all ICT concepts into trade signals.

Signal generation logic:
─────────────────────────
1. H1 bias: determine trend direction (bullish/bearish) from H1 structure.
2. M15 execution:
   a. Wait for liquidity sweep in direction of H1 bias.
   b. Confirm BOS or MSS on M15 in the bias direction.
   c. Identify nearest unfilled FVG in the bias direction.
   d. Confirm price is in discount (for longs) or premium (for shorts).
   e. Check session — prefer kill zones.
3. Score confluence (0–1). Only signal if score ≥ min_confluence_score.
4. Calculate entry, SL, TP.

Entry calculation:
    Long:  Limit order at FVG midpoint. SL = below FVG bottom (or sweep low). TP = nearest swing high.
    Short: Limit order at FVG midpoint. SL = above FVG top (or sweep high).  TP = nearest swing low.

This module ONLY generates signals. It does NOT place orders.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config.settings import CONFIG, StrategyConfig
from strategy.breaker_blocks import BreakerBlock, detect_breakers, get_nearest_breaker
from strategy.daily_bias import get_daily_bias, weekly_bias, d1_pd_score
from strategy.fvg import FVG, detect_fvgs, get_nearest_fvg
from strategy.ipda import IPDARange, calculate_ipda_ranges, get_ipda_tp, ipda_bias
from strategy.liquidity import (
    LiquidityLevel, LiquiditySweep, detect_sweeps, find_liquidity_levels,
    get_recent_sweep,
)
from strategy.order_blocks import OrderBlock, detect_order_blocks, get_nearest_ob
from strategy.cbdr import CBDRState, get_cbdr_state, cbdr_confluence_score, get_cbdr_tp
from strategy.power_of_3 import AMDState, get_amd_state, amd_confluence_score
from research.macro_filters import macro_confluence_delta
from strategy.sessions import is_tradeable_session, session_score
from strategy.structure import (
    calculate_atr, detect_bos, detect_mss, get_trend_bias, is_displacement,
)
from strategy.zones import DealingRange, calculate_dealing_range, pd_score
from utils.logger import strategy_log


# ---------------------------------------------------------------------------
# Signal data structure
# ---------------------------------------------------------------------------

@dataclass
class TradeSignal:
    """A complete, actionable trade signal."""
    symbol: str
    direction: str          # "bullish" (long) or "bearish" (short)
    entry_price: float
    stop_loss: float
    take_profit: float
    rr_ratio: float

    # Confluence components (each 0–1)
    score_bos: float = 0.0
    score_fvg: float = 0.0
    score_order_block: float = 0.0
    score_liquidity: float = 0.0
    score_session: float = 0.0
    score_pd_zone: float = 0.0
    confluence_score: float = 0.0

    # Context
    h1_bias: int = 0        # 1 = bullish, -1 = bearish
    d1_bias: int = 0        # 1 = bullish, -1 = bearish (from D1 structure)
    fvg_ref: Optional[FVG] = None
    ob_ref: Optional[OrderBlock] = None
    breaker_ref: Optional[object] = None   # BreakerBlock if breaker entry
    sweep_ref: Optional[LiquiditySweep] = None
    bar_time: Optional[pd.Timestamp] = None
    entry_type: str = "limit"   # "limit" or "market"
    amd_phase: str = "unknown"
    ipda_tp_used: bool = False   # True when TP comes from IPDA level
    cbdr_profile: str = "unknown"  # Intraday profile: A_classic/B_reversal/C_inside/D_directional
    cbdr_tp_used: bool = False      # True when TP comes from CBDR projection
    notes: str = ""


# ---------------------------------------------------------------------------
# Strategy Engine
# ---------------------------------------------------------------------------

class ICTStrategy:
    """
    Multi-timeframe ICT strategy engine.

    Flow:
        on_bar(symbol, tf, df) → internally updates state
        When M15 bar closes → run signal logic → return TradeSignal or None
    """

    def __init__(self, cfg: StrategyConfig = None):
        self.cfg = cfg or CONFIG.strategy
        self._h1_bias: Dict[str, int] = {}       # per symbol
        self._last_signal: Dict[str, Optional[TradeSignal]] = {}

    # ------------------------------------------------------------------
    # Entry point — called by DataHandler on_bar_close
    # ------------------------------------------------------------------

    def on_bar(
        self,
        symbol: str,
        tf: str,
        m15_df: pd.DataFrame,
        h1_df: pd.DataFrame,
        d1_df: Optional[pd.DataFrame] = None,
    ) -> Optional[TradeSignal]:
        """
        Called when a new M15 bar closes.
        Returns a TradeSignal if conditions are met, else None.

        Always pass CLOSED bars only (exclude current forming bar).

        Args:
            d1_df: Optional daily bars — enables D1 bias + IPDA ranges.
                   Falls back to H1-only bias if not provided.
        """
        if tf != "M15":
            return None

        if len(m15_df) < 50 or len(h1_df) < 30:
            return None

        # 1. Determine H1 bias
        h1_bias = self._get_h1_bias(h1_df)
        self._h1_bias[symbol] = h1_bias

        if h1_bias == 0:
            strategy_log.debug(f"{symbol}: H1 bias neutral, skipping.")
            return None

        # 2. D1 bias gate — must align with H1 bias (or be neutral)
        d1_bias = get_daily_bias(d1_df) if d1_df is not None else 0
        if d1_bias != 0 and d1_bias != h1_bias:
            strategy_log.debug(
                f"{symbol}: D1 bias ({d1_bias:+d}) conflicts with "
                f"H1 bias ({h1_bias:+d}) — skipping."
            )
            return None

        # 3. Session filter
        current_time = m15_df.index[-1]
        if not is_tradeable_session(current_time):
            strategy_log.debug(f"{symbol}: Outside tradeable session at {current_time}")
            return None

        # 4. Power of 3 — AMD phase filter
        amd = get_amd_state(m15_df, current_time)
        if amd.phase == "dead":
            return None   # Never trade dead zone

        # 5. CBDR — dealing range context + intraday profile
        cbdr = get_cbdr_state(m15_df, current_time)

        # Block C_inside (price never broke CBDR) in distribution phase
        if cbdr.is_valid and cbdr.intraday_profile == "C_inside" and amd.phase == "distribution":
            strategy_log.debug(f"{symbol}: CBDR profile=C_inside in distribution — skipping")
            return None

        # 6. Run analysis on closed bars
        direction = "bullish" if h1_bias == 1 else "bearish"
        signal = self._generate_signal(
            symbol, m15_df, h1_df, d1_df, direction, current_time, d1_bias, amd, cbdr
        )

        if signal:
            _flags = (
                ("[IPDA-TP]" if signal.ipda_tp_used else "")
                + ("[CBDR-TP]" if signal.cbdr_tp_used else "")
                + ("[BREAKER]" if signal.breaker_ref else "")
            )
            strategy_log.info(
                "SIGNAL [%s] %s | Entry=%.4f SL=%.4f TP=%.4f RR=%.1f "
                "Score=%.2f D1=%+d AMD=%s CBDR=%s %s",
                symbol, direction.upper(),
                signal.entry_price, signal.stop_loss, signal.take_profit,
                signal.rr_ratio, signal.confluence_score,
                d1_bias, amd.phase, cbdr.intraday_profile, _flags,
            )

        return signal

    # ------------------------------------------------------------------
    # H1 bias
    # ------------------------------------------------------------------

    def _get_h1_bias(self, h1_df: pd.DataFrame) -> int:
        """Return 1 (bullish), -1 (bearish), or 0 (neutral) from H1 structure."""
        bias_series = get_trend_bias(h1_df, self.cfg.swing_lookback)
        if bias_series.empty:
            return 0
        # Use the most recent confirmed bias
        return int(bias_series.iloc[-1])

    # ------------------------------------------------------------------
    # Main signal logic
    # ------------------------------------------------------------------

    def _generate_signal(
        self,
        symbol: str,
        m15_df: pd.DataFrame,
        h1_df: pd.DataFrame,
        d1_df: Optional[pd.DataFrame],
        direction: str,
        current_time: pd.Timestamp,
        d1_bias: int,
        amd: AMDState,
        cbdr: CBDRState = None,
    ) -> Optional[TradeSignal]:

        current_price = float(m15_df.iloc[-1]["close"])
        atr_series    = calculate_atr(m15_df)

        # ── 1. BOS / MSS on M15 WITH displacement filter ─────────────────
        bos_df = detect_bos(m15_df, self.cfg.swing_lookback)
        mss_df = detect_mss(m15_df, self.cfg.swing_lookback)

        if direction == "bullish":
            bos_mask = bos_df["bos_bullish"]
            mss_mask = mss_df["mss_bullish"]
        else:
            bos_mask = bos_df["bos_bearish"]
            mss_mask = mss_df["mss_bearish"]

        # Find last BOS/MSS within 5 bars and check displacement
        recent_bos_bars = bos_mask.iloc[-5:]
        recent_mss_bars = mss_mask.iloc[-5:]
        recent_bos = recent_bos_bars.any()
        recent_mss = recent_mss_bars.any()

        if not (recent_bos or recent_mss):
            return None   # No structural confirmation — don't trade

        # Displacement check: at least one of the recent BOS bars must be displaced
        displaced = False
        n = len(m15_df)
        for offset in range(1, 6):
            idx = n - offset
            if idx < 0:
                break
            if bos_mask.iloc[idx] or mss_mask.iloc[idx]:
                if is_displacement(m15_df, idx, atr_series, min_atr_multiple=0.8):
                    displaced = True
                    break

        score_bos = 1.0 if displaced else 0.6   # Weak BOS still scored, not blocked

        # ── 2. Fair Value Gap ─────────────────────────────────────────────
        fvgs = detect_fvgs(
            m15_df,
            min_size_pct=self.cfg.fvg_min_size_pct,
            max_age_bars=self.cfg.fvg_max_age_bars,
        )
        nearest_fvg = get_nearest_fvg(fvgs, direction, current_price, max_distance_pct=0.005)
        score_fvg   = 1.0 if nearest_fvg else 0.0

        # ── 3. Order Block ────────────────────────────────────────────────
        obs = detect_order_blocks(
            m15_df,
            lookback=self.cfg.swing_lookback,
            max_age_bars=self.cfg.fvg_max_age_bars,
        )
        nearest_ob = get_nearest_ob(obs, direction, current_price, max_distance_pct=0.005)
        score_ob   = 1.0 if nearest_ob else 0.0

        # ── 4. Breaker Block (higher priority entry zone) ─────────────────
        breakers      = detect_breakers(m15_df, lookback=self.cfg.swing_lookback)
        nearest_breaker = get_nearest_breaker(
            breakers, direction, current_price, max_distance_pct=0.008
        )
        score_breaker = 1.5 if nearest_breaker else 0.0   # Bonus weight for breakers

        # ── 5. Liquidity sweep ────────────────────────────────────────────
        liq_levels = find_liquidity_levels(
            m15_df,
            lookback=self.cfg.swing_lookback,
            tolerance_pct=self.cfg.equal_hl_tolerance_pct,
        )
        sweeps         = detect_sweeps(m15_df, liq_levels)
        current_bar_idx = len(m15_df) - 1
        recent_sweep   = get_recent_sweep(sweeps, direction, current_bar_idx, max_age_bars=8)
        score_liquidity = 1.0 if recent_sweep else 0.0

        # ── 6. Session score ──────────────────────────────────────────────
        score_session = session_score(current_time)

        # ── 7. Power of 3 AMD alignment ───────────────────────────────────
        score_amd = amd_confluence_score(amd, direction)
        if score_amd == 0.0 and amd.is_valid and amd.phase == "distribution":
            # AMD is valid, in distribution, and says opposite direction — hard block
            strategy_log.debug(
                f"{symbol}: AMD says {amd.distribution_bias}, signal is {direction} — blocked"
            )
            return None

        # ── 8. Premium/Discount: M15 dealing range ────────────────────────
        dr       = calculate_dealing_range(m15_df, self.cfg.swing_lookback)
        score_pd = pd_score(current_price, dr, direction) if dr else 0.5

        # ── 9. D1 premium/discount alignment ─────────────────────────────
        score_d1_pd = d1_pd_score(d1_df, current_price, direction) if d1_df is not None else 0.5

        # ── 10. IPDA ranges (macro bias check) ────────────────────────────
        ipda_ranges: dict = {}
        score_ipda  = 0.5   # neutral default
        if d1_df is not None and len(d1_df) >= 5:
            ipda_ranges = calculate_ipda_ranges(d1_df, current_price)
            macro_bias  = ipda_bias(ipda_ranges, current_price)
            if macro_bias != 0:
                score_ipda = 1.0 if macro_bias == (1 if direction == "bullish" else -1) else 0.0

        # ── 11. CBDR — dealing range + intraday profile ────────────────────
        score_cbdr = 0.5   # neutral default (no CBDR data)
        if cbdr is not None:
            score_cbdr = cbdr_confluence_score(cbdr, direction, current_price)
            # D_directional profile is high confidence continuation — boost
            if cbdr.intraday_profile == "D_directional" and cbdr.expansion_bias == direction:
                score_cbdr = min(1.0, score_cbdr + 0.15)

        # ── 12. Weighted confluence score ─────────────────────────────────
        w = self.cfg.score_weights
        confluence = (
            w["bos"]              * score_bos
            + w["fvg"]            * score_fvg
            + w.get("order_block", 0.0) * score_ob
            + w["liquidity_sweep"] * score_liquidity
            + w["session"]        * score_session
            + w["pd_zone"]        * score_pd
            # Advanced ICT layers (additive bonuses)
            + 0.10 * score_amd
            + 0.05 * score_d1_pd
            + 0.05 * score_ipda
            + 0.10 * score_cbdr
            + 0.10 * (1.0 if score_breaker > 0 else 0.0)
        )

        # Macro headwind: raise the confluence bar when macro is against the trade.
        # For EURUSD we can pass h1_df as the USD proxy; for others, USD check is skipped.
        macro_delta = macro_confluence_delta(
            symbol, direction, h1_df,
            eurusd_h1=(h1_df if symbol == "EURUSD" else None),
        )
        effective_threshold = self.cfg.min_confluence_score + macro_delta

        if confluence < effective_threshold:
            strategy_log.debug(
                f"{symbol} {direction}: Score {confluence:.2f} below threshold "
                f"{effective_threshold:.2f} (base={self.cfg.min_confluence_score:.2f} "
                f"macro_delta=+{macro_delta:.2f})"
            )
            return None

        # ── 12. Build levels ──────────────────────────────────────────────
        # Entry zone priority: Breaker > FVG > OB > market
        entry_zone = nearest_breaker or nearest_fvg or nearest_ob
        entry, sl, tp = self._calculate_levels(
            direction, current_price, nearest_fvg, nearest_ob, recent_sweep,
            m15_df, dr, nearest_breaker,
        )
        if entry is None:
            return None

        # ── 13. IPDA TP override ──────────────────────────────────────────
        ipda_tp_used = False
        stop_dist = abs(entry - sl)
        if ipda_ranges and stop_dist > 0:
            ipda_tp = get_ipda_tp(
                direction, entry, ipda_ranges,
                min_rr=CONFIG.risk.min_rr_ratio,
                stop_distance=stop_dist,
            )
            if ipda_tp is not None:
                ipda_rr  = abs(ipda_tp - entry) / stop_dist
                swing_rr = abs(tp - entry) / stop_dist
                if ipda_rr > swing_rr:
                    tp           = ipda_tp
                    ipda_tp_used = True

        # ── 14. CBDR projected H/L TP override ────────────────────────────
        cbdr_tp_used = False
        if cbdr is not None and cbdr.is_valid and stop_dist > 0:
            cbdr_tp = get_cbdr_tp(
                direction, cbdr, entry,
                min_rr=CONFIG.risk.min_rr_ratio,
                stop_distance=stop_dist,
            )
            if cbdr_tp is not None:
                cbdr_rr    = abs(cbdr_tp - entry) / stop_dist
                current_rr = abs(tp - entry) / stop_dist
                if cbdr_rr > current_rr:
                    tp           = cbdr_tp
                    cbdr_tp_used = True
                    ipda_tp_used = False   # CBDR overrides IPDA

        # Cap TP at max_rr_ratio
        risk = abs(entry - sl)
        if risk > 0:
            if direction == "bullish":
                tp = min(tp, entry + CONFIG.strategy.max_rr_ratio * risk)
            else:
                tp = max(tp, entry - CONFIG.strategy.max_rr_ratio * risk)

        reward = abs(tp - entry)
        if risk <= 0:
            return None
        rr = reward / risk

        if rr < CONFIG.risk.min_rr_ratio:
            strategy_log.debug(
                f"{symbol}: RR {rr:.2f} below minimum {CONFIG.risk.min_rr_ratio}"
            )
            return None

        return TradeSignal(
            symbol=symbol,
            direction=direction,
            entry_price=round(entry, 5),
            stop_loss=round(sl, 5),
            take_profit=round(tp, 5),
            rr_ratio=round(rr, 2),
            score_bos=score_bos,
            score_fvg=score_fvg,
            score_order_block=score_ob,
            score_liquidity=score_liquidity,
            score_session=score_session,
            score_pd_zone=score_pd,
            confluence_score=round(confluence, 3),
            h1_bias=self._h1_bias.get(symbol, 0),
            d1_bias=d1_bias,
            fvg_ref=nearest_fvg,
            ob_ref=nearest_ob,
            breaker_ref=nearest_breaker,
            sweep_ref=recent_sweep,
            bar_time=current_time,
            entry_type=self.cfg.entry_type,
            amd_phase=amd.phase,
            ipda_tp_used=ipda_tp_used,
            cbdr_profile=cbdr.intraday_profile if cbdr is not None else "unknown",
            cbdr_tp_used=cbdr_tp_used,
        )

    # ------------------------------------------------------------------
    # Level calculation
    # ------------------------------------------------------------------

    def _calculate_levels(
        self,
        direction: str,
        current_price: float,
        fvg: Optional[FVG],
        ob: Optional[OrderBlock],
        sweep: Optional[LiquiditySweep],
        df: pd.DataFrame,
        dr: Optional[DealingRange],
        breaker: Optional[object] = None,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Calculate entry, stop-loss, and take-profit prices.

        Priority: Breaker > FVG > Order Block > market entry.
        Entry: Limit at zone midpoint, or current price.
        SL:    Below zone bottom for longs, above zone top for shorts.
        TP:    Nearest swing high (longs) or swing low (shorts).
              (IPDA override applied in _generate_signal after this call.)
        """
        from strategy.structure import find_swing_highs, find_swing_lows

        if direction == "bullish":
            # Entry zone priority: Breaker > FVG > OB > market
            if breaker:
                entry = breaker.midpoint
                sl    = breaker.bottom - (breaker.bottom * 0.0002)
            elif fvg:
                entry = (fvg.top + fvg.bottom) / 2
                sl    = fvg.bottom - (fvg.bottom * 0.0002)
            elif ob:
                entry = (ob.top + ob.bottom) / 2
                sl    = ob.bottom - (ob.bottom * 0.0002)
            else:
                entry = current_price
                sl    = float(df["low"].iloc[-10:].min()) * 0.9998

            sh_mask   = find_swing_highs(df, lookback=5)
            sh_prices = df.loc[sh_mask, "high"]
            sh_above  = sh_prices[sh_prices > entry]
            if sh_above.empty:
                return None, None, None
            tp   = float(sh_above.iloc[0])
            risk = entry - sl
            if risk > 0:
                max_tp = entry + CONFIG.strategy.max_rr_ratio * risk
                tp     = min(tp, max_tp)

        else:  # bearish
            if breaker:
                entry = breaker.midpoint
                sl    = breaker.top + (breaker.top * 0.0002)
            elif fvg:
                entry = (fvg.top + fvg.bottom) / 2
                sl    = fvg.top + (fvg.top * 0.0002)
            elif ob:
                entry = (ob.top + ob.bottom) / 2
                sl    = ob.top + (ob.top * 0.0002)
            else:
                entry = current_price
                sl    = float(df["high"].iloc[-10:].max()) * 1.0002

            sl_mask   = find_swing_lows(df, lookback=5)
            sl_prices = df.loc[sl_mask, "low"]
            sl_below  = sl_prices[sl_prices < entry]
            if sl_below.empty:
                return None, None, None
            tp   = float(sl_below.iloc[-1])
            risk = sl - entry
            if risk > 0:
                min_tp = entry - CONFIG.strategy.max_rr_ratio * risk
                tp     = max(tp, min_tp)

        return entry, sl, tp
