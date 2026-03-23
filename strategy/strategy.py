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
from strategy.fvg import FVG, detect_fvgs, get_nearest_fvg
from strategy.liquidity import (
    LiquidityLevel, LiquiditySweep, detect_sweeps, find_liquidity_levels,
    get_recent_sweep,
)
from strategy.order_blocks import OrderBlock, detect_order_blocks, get_nearest_ob
from strategy.sessions import is_tradeable_session, session_score
from strategy.structure import detect_bos, detect_mss, get_trend_bias
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
    fvg_ref: Optional[FVG] = None
    ob_ref: Optional[OrderBlock] = None
    sweep_ref: Optional[LiquiditySweep] = None
    bar_time: Optional[pd.Timestamp] = None
    entry_type: str = "limit"   # "limit" or "market"
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
    ) -> Optional[TradeSignal]:
        """
        Called when a new M15 bar closes.
        Returns a TradeSignal if conditions are met, else None.

        Always pass CLOSED bars only (exclude current forming bar).
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

        # 2. Session filter
        current_time = m15_df.index[-1]
        if not is_tradeable_session(current_time):
            strategy_log.debug(f"{symbol}: Outside tradeable session at {current_time}")
            return None

        # 3. Run analysis on closed bars
        direction = "bullish" if h1_bias == 1 else "bearish"
        signal = self._generate_signal(symbol, m15_df, h1_df, direction, current_time)

        if signal:
            strategy_log.info(
                f"SIGNAL [{symbol}] {direction.upper()} | "
                f"Entry={signal.entry_price:.4f} SL={signal.stop_loss:.4f} "
                f"TP={signal.take_profit:.4f} RR={signal.rr_ratio:.1f} "
                f"Score={signal.confluence_score:.2f}"
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
        direction: str,
        current_time: pd.Timestamp,
    ) -> Optional[TradeSignal]:

        current_price = float(m15_df.iloc[-1]["close"])

        # --- BOS / MSS on M15 ---
        bos_df = detect_bos(m15_df, self.cfg.swing_lookback)
        mss_df = detect_mss(m15_df, self.cfg.swing_lookback)

        if direction == "bullish":
            recent_bos = bos_df["bos_bullish"].iloc[-5:].any()
            recent_mss = mss_df["mss_bullish"].iloc[-5:].any()
        else:
            recent_bos = bos_df["bos_bearish"].iloc[-5:].any()
            recent_mss = mss_df["mss_bearish"].iloc[-5:].any()

        score_bos = 1.0 if (recent_bos or recent_mss) else 0.0

        if score_bos == 0.0:
            return None   # No structural confirmation — don't trade

        # --- Fair Value Gap ---
        fvgs = detect_fvgs(
            m15_df,
            min_size_pct=self.cfg.fvg_min_size_pct,
            max_age_bars=self.cfg.fvg_max_age_bars,
        )
        nearest_fvg = get_nearest_fvg(fvgs, direction, current_price, max_distance_pct=0.005)
        score_fvg = 1.0 if nearest_fvg else 0.0

        # --- Order Block ---
        obs = detect_order_blocks(
            m15_df,
            lookback=self.cfg.swing_lookback,
            max_age_bars=self.cfg.fvg_max_age_bars,
        )
        nearest_ob = get_nearest_ob(obs, direction, current_price, max_distance_pct=0.005)
        score_ob = 1.0 if nearest_ob else 0.0

        # --- Liquidity sweep ---
        liq_levels = find_liquidity_levels(
            m15_df,
            lookback=self.cfg.swing_lookback,
            tolerance_pct=self.cfg.equal_hl_tolerance_pct,
        )
        sweeps = detect_sweeps(m15_df, liq_levels)
        current_bar_idx = len(m15_df) - 1
        recent_sweep = get_recent_sweep(sweeps, direction, current_bar_idx, max_age_bars=8)
        score_liquidity = 1.0 if recent_sweep else 0.0

        # --- Session ---
        score_session = session_score(current_time)

        # --- Premium/Discount zone (M15 dealing range) ---
        dr = calculate_dealing_range(m15_df, self.cfg.swing_lookback)
        if dr:
            score_pd = pd_score(current_price, dr, direction)
        else:
            score_pd = 0.5  # neutral if can't determine

        # --- Weighted confluence score ---
        w = self.cfg.score_weights
        confluence = (
            w["bos"] * score_bos
            + w["fvg"] * score_fvg
            + w.get("order_block", 0.0) * score_ob
            + w["liquidity_sweep"] * score_liquidity
            + w["session"] * score_session
            + w["pd_zone"] * score_pd
        )

        if confluence < self.cfg.min_confluence_score:
            strategy_log.debug(
                f"{symbol} {direction}: Score {confluence:.2f} below threshold "
                f"{self.cfg.min_confluence_score}"
            )
            return None

        # --- Build signal ---
        entry, sl, tp = self._calculate_levels(
            direction, current_price, nearest_fvg, nearest_ob, recent_sweep, m15_df, dr
        )
        if entry is None:
            return None

        risk = abs(entry - sl)
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
            fvg_ref=nearest_fvg,
            ob_ref=nearest_ob,
            sweep_ref=recent_sweep,
            bar_time=current_time,
            entry_type=self.cfg.entry_type,
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
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Calculate entry, stop-loss, and take-profit prices.

        Priority: FVG > Order Block > market entry.
        Entry: Limit at FVG midpoint, or OB midpoint, or current price.
        SL:    Below FVG/OB bottom for longs, above FVG/OB top for shorts.
        TP:    Nearest swing high (longs) or swing low (shorts).
        """
        from strategy.structure import find_swing_highs, find_swing_lows

        if direction == "bullish":
            if fvg:
                entry = (fvg.top + fvg.bottom) / 2
                sl = fvg.bottom - (fvg.bottom * 0.0002)
            elif ob:
                entry = (ob.top + ob.bottom) / 2
                sl = ob.bottom - (ob.bottom * 0.0002)
            else:
                entry = current_price
                sl = float(df["low"].iloc[-10:].min()) * 0.9998

            sh_mask = find_swing_highs(df, lookback=5)
            sh_prices = df.loc[sh_mask, "high"]
            sh_above = sh_prices[sh_prices > entry]
            if sh_above.empty:
                return None, None, None
            tp = float(sh_above.iloc[0])

        else:  # bearish
            if fvg:
                entry = (fvg.top + fvg.bottom) / 2
                sl = fvg.top + (fvg.top * 0.0002)
            elif ob:
                entry = (ob.top + ob.bottom) / 2
                sl = ob.top + (ob.top * 0.0002)
            else:
                entry = current_price
                sl = float(df["high"].iloc[-10:].max()) * 1.0002

            sl_mask = find_swing_lows(df, lookback=5)
            sl_prices = df.loc[sl_mask, "low"]
            sl_below = sl_prices[sl_prices < entry]
            if sl_below.empty:
                return None, None, None
            tp = float(sl_below.iloc[-1])

        return entry, sl, tp
