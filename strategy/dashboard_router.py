"""
dashboard_router.py — DashboardRouter

Replicates the signal logic of the Gold Weekly Bias Dashboard
(https://tradinginvesting-6ltxxnck9m2gfj2i473ntd.streamlit.app/) and
feeds the resulting ICT trade ideas into the IBKR paper trading system.

Architecture
------------
This router is XAUUSD-only and runs on Daily/Weekly/Monthly data — a completely
different cadence from the M15/H1 ICT or research routers.

Signal lifecycle:
  1. Once per day at 08:00 UTC (London open) → fetch fresh data, run analysis.
  2. Up to 3 ICT trade ideas generated (Primary Trend, OTE, Liquidity Hunt).
  3. Best HIGH/MEDIUM confidence trade → converted to TradeSignal.
  4. Signal returned on the next route() call; None for the rest of the day.
  5. Already-sent signal is cleared after one emission (gate = once per day).

Bias score (simplified — no FRED/COT needed):
  Technical  50%: price above 20W/50W/200W MA, weekly RSI(14), weekly MACD
  Cross-asset 50%: DXY weekly direction, SPX weekly direction,
                   EURUSD weekly direction, VIX level

ICT analysis: ported directly from the dashboard's ict_analysis.py —
  Market structure (bullish/bearish/ranging), Fibonacci OTE zones,
  Fair Value Gaps, Order Blocks, key levels (PWH/PWL/PMH/PML).

Deploy as: trading-bot-dashboard.service  (clientId=3, port 4002)
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from strategy.strategy import TradeSignal

_log = logging.getLogger("strategy.dashboard")

# ---------------------------------------------------------------------------
# Constants (ported from dashboard config.py)
# ---------------------------------------------------------------------------

_ICT_SWING_ORDER          = 4
_ICT_FVG_LOOKBACK         = 20
_ICT_OB_LOOKBACK          = 20
_ICT_OB_MIN_IMPULSE       = 0.3
_ICT_FIB_LEVELS           = [0.236, 0.382, 0.5, 0.618, 0.705, 0.786]
_ICT_OTE_LOW              = 0.618
_ICT_OTE_HIGH             = 0.705
_ICT_DAILY_SWING_LOOKBACK = 30
_ICT_DAILY_SWING_FALLBACK = 60
_ICT_DAILY_SWING_MIN_RANGE_PCT = 0.01
_ICT_BIAS_BULL_THRESHOLD  = 0.05
_ICT_BIAS_BEAR_THRESHOLD  = -0.05
_ICT_CONFIDENCE_HIGH_THRESHOLD = 0.30
_ICT_OB_NEAR_LONG_UPPER   = 1.03
_ICT_OB_NEAR_LONG_LOWER   = 0.92
_ICT_OB_NEAR_SHORT_LOWER  = 0.97
_ICT_OB_NEAR_SHORT_UPPER  = 1.08
_ICT_OB_STOP_BUFFER_FRACTION  = 0.50
_ICT_OB_STOP_BUFFER_FALLBACK  = 0.003
_ICT_ATR_PERIOD           = 14
_ICT_OTE_ATR_MULTIPLIER   = 1.5
_ICT_LIQ_ATR_MULTIPLIER   = 1.0
_ICT_OTE_STOP_FALLBACK_PCT= 0.006
_ICT_LIQ_STOP_FALLBACK_PCT= 0.004

# Min RR to emit a signal (dashboard trade ideas target 2:1+)
_MIN_RR = 1.5

# London open — time to refresh daily signals
_LONDON_OPEN_HOUR = 8   # 08:00 UTC

# ---------------------------------------------------------------------------
# Per-instrument yfinance ticker map
# ---------------------------------------------------------------------------
_INSTRUMENT_TICKERS: Dict[str, str] = {
    "XAUUSD": "GC=F",       # Gold futures (continuous)
    "XAGUSD": "SI=F",       # Silver futures (continuous)
    "EURUSD": "EURUSD=X",   # EUR/USD spot
    "NAS100": "QQQ",        # Nasdaq-100 ETF (best weekly data)
    "GBPUSD": "GBPUSD=X",   # GBP/USD spot
    "GBPJPY": "GBPJPY=X",   # GBP/JPY cross
    "BTC":    "BTC-USD",    # Bitcoin spot
    "OIL":    "CL=F",       # WTI Crude futures (continuous)
}

# Cross-asset tickers shared by multiple instruments
_DXY_TICKER   = "DX-Y.NYB"
_SPX_TICKER   = "^GSPC"
_VIX_TICKER   = "^VIX"
_EURUSD_TICKER = "EURUSD=X"
_USDJPY_TICKER = "JPY=X"
_GBPUSD_TICKER = "GBPUSD=X"
_GOLD_TICKER   = "GC=F"

# Cross-asset tickers to fetch per instrument
_INSTRUMENT_CROSS_TICKERS: Dict[str, Dict[str, str]] = {
    "XAUUSD": {"dxy": _DXY_TICKER, "spx": _SPX_TICKER, "eurusd": _EURUSD_TICKER, "vix": _VIX_TICKER},
    "XAGUSD": {"gold": _GOLD_TICKER, "dxy": _DXY_TICKER, "spx": _SPX_TICKER, "vix": _VIX_TICKER},
    "EURUSD": {"dxy": _DXY_TICKER, "spx": _SPX_TICKER, "vix": _VIX_TICKER, "usdjpy": _USDJPY_TICKER},
    "NAS100": {"spx": _SPX_TICKER, "vix": _VIX_TICKER, "dxy": _DXY_TICKER},
    "GBPUSD": {"dxy": _DXY_TICKER, "eurusd": _EURUSD_TICKER, "vix": _VIX_TICKER},
    "GBPJPY": {"spx": _SPX_TICKER, "vix": _VIX_TICKER, "gbpusd": _GBPUSD_TICKER, "usdjpy": _USDJPY_TICKER},
    "BTC":    {"spx": _SPX_TICKER, "dxy": _DXY_TICKER, "vix": _VIX_TICKER},
    "OIL":    {"dxy": _DXY_TICKER, "spx": _SPX_TICKER, "vix": _VIX_TICKER, "gold": _GOLD_TICKER},
}


# ---------------------------------------------------------------------------
# ICT Analysis Engine (ported from dashboard/ict_analysis.py)
# ---------------------------------------------------------------------------

def _find_swing_points(df: pd.DataFrame, order: int = 3) -> dict:
    if df.empty or len(df) < 2 * order + 1:
        empty = pd.Series(dtype=float)
        return {"highs": empty, "lows": empty}

    highs: dict = {}
    lows:  dict = {}
    high_arr = df["High"].values
    low_arr  = df["Low"].values
    idx      = df.index

    for i in range(order, len(df) - order):
        window_h = high_arr[i - order : i + order + 1]
        window_l = low_arr[i - order : i + order + 1]
        if high_arr[i] == np.max(window_h):
            highs[idx[i]] = float(high_arr[i])
        if low_arr[i] == np.min(window_l):
            lows[idx[i]] = float(low_arr[i])

    return {
        "highs": pd.Series(highs, dtype=float),
        "lows":  pd.Series(lows,  dtype=float),
    }


def _detect_market_structure(df: pd.DataFrame, lookback: int = 10) -> str:
    swings = _find_swing_points(df, order=_ICT_SWING_ORDER)
    sh = swings["highs"].dropna()
    sl = swings["lows"].dropna()

    if len(sh) < 2 or len(sl) < 2:
        return "ranging"

    sh_tail = sh.tail(lookback)
    sl_tail = sl.tail(lookback)

    hh = float(sh_tail.iloc[-1]) > float(sh_tail.iloc[-2])
    hl = float(sl_tail.iloc[-1]) > float(sl_tail.iloc[-2])
    lh = float(sh_tail.iloc[-1]) < float(sh_tail.iloc[-2])
    ll = float(sl_tail.iloc[-1]) < float(sl_tail.iloc[-2])

    if hh and hl:
        return "bullish"
    if lh and ll:
        return "bearish"
    return "ranging"


def _calc_fibonacci_levels(swing_high: float, swing_low: float) -> dict:
    if swing_high <= swing_low:
        return {}
    rng = swing_high - swing_low
    result: dict = {
        level: round(swing_high - level * rng, 2)
        for level in _ICT_FIB_LEVELS
    }
    result["swing_high"] = round(swing_high, 2)
    result["swing_low"]  = round(swing_low,  2)
    result["range"]      = round(rng, 2)
    return result


def _fvg_filled(df: pd.DataFrame, gap_bar_idx: int,
                top: float, bottom: float, direction: str) -> bool:
    subsequent = df.iloc[gap_bar_idx + 1:]
    if subsequent.empty:
        return False
    midpoint = (top + bottom) / 2.0
    if direction == "bullish":
        return bool((subsequent["Low"] <= midpoint).any())
    return bool((subsequent["High"] >= midpoint).any())


def _find_fvgs(df: pd.DataFrame, n_recent: int = 20) -> list[dict]:
    if df.empty or len(df) < 3:
        return []

    fvgs: list[dict] = []
    scan_start = max(0, len(df) - n_recent - 2)

    for i in range(scan_start + 2, len(df)):
        c0_high = float(df["High"].iloc[i - 2])
        c2_high = float(df["High"].iloc[i])
        c2_low  = float(df["Low"].iloc[i])
        c0_low  = float(df["Low"].iloc[i - 2])
        dt = df.index[i]

        if c0_high < c2_low:
            top, bottom = c2_low, c0_high
            if top > bottom:
                fvgs.append({
                    "direction": "bullish", "top": round(top, 2),
                    "bottom": round(bottom, 2), "midpoint": round((top + bottom) / 2, 2),
                    "date": dt, "filled": _fvg_filled(df, i, top, bottom, "bullish"),
                })
        elif c0_low > c2_high:
            top, bottom = c0_low, c2_high
            if top > bottom:
                fvgs.append({
                    "direction": "bearish", "top": round(top, 2),
                    "bottom": round(bottom, 2), "midpoint": round((top + bottom) / 2, 2),
                    "date": dt, "filled": _fvg_filled(df, i, top, bottom, "bearish"),
                })

    return list(reversed(fvgs))


def _find_order_blocks(df: pd.DataFrame, n_recent: int = 20,
                       min_impulse_pct: float = 0.3) -> list[dict]:
    if df.empty or len(df) < 3:
        return []

    obs: list[dict] = []
    scan_start = max(0, len(df) - n_recent - 1)

    for i in range(scan_start + 1, len(df) - 1):
        prev_o = float(df["Open"].iloc[i - 1])
        prev_c = float(df["Close"].iloc[i - 1])
        prev_h = float(df["High"].iloc[i - 1])
        prev_l = float(df["Low"].iloc[i - 1])
        nxt_o  = float(df["Open"].iloc[i + 1])
        nxt_c  = float(df["Close"].iloc[i + 1])
        dt     = df.index[i - 1]

        if prev_c < prev_o and nxt_o != 0:
            impulse_pct = (nxt_c - nxt_o) / abs(nxt_o) * 100
            if impulse_pct >= min_impulse_pct:
                subsequent = df.iloc[i + 1:]
                valid = not bool((subsequent["Low"] < prev_l).any())
                obs.append({
                    "direction": "bullish", "high": round(prev_h, 2),
                    "low": round(prev_l, 2), "date": dt, "valid": valid,
                    "impulse_pct": round(impulse_pct, 2),
                })
        elif prev_c > prev_o and nxt_o != 0:
            impulse_pct = (nxt_o - nxt_c) / abs(nxt_o) * 100
            if impulse_pct >= min_impulse_pct:
                subsequent = df.iloc[i + 1:]
                valid = not bool((subsequent["High"] > prev_h).any())
                obs.append({
                    "direction": "bearish", "high": round(prev_h, 2),
                    "low": round(prev_l, 2), "date": dt, "valid": valid,
                    "impulse_pct": round(impulse_pct, 2),
                })

    return list(reversed(obs))


def _get_key_levels(monthly_df: pd.DataFrame, weekly_df: pd.DataFrame) -> dict:
    def _safe(df, row_idx, col):
        try:
            if len(df) > abs(row_idx):
                return float(df[col].iloc[row_idx])
        except Exception:
            pass
        return None

    return {
        "PMH": _safe(monthly_df, -2, "High"),
        "PML": _safe(monthly_df, -2, "Low"),
        "CMH": _safe(monthly_df, -1, "High"),
        "CML": _safe(monthly_df, -1, "Low"),
        "PWH": _safe(weekly_df,  -2, "High"),
        "PWL": _safe(weekly_df,  -2, "Low"),
        "CWH": _safe(weekly_df,  -1, "High"),
        "CWL": _safe(weekly_df,  -1, "Low"),
    }


def _find_major_swing(df: pd.DataFrame,
                      lookback_bars: int = 60) -> Tuple[float, float, str]:
    window = df.tail(lookback_bars)
    if window.empty:
        return 0.0, 0.0, "ranging"
    high_idx = window["High"].idxmax()
    low_idx  = window["Low"].idxmin()
    sh = float(window["High"].max())
    sl = float(window["Low"].min())
    return sh, sl, ("up" if high_idx > low_idx else "down")


def _calc_rr(entry, stop, target) -> Optional[float]:
    if entry is None or stop is None or target is None:
        return None
    risk = abs(entry - stop)
    if risk == 0:
        return None
    return round(abs(target - entry) / risk, 2)


def _calc_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    if len(df) < period + 1:
        return None
    high = df["High"]
    low  = df["Low"]
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return float(tr.tail(period).mean())


def _wait_trade(trade_id: int, reason: str = "Insufficient data") -> dict:
    return {
        "id": trade_id, "direction": "WAIT", "setup_name": "No Setup",
        "entry": None, "stop": None, "target1": None, "target2": None,
        "rr1": None, "rr2": None, "confidence": "LOW",
        "rationale": reason, "key_levels_used": [], "timeframe": "—",
    }


def _generate_ict_trades(monthly_df: pd.DataFrame, weekly_df: pd.DataFrame,
                         daily_df: pd.DataFrame, bias_score: float) -> list[dict]:
    """Port of dashboard's generate_ict_trades() — generates 3 trade setups."""
    MIN_BARS = 10
    if (monthly_df.empty or weekly_df.empty or daily_df.empty
            or len(monthly_df) < MIN_BARS
            or len(weekly_df) < MIN_BARS
            or len(daily_df) < 3):
        reason = "Insufficient historical data."
        return [_wait_trade(i, reason) for i in range(1, 4)]

    try:
        monthly_structure = _detect_market_structure(monthly_df)
        weekly_structure  = _detect_market_structure(weekly_df)
        key_levels        = _get_key_levels(monthly_df, weekly_df)
        current_price     = float(daily_df["Close"].iloc[-1])
        atr               = _calc_atr(daily_df, period=_ICT_ATR_PERIOD)

        sh, sl, swing_dir = _find_major_swing(daily_df, lookback_bars=_ICT_DAILY_SWING_LOOKBACK)
        if sh - sl < current_price * _ICT_DAILY_SWING_MIN_RANGE_PCT:
            sh, sl, swing_dir = _find_major_swing(daily_df, lookback_bars=_ICT_DAILY_SWING_FALLBACK)

        fib       = _calc_fibonacci_levels(sh, sl)
        fifty_pct = fib.get(0.5)

        weekly_obs  = [o for o in _find_order_blocks(weekly_df, n_recent=_ICT_OB_LOOKBACK,
                                                     min_impulse_pct=_ICT_OB_MIN_IMPULSE)
                       if o["valid"]]
        daily_obs   = [o for o in _find_order_blocks(daily_df, n_recent=_ICT_OB_LOOKBACK,
                                                     min_impulse_pct=_ICT_OB_MIN_IMPULSE)
                       if o["valid"]]
        weekly_fvgs = [f for f in _find_fvgs(weekly_df, n_recent=_ICT_FVG_LOOKBACK)
                       if not f["filled"]]
        daily_fvgs  = [f for f in _find_fvgs(daily_df, n_recent=_ICT_FVG_LOOKBACK)
                       if not f["filled"]]
    except Exception as exc:
        reason = f"ICT computation error: {exc}"
        return [_wait_trade(i, reason) for i in range(1, 4)]

    # Overall bias
    if (monthly_structure == "bullish" and weekly_structure == "bullish"
            and bias_score > _ICT_BIAS_BULL_THRESHOLD):
        overall_bias = "bullish"
    elif (monthly_structure == "bearish" and weekly_structure == "bearish"
            and bias_score < _ICT_BIAS_BEAR_THRESHOLD):
        overall_bias = "bearish"
    else:
        overall_bias = "ranging"

    # ── Trade 1: Primary Trend ────────────────────────────────────────────
    def _trade1() -> dict:
        base = {"id": 1, "timeframe": "Monthly + Weekly", "key_levels_used": []}

        if overall_bias == "bullish":
            candidates = sorted(
                [o for o in (daily_obs + weekly_obs)
                 if o["direction"] == "bullish"
                 and o["high"] <= current_price * _ICT_OB_NEAR_LONG_UPPER
                 and o["high"] >= current_price * _ICT_OB_NEAR_LONG_LOWER],
                key=lambda o: abs(o["high"] - current_price),
            )
            ob = candidates[0] if candidates else None

            if ob:
                ob_range = ob["high"] - ob["low"]
                entry = round((ob["high"] + ob["low"]) / 2, 1)
                stop  = round(ob["low"] - max(ob_range * _ICT_OB_STOP_BUFFER_FRACTION,
                                             current_price * _ICT_OB_STOP_BUFFER_FALLBACK), 1)
                conf  = "HIGH" if abs(bias_score) > _ICT_CONFIDENCE_HIGH_THRESHOLD else "MEDIUM"
                levels = [f"Bullish OB ${ob['low']:.1f}–${ob['high']:.1f}"]
            else:
                pwl = key_levels.get("PWL") or key_levels.get("CWL")
                entry = round(current_price, 1)
                stop  = round(pwl * 0.998, 1) if pwl else round(current_price * 0.985, 1)
                conf  = "MEDIUM"
                levels = [f"No OB — stop below PWL ${stop:.1f}"]

            tp1_cands = [v for v in [key_levels.get("PWH"), key_levels.get("CMH")]
                         if v and v > current_price]
            tp1 = round(min(tp1_cands), 1) if tp1_cands else None
            tp2_cands = [v for v in [key_levels.get("PMH"), sh] if v and v > current_price]
            tp2 = round(max(tp2_cands), 1) if tp2_cands else None
            fvg_t = next((f for f in (weekly_fvgs + daily_fvgs)
                          if f["direction"] == "bullish" and f["bottom"] > entry), None)

            return {**base, "direction": "LONG", "setup_name": "Weekly OB — Primary Trend Long",
                    "entry": entry, "stop": stop, "target1": tp1, "target2": tp2,
                    "rr1": _calc_rr(entry, stop, tp1), "rr2": _calc_rr(entry, stop, tp2),
                    "confidence": conf, "key_levels_used": levels,
                    "rationale": (f"Monthly:{monthly_structure.upper()} Weekly:{weekly_structure.upper()} "
                                  f"bias={bias_score:+.2f}. "
                                  + (f"Bullish OB ${ob['low']:.1f}–${ob['high']:.1f}." if ob else ""))}

        elif overall_bias == "bearish":
            candidates = sorted(
                [o for o in (daily_obs + weekly_obs)
                 if o["direction"] == "bearish"
                 and o["low"] >= current_price * _ICT_OB_NEAR_SHORT_LOWER
                 and o["low"] <= current_price * _ICT_OB_NEAR_SHORT_UPPER],
                key=lambda o: abs(o["low"] - current_price),
            )
            ob = candidates[0] if candidates else None

            if ob:
                ob_range = ob["high"] - ob["low"]
                entry = round((ob["high"] + ob["low"]) / 2, 1)
                stop  = round(ob["high"] + max(ob_range * _ICT_OB_STOP_BUFFER_FRACTION,
                                              current_price * _ICT_OB_STOP_BUFFER_FALLBACK), 1)
                conf  = "HIGH" if abs(bias_score) > _ICT_CONFIDENCE_HIGH_THRESHOLD else "MEDIUM"
                levels = [f"Bearish OB ${ob['low']:.1f}–${ob['high']:.1f}"]
            else:
                pwh = key_levels.get("PWH") or key_levels.get("CWH")
                entry = round(current_price, 1)
                stop  = round(pwh * 1.002, 1) if pwh else round(current_price * 1.015, 1)
                conf  = "MEDIUM"
                levels = [f"No OB — stop above PWH ${stop:.1f}"]

            tp1_cands = [v for v in [key_levels.get("PWL"), key_levels.get("CML")]
                         if v and v < current_price]
            tp1 = round(max(tp1_cands), 1) if tp1_cands else None
            tp2_cands = [v for v in [key_levels.get("PML"), sl] if v and v < current_price]
            tp2 = round(min(tp2_cands), 1) if tp2_cands else None
            fvg_t = next((f for f in (weekly_fvgs + daily_fvgs)
                          if f["direction"] == "bearish" and f["top"] < entry), None)

            return {**base, "direction": "SHORT", "setup_name": "Weekly OB — Primary Trend Short",
                    "entry": entry, "stop": stop, "target1": tp1, "target2": tp2,
                    "rr1": _calc_rr(entry, stop, tp1), "rr2": _calc_rr(entry, stop, tp2),
                    "confidence": conf, "key_levels_used": levels,
                    "rationale": (f"Monthly:{monthly_structure.upper()} Weekly:{weekly_structure.upper()} "
                                  f"bias={bias_score:+.2f}. "
                                  + (f"Bearish OB ${ob['low']:.1f}–${ob['high']:.1f}." if ob else ""))}

        return _wait_trade(1, f"Ranging market — Monthly:{monthly_structure}, Weekly:{weekly_structure}.")

    # ── Trade 2: OTE Retracement ──────────────────────────────────────────
    def _trade2() -> dict:
        base = {"id": 2, "timeframe": "Weekly Fibonacci"}
        if not fib or "swing_high" not in fib or "swing_low" not in fib:
            return _wait_trade(2, "Fibonacci not computable.")

        _sh  = fib["swing_high"]
        _rng = fib["range"]
        ote_lo = round(_sh - _ICT_OTE_LOW  * _rng, 2)
        ote_hi = round(_sh - _ICT_OTE_HIGH * _rng, 2)

        if overall_bias == "bullish":
            entry = round((ote_lo + ote_hi) / 2, 1)
            _fib786 = fib.get(0.786, ote_hi * (1 - _ICT_OTE_STOP_FALLBACK_PCT))
            _buf = (atr * _ICT_OTE_ATR_MULTIPLIER) if atr else (current_price * _ICT_OTE_STOP_FALLBACK_PCT)
            stop  = round(_fib786 - _buf, 1)
            tp1   = round(sh, 1)
            fvg_a = next((f for f in (weekly_fvgs + daily_fvgs)
                          if f["direction"] == "bullish" and f["bottom"] > entry), None)
            tp2 = round(fvg_a["top"], 1) if fvg_a else round(sh * 1.005, 1)

            if ote_hi <= current_price <= ote_lo:
                conf = "HIGH"
            elif abs(current_price - ote_lo) / ote_lo < 0.02:
                conf = "MEDIUM"
            elif current_price > ote_lo:
                return _wait_trade(2, f"Price ${current_price:,.1f} above OTE (${ote_hi:,.1f}–${ote_lo:,.1f}).")
            else:
                conf = "LOW"

            return {**base, "direction": "LONG", "setup_name": "OTE Retracement — Fib 0.618–0.705",
                    "entry": entry, "stop": stop, "target1": tp1, "target2": tp2,
                    "rr1": _calc_rr(entry, stop, tp1), "rr2": _calc_rr(entry, stop, tp2),
                    "confidence": conf, "key_levels_used": [f"OTE ${ote_hi:.1f}–${ote_lo:.1f}"],
                    "rationale": f"OTE zone ${ote_hi:.1f}–${ote_lo:.1f} (Fib 0.618–0.705). bias={bias_score:+.2f}."}

        elif overall_bias == "bearish":
            entry = round((ote_lo + ote_hi) / 2, 1)
            _fib786 = fib.get(0.786, ote_lo * (1 + _ICT_OTE_STOP_FALLBACK_PCT))
            _buf = (atr * _ICT_OTE_ATR_MULTIPLIER) if atr else (current_price * _ICT_OTE_STOP_FALLBACK_PCT)
            stop  = round(_fib786 + _buf, 1)
            tp1   = round(sl, 1)
            fvg_b = next((f for f in (weekly_fvgs + daily_fvgs)
                          if f["direction"] == "bearish" and f["top"] < entry), None)
            tp2 = round(fvg_b["bottom"], 1) if fvg_b else round(sl * 0.995, 1)

            if ote_hi <= current_price <= ote_lo:
                conf = "HIGH"
            elif abs(current_price - ote_hi) / ote_hi < 0.02:
                conf = "MEDIUM"
            elif current_price < ote_hi:
                return _wait_trade(2, f"Price ${current_price:,.1f} below OTE (${ote_hi:,.1f}–${ote_lo:,.1f}).")
            else:
                conf = "LOW"

            return {**base, "direction": "SHORT", "setup_name": "OTE Retracement — Fib 0.618–0.705",
                    "entry": entry, "stop": stop, "target1": tp1, "target2": tp2,
                    "rr1": _calc_rr(entry, stop, tp1), "rr2": _calc_rr(entry, stop, tp2),
                    "confidence": conf, "key_levels_used": [f"OTE ${ote_hi:.1f}–${ote_lo:.1f}"],
                    "rationale": f"OTE zone ${ote_hi:.1f}–${ote_lo:.1f} (Fib 0.618–0.705). bias={bias_score:+.2f}."}

        return _wait_trade(2, "Ranging market — no OTE trade.")

    # ── Trade 3: Liquidity Hunt ───────────────────────────────────────────
    def _trade3() -> dict:
        base = {"id": 3, "timeframe": "Daily Liquidity"}
        pwh = key_levels.get("PWH")
        pwl = key_levels.get("PWL")
        if not pwh or not pwl:
            return _wait_trade(3, "PWH/PWL unavailable.")

        _buf = (atr * _ICT_LIQ_ATR_MULTIPLIER) if atr else (current_price * _ICT_LIQ_STOP_FALLBACK_PCT)

        # Counter-trend: if bullish bias, look for short after PWH sweep
        if overall_bias == "bullish" and current_price >= pwh * 0.998:
            entry = round(pwh, 1)
            stop  = round(pwh + _buf, 1)
            tp1   = round(current_price - (stop - entry), 1)   # 1R target
            tp2   = round(pwl, 1)
            conf  = "MEDIUM"
            return {**base, "direction": "SHORT", "setup_name": "Liquidity Hunt — PWH Sweep",
                    "entry": entry, "stop": stop, "target1": tp1, "target2": tp2,
                    "rr1": _calc_rr(entry, stop, tp1), "rr2": _calc_rr(entry, stop, tp2),
                    "confidence": conf, "key_levels_used": [f"PWH ${pwh:.1f}"],
                    "rationale": f"Price approaching PWH ${pwh:.1f} — liquidity sweep short."}

        elif overall_bias == "bearish" and current_price <= pwl * 1.002:
            entry = round(pwl, 1)
            stop  = round(pwl - _buf, 1)
            tp1   = round(current_price + (entry - stop), 1)
            tp2   = round(pwh, 1)
            conf  = "MEDIUM"
            return {**base, "direction": "LONG", "setup_name": "Liquidity Hunt — PWL Sweep",
                    "entry": entry, "stop": stop, "target1": tp1, "target2": tp2,
                    "rr1": _calc_rr(entry, stop, tp1), "rr2": _calc_rr(entry, stop, tp2),
                    "confidence": conf, "key_levels_used": [f"PWL ${pwl:.1f}"],
                    "rationale": f"Price approaching PWL ${pwl:.1f} — liquidity sweep long."}

        return _wait_trade(3, "No liquidity sweep setup at PWH/PWL.")

    return [_trade1(), _trade2(), _trade3()]


# ---------------------------------------------------------------------------
# Simplified Bias Score (price-only — no FRED/COT needed)
# ---------------------------------------------------------------------------

def _chg(df: pd.DataFrame) -> float:
    """Weekly % change from second-to-last to last close. Returns 0.0 on error."""
    try:
        c = df["Close"]
        prev = float(c.iloc[-2])
        return (float(c.iloc[-1]) - prev) / prev if prev else 0.0
    except Exception:
        return 0.0


def _score_technical(weekly_df: pd.DataFrame) -> float:
    """Generic technical score [-0.5, +0.5]: MAs 0.40, RSI 0.05, MACD 0.05."""
    if weekly_df.empty or len(weekly_df) < 5:
        return 0.0
    score = 0.0
    close = weekly_df["Close"]
    cp = float(close.iloc[-1])
    for period, weight in [(20, 0.15), (50, 0.15), (200, 0.10)]:
        if len(close) >= period:
            score += weight * (1.0 if cp > float(close.tail(period).mean()) else -1.0)
    if len(close) >= 15:
        delta = close.diff().dropna()
        gain = delta.clip(lower=0).tail(14).mean()
        loss = (-delta.clip(upper=0)).tail(14).mean()
        if loss > 0:
            rsi = 100 - (100 / (1 + gain / loss))
            score += 0.05 if 50 <= rsi <= 70 else (-0.05 if rsi > 75 else 0.0)
    if len(close) >= 30:
        ema12 = float(close.ewm(span=12, adjust=False).mean().iloc[-1])
        ema26 = float(close.ewm(span=26, adjust=False).mean().iloc[-1])
        score += 0.05 * (1.0 if ema12 > ema26 else -1.0)
    return score


def _score_cross_asset(symbol: str, cross_df: Dict[str, pd.DataFrame]) -> float:
    """
    Per-instrument cross-asset score [-0.5, +0.5].
    Relationships are directional — rising/falling cross-asset moves add/subtract.
    """
    sym = symbol.upper()
    score = 0.0

    def _safe_chg(key: str) -> float:
        df = cross_df.get(key)
        return _chg(df) if df is not None and len(df) >= 2 else 0.0

    def _safe_val(key: str) -> float:
        try:
            return float(cross_df[key]["Close"].iloc[-1])
        except Exception:
            return 0.0

    dxy_chg   = _safe_chg("dxy")
    spx_chg   = _safe_chg("spx")
    vix_val   = _safe_val("vix")
    vix_chg   = _safe_chg("vix") * vix_val   # absolute change approximation
    eur_chg   = _safe_chg("eurusd")
    usdjpy_chg = _safe_chg("usdjpy")
    gbpusd_chg = _safe_chg("gbpusd")
    gold_chg  = _safe_chg("gold")

    if sym == "XAUUSD":
        # Falling DXY = bullish gold
        score += 0.15 * (-1.0 if dxy_chg > 0 else 1.0)
        # Falling SPX = flight-to-safety bid
        score += 0.10 * (-1.0 if spx_chg > 0.02 else (1.0 if spx_chg < -0.02 else 0.0))
        # Rising EUR = weak USD = bullish gold
        if abs(eur_chg) > 0.003:
            score += 0.10 * (1.0 if eur_chg > 0 else -1.0)
        # VIX spike = risk-off = safe-haven bid
        if vix_val > 25 or vix_chg > 3:
            score += 0.05
        elif vix_val < 15 and vix_chg < -3:
            score -= 0.05

    elif sym == "XAGUSD":
        # Gold direction is strongest driver (corr ~0.85)
        score += 0.15 * (1.0 if gold_chg > 0 else -1.0)
        # DXY inverse
        score += 0.10 * (-1.0 if dxy_chg > 0 else 1.0)
        # SPX rising = industrial demand bullish
        score += 0.10 * (1.0 if spx_chg > 0.01 else (-1.0 if spx_chg < -0.02 else 0.0))
        # VIX spike = risk-off hurts industrial demand
        if vix_val > 25:
            score -= 0.05
        elif vix_val < 15:
            score += 0.05

    elif sym == "EURUSD":
        # DXY direct inverse (strongest driver)
        score += 0.20 * (-1.0 if dxy_chg > 0 else 1.0)
        # Risk-on = sell USD = bullish EUR/USD
        score += 0.10 * (1.0 if spx_chg > 0.01 else (-1.0 if spx_chg < -0.02 else 0.0))
        # VIX spike = flight to USD = bearish EUR/USD
        score += 0.10 * (-1.0 if vix_val > 25 else (1.0 if vix_val < 15 else 0.0))
        # USDJPY rising = broad USD strength = bearish EUR/USD
        score += 0.10 * (-1.0 if usdjpy_chg > 0.003 else (1.0 if usdjpy_chg < -0.003 else 0.0))

    elif sym == "NAS100":
        # SPX direction is the dominant driver (corr ~0.95)
        score += 0.15 * (1.0 if spx_chg > 0 else -1.0)
        # VIX = risk-off destroys NAS
        score += 0.15 * (-1.0 if vix_val > 20 else (1.0 if vix_val < 15 else 0.0))
        # DXY: weak USD = risk-on / growth bias = mildly bullish tech
        score += 0.10 * (-1.0 if dxy_chg > 0.005 else (1.0 if dxy_chg < -0.005 else 0.0))

    elif sym == "GBPUSD":
        # DXY inverse
        score += 0.15 * (-1.0 if dxy_chg > 0 else 1.0)
        # EUR/USD correlated — GBP follows EUR direction
        if abs(eur_chg) > 0.002:
            score += 0.15 * (1.0 if eur_chg > 0 else -1.0)
        # VIX: risk-off = sell GBP (risk currency vs USD)
        score += 0.10 * (-1.0 if vix_val > 25 else (1.0 if vix_val < 15 else 0.0))

    elif sym == "GBPJPY":
        # SPX: risk-on = sell JPY = bullish GBP/JPY
        score += 0.15 * (1.0 if spx_chg > 0.01 else (-1.0 if spx_chg < -0.02 else 0.0))
        # VIX: risk-off = buy JPY = bearish GBP/JPY
        score += 0.15 * (-1.0 if vix_val > 25 else (1.0 if vix_val < 15 else 0.0))
        # GBP/USD direction adds GBP-side impulse
        if abs(gbpusd_chg) > 0.002:
            score += 0.10 * (1.0 if gbpusd_chg > 0 else -1.0)
        # USD/JPY rising = JPY weak = bullish GBP/JPY
        if abs(usdjpy_chg) > 0.003:
            score += 0.10 * (1.0 if usdjpy_chg > 0 else -1.0)

    elif sym == "BTC":
        # SPX: risk-on = bullish crypto (corr ~0.60)
        score += 0.15 * (1.0 if spx_chg > 0.01 else (-1.0 if spx_chg < -0.02 else 0.0))
        # DXY: weak USD = anti-dollar bid = bullish BTC
        score += 0.15 * (-1.0 if dxy_chg > 0 else 1.0)
        # VIX: risk-off = sell crypto
        score += 0.10 * (-1.0 if vix_val > 25 else (1.0 if vix_val < 15 else 0.0))

    elif sym == "OIL":
        # DXY inverse (oil priced in USD)
        score += 0.15 * (-1.0 if dxy_chg > 0 else 1.0)
        # SPX: economic activity = demand
        score += 0.10 * (1.0 if spx_chg > 0.01 else (-1.0 if spx_chg < -0.02 else 0.0))
        # VIX: demand fears
        score += 0.10 * (-1.0 if vix_val > 25 else (1.0 if vix_val < 15 else 0.0))
        # Gold rising = inflation narrative = bullish OIL
        if abs(gold_chg) > 0.005:
            score += 0.05 * (1.0 if gold_chg > 0 else -1.0)

    return score


def _compute_bias_score(symbol: str, weekly_df: pd.DataFrame, cross_df: Dict[str, pd.DataFrame]) -> float:
    """
    Multi-instrument bias score [-1, +1].
    Technical 50% (generic) + Cross-Asset 50% (per-instrument).
    """
    tech  = _score_technical(weekly_df)
    cross = _score_cross_asset(symbol, cross_df)
    return round(max(-1.0, min(1.0, tech + cross)), 4)


# ---------------------------------------------------------------------------
# Data Fetcher
# ---------------------------------------------------------------------------

def _fetch_instrument_data(symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch Monthly, Weekly, Daily OHLCV from yfinance for any supported instrument."""
    ticker = _INSTRUMENT_TICKERS.get(symbol.upper())
    if not ticker:
        raise ValueError(f"No yfinance ticker configured for {symbol}")
    monthly = yf.download(ticker, period="10y", interval="1mo", progress=False, auto_adjust=True)
    weekly  = yf.download(ticker, period="5y",  interval="1wk", progress=False, auto_adjust=True)
    daily   = yf.download(ticker, period="90d", interval="1d",  progress=False, auto_adjust=True)
    for df in [monthly, weekly, daily]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    return monthly, weekly, daily


def _fetch_cross_data_for(symbol: str) -> Dict[str, pd.DataFrame]:
    """Fetch relevant cross-asset tickers (weekly) for the given instrument."""
    tickers = _INSTRUMENT_CROSS_TICKERS.get(symbol.upper(), {})
    result: Dict[str, pd.DataFrame] = {}
    for key, ticker in tickers.items():
        try:
            df = yf.download(ticker, period="4wk", interval="1wk", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            result[key] = df
        except Exception as e:
            _log.debug("DashboardRouter: cross-asset fetch failed for %s (%s): %s", key, ticker, e)
    return result


# ---------------------------------------------------------------------------
# DashboardRouter
# ---------------------------------------------------------------------------

class DashboardRouter:
    """
    Multi-instrument router replicating the Gold Bias Dashboard ICT signal logic.

    Supports: XAUUSD, XAGUSD, EURUSD, NAS100, GBPUSD, GBPJPY, BTC, OIL.
    Each instrument gets an independent per-day signal cache.
    Public interface matches StrategyRouter / ResearchRouter.
    """

    _SUPPORTED = set(_INSTRUMENT_TICKERS.keys())

    def __init__(self, strategy_cfg=None, risk_cfg=None) -> None:
        self._strategy_cfg = strategy_cfg
        self._risk_cfg = risk_cfg

        # Per-symbol daily signal cache: symbol -> {date, signal, emitted}
        self._cache: Dict[str, dict] = {
            sym: {"date": None, "signal": None, "emitted": False}
            for sym in self._SUPPORTED
        }

        _log.info(
            "DashboardRouter initialised — %d instruments: %s",
            len(self._SUPPORTED),
            ", ".join(sorted(self._SUPPORTED)),
        )

    def symbols(self) -> List[str]:
        return sorted(self._SUPPORTED)

    def route(
        self,
        symbol: str,
        m15_df: pd.DataFrame,
        h1_df: pd.DataFrame,
        current_dt: datetime,
        open_positions: List,
        extra_data=None,
        d1_df=None,
    ) -> Optional[TradeSignal]:
        sym = symbol.upper().strip()
        if sym not in self._SUPPORTED:
            return None

        now_utc = current_dt.astimezone(timezone.utc) if current_dt.tzinfo else current_dt.replace(tzinfo=timezone.utc)
        today   = now_utc.date()
        cache   = self._cache[sym]

        # Refresh signal once per day at / after London open
        if cache["date"] != today and now_utc.hour >= _LONDON_OPEN_HOUR:
            self._refresh_signal(sym, today)

        # Emit pending signal once per day
        if cache["signal"] and not cache["emitted"]:
            cache["emitted"] = True
            sig = cache["signal"]
            _log.info(
                "DashboardRouter: SIGNAL [%s] %s Entry=%.5f SL=%.5f TP=%.5f RR=%.1f",
                sym, sig.direction.upper(),
                sig.entry_price, sig.stop_loss, sig.take_profit, sig.rr_ratio,
            )
            return sig

        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _refresh_signal(self, symbol: str, today: date) -> None:
        """Fetch data, compute ICT trades, cache the best one for this symbol."""
        cache = self._cache[symbol]
        cache["date"]    = today
        cache["signal"]  = None
        cache["emitted"] = False

        _log.info("DashboardRouter: [%s] refreshing daily signal for %s", symbol, today)

        try:
            monthly_df, weekly_df, daily_df = _fetch_instrument_data(symbol)
            cross_df = _fetch_cross_data_for(symbol)
        except Exception as e:
            _log.error("DashboardRouter: [%s] data fetch failed — %s", symbol, e)
            return

        try:
            bias_score = _compute_bias_score(symbol, weekly_df, cross_df)
            trades     = _generate_ict_trades(monthly_df, weekly_df, daily_df, bias_score)
        except Exception as e:
            _log.error("DashboardRouter: [%s] analysis failed — %s", symbol, e)
            return

        _log.info("DashboardRouter: [%s] bias_score=%.2f | trades=%s",
                  symbol, bias_score,
                  [(t["direction"], t["confidence"], t.get("rr1")) for t in trades])

        # Pick best actionable trade (prefer HIGH, then MEDIUM; skip WAIT/LOW)
        best = None
        for trade in trades:
            if trade["direction"] == "WAIT":
                continue
            if trade["confidence"] not in ("HIGH", "MEDIUM"):
                continue
            rr = trade.get("rr1") or 0.0
            if rr < _MIN_RR:
                _log.debug("DashboardRouter: [%s] skipping trade %d — RR %.2f < %.2f min",
                           symbol, trade["id"], rr, _MIN_RR)
                continue
            if best is None or trade["confidence"] == "HIGH":
                best = trade
            break

        if best is None:
            _log.info("DashboardRouter: [%s] no actionable trade today (all WAIT/LOW/low-RR)", symbol)
            return

        cache["signal"] = self._to_trade_signal(symbol, best, bias_score)

    @staticmethod
    def _to_trade_signal(symbol: str, trade: dict, bias_score: float) -> TradeSignal:
        direction  = "bullish" if trade["direction"] == "LONG" else "bearish"
        entry      = trade["entry"]
        stop       = trade["stop"]
        tp         = trade["target1"]
        rr         = trade.get("rr1") or 0.0
        confluence = min(0.50 + abs(bias_score) * 0.40, 0.90)

        return TradeSignal(
            symbol=symbol,
            direction=direction,
            entry_price=float(entry),
            stop_loss=float(stop),
            take_profit=float(tp) if tp else float(entry + (entry - stop) * 2),
            rr_ratio=float(rr),
            confluence_score=round(confluence, 3),
            score_bos=0.0,
            score_fvg=0.0,
            score_order_block=float(abs(bias_score)),
            score_liquidity=0.0,
            score_session=0.0,
            score_pd_zone=float(abs(bias_score)),
        )
