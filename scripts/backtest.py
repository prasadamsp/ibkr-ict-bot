"""
scripts/backtest.py — Vectorised algo backtester

Runs any research algo against historical M15+H1 data.
Vectorises indicator computation up front — O(n) not O(n²).

Usage:
    python scripts/backtest.py --symbol GBPUSD --algo macd_momentum
    python scripts/backtest.py --symbol XAUUSD --algo donchian_breakout
    python scripts/backtest.py --all
    python scripts/backtest.py --symbol GBPUSD --algo macd_momentum --entry limit

Output:
    Terminal report + logs/backtest_{symbol}_{algo}.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def _load_cache(symbol: str, timeframe: str) -> pd.DataFrame:
    cache_dir = ROOT / "data" / "cache"
    tf_map = {"15min": ["M15","15min"], "1h": ["H1","1h"], "1d": ["D1","1d"],
              "M15": ["M15","15min"], "H1": ["H1","1h"], "D1": ["D1","1d"]}
    for tf in tf_map.get(timeframe, [timeframe]):
        for sym in [symbol, symbol.upper(), symbol.lower()]:
            p = cache_dir / f"{sym}_{tf}.csv"
            if p.exists():
                df = pd.read_csv(p)
                for col in ["datetime","date","time","timestamp"]:
                    if col in df.columns:
                        df = df.rename(columns={col: "date"}); break
                df["date"] = pd.to_datetime(df["date"], utc=True)
                for c in ["open","high","low","close","volume"]:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                return df.sort_values("date").reset_index(drop=True)
    raise FileNotFoundError(f"No cache for {symbol} {timeframe} in {cache_dir}")


# ---------------------------------------------------------------------------
# Vectorised indicator helpers
# ---------------------------------------------------------------------------

def ema_series(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()

def atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hi, lo, cl = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    tr = pd.concat([hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# Per-algo signal generators (vectorised)
# ---------------------------------------------------------------------------

def signals_macd(m15: pd.DataFrame, h1: pd.DataFrame,
                 fast=12, slow=26, signal=9,
                 hist_threshold=0.0, sl_mult=1.5, tp_mult=2.5,
                 h1_ema_period=50) -> pd.DataFrame:
    """Returns DataFrame of signals with entry/sl/tp/direction."""
    close = m15["close"].astype(float)
    macd  = ema_series(close, fast) - ema_series(close, slow)
    sig   = ema_series(macd, signal)
    hist  = macd - sig
    atr   = atr_series(m15)

    h_prev = hist.shift(1)
    long_cross  = (h_prev < -hist_threshold) & (hist > 0)
    short_cross = (h_prev >  hist_threshold) & (hist < 0)

    # H1 trend filter — pre-compute H1 EMA, forward-fill onto M15 timeline
    if len(h1) >= h1_ema_period:
        h1_ema = ema_series(h1["close"].astype(float), h1_ema_period)
        h1["ema"] = h1_ema.values
        h1_ts = h1.set_index("date")["ema"]
        # Merge onto M15 by forward fill
        merged = pd.merge_asof(
            m15[["date"]].assign(date=m15["date"]),
            h1[["date","close","ema"]].rename(columns={"close":"h1_close"}),
            on="date", direction="backward"
        )
        h1_bull = (merged["h1_close"] > merged["ema"]).values
        h1_bear = (merged["h1_close"] < merged["ema"]).values
    else:
        h1_bull = h1_bear = np.ones(len(m15), dtype=bool)

    rows = []
    warmup = slow + signal + 5
    for i in range(warmup, len(m15) - 1):
        atr_v = atr.iloc[i]
        if np.isnan(atr_v) or atr_v <= 0:
            continue
        entry = float(m15["close"].iloc[i])
        if long_cross.iloc[i] and h1_bull[i]:
            sl = entry - sl_mult * atr_v
            tp = entry + tp_mult * atr_v
            rows.append({"bar": i, "date": m15["date"].iloc[i], "direction": "long",
                         "entry": entry, "sl": sl, "tp": tp,
                         "rr": round(tp_mult / sl_mult, 2)})
        elif short_cross.iloc[i] and h1_bear[i]:
            sl = entry + sl_mult * atr_v
            tp = entry - tp_mult * atr_v
            rows.append({"bar": i, "date": m15["date"].iloc[i], "direction": "short",
                         "entry": entry, "sl": sl, "tp": tp,
                         "rr": round(tp_mult / sl_mult, 2)})
    return pd.DataFrame(rows)


def signals_donchian(m15: pd.DataFrame, h1: pd.DataFrame,
                     period=20, sl_mult=1.0, tp_mult=2.0) -> pd.DataFrame:
    high_roll = m15["high"].astype(float).rolling(period).max()
    low_roll  = m15["low"].astype(float).rolling(period).min()
    close     = m15["close"].astype(float)
    atr       = atr_series(m15)

    long_break  = close > high_roll.shift(1)
    short_break = close < low_roll.shift(1)

    rows = []
    for i in range(period + 5, len(m15) - 1):
        atr_v = atr.iloc[i]
        if np.isnan(atr_v) or atr_v <= 0: continue
        entry = float(close.iloc[i])
        if long_break.iloc[i]:
            sl = entry - sl_mult * atr_v
            tp = entry + tp_mult * atr_v
            rows.append({"bar": i, "date": m15["date"].iloc[i], "direction": "long",
                         "entry": entry, "sl": sl, "tp": tp, "rr": round(tp_mult/sl_mult,2)})
        elif short_break.iloc[i]:
            sl = entry + sl_mult * atr_v
            tp = entry - tp_mult * atr_v
            rows.append({"bar": i, "date": m15["date"].iloc[i], "direction": "short",
                         "entry": entry, "sl": sl, "tp": tp, "rr": round(tp_mult/sl_mult,2)})
    return pd.DataFrame(rows)


def signals_ema_pullback(m15: pd.DataFrame, h1: pd.DataFrame,
                         fast=20, slow=200, trend_ema=50,
                         touch_atr_mult=0.5, sl_mult=1.0, tp_mult=2.0) -> pd.DataFrame:
    close   = m15["close"].astype(float)
    ema_f   = ema_series(close, fast)
    ema_s   = ema_series(close, slow)
    ema_tr  = ema_series(close, trend_ema)
    atr     = atr_series(m15)
    bullish = ema_f > ema_s
    bearish = ema_f < ema_s

    rows = []
    for i in range(slow + 10, len(m15) - 1):
        atr_v = atr.iloc[i]
        if np.isnan(atr_v) or atr_v <= 0: continue
        entry = float(close.iloc[i])
        ema_val = float(ema_tr.iloc[i])
        touch_dist = touch_atr_mult * atr_v
        near_ema = abs(entry - ema_val) <= touch_dist

        if bullish.iloc[i] and near_ema:
            sl = entry - sl_mult * atr_v
            tp = entry + tp_mult * atr_v
            rows.append({"bar": i, "date": m15["date"].iloc[i], "direction": "long",
                         "entry": entry, "sl": sl, "tp": tp, "rr": round(tp_mult/sl_mult,2)})
        elif bearish.iloc[i] and near_ema:
            sl = entry + sl_mult * atr_v
            tp = entry - tp_mult * atr_v
            rows.append({"bar": i, "date": m15["date"].iloc[i], "direction": "short",
                         "entry": entry, "sl": sl, "tp": tp, "rr": round(tp_mult/sl_mult,2)})
    return pd.DataFrame(rows)


SIGNAL_FNS = {
    "macd_momentum":    signals_macd,
    "donchian_breakout": signals_donchian,
    "ema_pullback":     signals_ema_pullback,
}


# ---------------------------------------------------------------------------
# Fill simulator
# ---------------------------------------------------------------------------

def simulate_fills(signals: pd.DataFrame, m15: pd.DataFrame,
                   min_rr: float, entry_mode: str) -> pd.DataFrame:
    """Given signal rows, simulate fills and exits. Returns trade log."""
    trades = []
    last_exit_bar = -1  # prevent overlapping trades

    for _, sig in signals.iterrows():
        i = int(sig["bar"])
        if i <= last_exit_bar:
            continue
        if sig["rr"] < min_rr:
            continue

        # Fill
        if entry_mode == "market":
            if i + 1 >= len(m15): continue
            fill = float(m15["open"].iloc[i + 1])
            fill_bar = i + 1
        else:  # limit
            fill = None
            fill_bar = None
            for j in range(i + 1, min(i + 5, len(m15))):
                bar = m15.iloc[j]
                if sig["direction"] == "long" and float(bar["low"]) <= sig["entry"]:
                    fill, fill_bar = sig["entry"], j; break
                elif sig["direction"] == "short" and float(bar["high"]) >= sig["entry"]:
                    fill, fill_bar = sig["entry"], j; break
            if fill is None: continue

        # Recalculate sl/tp relative to fill for market orders
        atr_at_signal = abs(sig["entry"] - sig["sl"])  # same distance
        if sig["direction"] == "long":
            sl = fill - atr_at_signal
            tp = fill + atr_at_signal * sig["rr"]
        else:
            sl = fill + atr_at_signal
            tp = fill - atr_at_signal * sig["rr"]

        # Scan for exit
        exit_price, exit_reason, hold_bars = None, "timeout", 0
        max_hold = 50
        for j in range(fill_bar + 1, min(fill_bar + max_hold + 1, len(m15))):
            bar = m15.iloc[j]
            hi, lo = float(bar["high"]), float(bar["low"])
            hold_bars = j - fill_bar
            if sig["direction"] == "long":
                if lo <= sl:   exit_price, exit_reason = sl, "sl";   break
                if hi >= tp:   exit_price, exit_reason = tp, "tp";   break
            else:
                if hi >= sl:   exit_price, exit_reason = sl, "sl";   break
                if lo <= tp:   exit_price, exit_reason = tp, "tp";   break

        if exit_price is None:
            j = min(fill_bar + max_hold, len(m15) - 1)
            exit_price = float(m15["close"].iloc[j])
            hold_bars  = j - fill_bar

        risk = abs(fill - sl)
        if risk <= 0: continue

        pnl_r = ((exit_price - fill) / risk) if sig["direction"] == "long" \
                else ((fill - exit_price) / risk)

        trades.append({
            "date":        str(sig["date"]),
            "direction":   sig["direction"],
            "entry":       round(fill, 6),
            "sl":          round(sl, 6),
            "tp":          round(tp, 6),
            "exit":        round(exit_price, 6),
            "exit_reason": exit_reason,
            "rr":          sig["rr"],
            "pnl_r":       round(pnl_r, 3),
            "hold_bars":   hold_bars,
        })
        last_exit_bar = fill_bar + hold_bars

    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(df: pd.DataFrame, symbol: str, algo: str):
    if df.empty:
        print("  No trades generated.")
        return
    wins   = df[df["pnl_r"] > 0]
    losses = df[df["pnl_r"] <= 0]
    win_rate   = len(wins) / len(df) * 100
    expectancy = df["pnl_r"].mean()
    total_r    = df["pnl_r"].sum()
    sharpe     = (expectancy / df["pnl_r"].std() * np.sqrt(len(df))) if df["pnl_r"].std() > 0 else 0
    cumulative = df["pnl_r"].cumsum()
    max_dd     = (cumulative - cumulative.cummax()).min()
    by_exit    = df["exit_reason"].value_counts().to_dict()

    print(f"\n  RESULTS ({len(df)} trades)")
    print(f"  Win rate:    {win_rate:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"  Avg win:     +{wins['pnl_r'].mean():.2f}R" if len(wins) else "  Avg win:     —")
    print(f"  Avg loss:    {losses['pnl_r'].mean():.2f}R" if len(losses) else "  Avg loss:    —")
    print(f"  Expectancy:  {expectancy:.3f}R per trade")
    print(f"  Total P&L:   {total_r:.1f}R")
    print(f"  Sharpe:      {sharpe:.2f}")
    print(f"  Max DD:      {max_dd:.2f}R")
    print(f"  Exits:       {by_exit}")
    print(f"  Avg hold:    {df['hold_bars'].mean():.1f} bars ({df['hold_bars'].mean()*15/60:.1f}h)")

    out = ROOT / "logs" / f"backtest_{symbol}_{algo}.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"  Trade log → {out}")

    return {
        "symbol": symbol, "algo": algo, "trades": len(df),
        "win_rate": round(win_rate,1), "expectancy": round(expectancy,3),
        "total_r": round(total_r,2), "sharpe": round(sharpe,2), "max_dd": round(max_dd,2),
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_backtest(symbol, algo_name, algo_params, min_rr=1.5, entry_mode="market"):
    print(f"\n{'='*60}")
    print(f"  BACKTEST: {symbol} | algo={algo_name}")
    print(f"  entry={entry_mode}  min_rr={min_rr}  params={algo_params}")
    print(f"{'='*60}")

    try:
        m15 = _load_cache(symbol, "M15")
        h1  = _load_cache(symbol, "H1")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}"); return {}

    print(f"  M15 bars: {len(m15)}  H1 bars: {len(h1)}")
    print(f"  Period: {m15['date'].iloc[0]} → {m15['date'].iloc[-1]}")

    gen_fn = SIGNAL_FNS.get(algo_name)
    if gen_fn is None:
        print(f"  Vectorised generator not available for '{algo_name}' — skipping.")
        return {}

    signals = gen_fn(m15, h1, **algo_params) if algo_params else gen_fn(m15, h1)
    print(f"  Raw signals: {len(signals)}")

    if signals.empty:
        print("  No signals generated."); return {}

    trades_df = simulate_fills(signals, m15, min_rr, entry_mode)
    result = print_report(trades_df, symbol, algo_name)
    return result or {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",  default=None)
    parser.add_argument("--algo",    default=None)
    parser.add_argument("--params",  default="{}")
    parser.add_argument("--min-rr",  type=float, default=1.5)
    parser.add_argument("--entry",   default="market", choices=["market","limit"])
    parser.add_argument("--all",     action="store_true")
    args = parser.parse_args()

    best_path = ROOT / "data" / "research" / "best_algos.json"

    if args.all:
        with open(best_path) as f:
            best = json.load(f)
        results = []
        for sym, info in best.items():
            r = run_backtest(sym, info["algo"], info.get("params",{}), args.min_rr, args.entry)
            if r: results.append(r)

        if results:
            print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
            print(f"  {'Symbol':<10} {'Algo':<22} {'#':>4} {'Win%':>6} {'Exp':>7} {'Sharpe':>7} {'MaxDD':>7}")
            print(f"  {'-'*10} {'-'*22} {'-'*4} {'-'*6} {'-'*7} {'-'*7} {'-'*7}")
            for r in results:
                print(f"  {r['symbol']:<10} {r['algo']:<22} {r['trades']:>4} "
                      f"{r['win_rate']:>5.1f}% {r['expectancy']:>7.3f} "
                      f"{r['sharpe']:>7.2f} {r['max_dd']:>6.2f}R")
    else:
        if not args.symbol or not args.algo:
            if best_path.exists():
                with open(best_path) as f: best = json.load(f)
                sym  = args.symbol or list(best.keys())[0]
                info = best.get(sym, {})
                algo = args.algo or info.get("algo","macd_momentum")
                params = info.get("params",{})
            else:
                parser.print_help(); sys.exit(1)
        else:
            sym, algo, params = args.symbol, args.algo, json.loads(args.params)
        run_backtest(sym, algo, params, args.min_rr, args.entry)


if __name__ == "__main__":
    main()
