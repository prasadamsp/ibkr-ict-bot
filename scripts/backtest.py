"""
scripts/backtest.py — Quick algo backtester

Runs any research algo against historical M15+H1 data bar-by-bar,
simulates fills at next-bar open (market order) or limit fill logic,
and outputs a performance report with win rate, Sharpe, drawdown.

Usage:
    python scripts/backtest.py --symbol GBPUSD --algo macd_momentum
    python scripts/backtest.py --symbol XAUUSD --algo donchian_breakout
    python scripts/backtest.py --symbol GBPUSD --algo macd_momentum --no-h1-filter
    python scripts/backtest.py --all   # run all symbols from best_algos.json

Output:
    Terminal report + logs/backtest_{symbol}_{algo}.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make sure project root is on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from research.algos.donchian   import DonchianBreakoutAlgo
from research.algos.ema_pullback import EMAPullbackAlgo
from research.algos.macd_momentum import MACDMomentumAlgo
from research.algos.bb_rsi     import BBRSIAlgo
from research.algos.rsi_extreme import RSIExtremeAlgo
from research.algos.zscore     import ZScoreReversionAlgo
from research.algos.ma_crossover import MACrossoverAlgo
from research.algos.ict_fvg    import ICTFVGAlgo

ALGO_REGISTRY = {
    "donchian_breakout": DonchianBreakoutAlgo,
    "ema_pullback":      EMAPullbackAlgo,
    "macd_momentum":     MACDMomentumAlgo,
    "bb_rsi":            BBRSIAlgo,
    "rsi_extreme":       RSIExtremeAlgo,
    "zscore_reversion":  ZScoreReversionAlgo,
    "ma_crossover":      MACrossoverAlgo,
    "ict_fvg":           ICTFVGAlgo,
}

# ---------------------------------------------------------------------------
# Data loading — reads from local cache (Twelve Data CSV format)
# ---------------------------------------------------------------------------

def _load_cache(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load cached OHLCV data.

    Tries multiple naming conventions:
      GBPUSD_M15.csv, GBPUSD_15min.csv, GBPUSD_H1.csv, GBPUSD_1h.csv
    """
    cache_dir = ROOT / "data" / "cache"

    # Normalise timeframe aliases
    tf_aliases = {
        "15min": ["M15", "15min", "15m"],
        "1h":    ["H1",  "1h",   "60min"],
        "1d":    ["D1",  "1d",   "daily"],
        "M15":   ["M15", "15min", "15m"],
        "H1":    ["H1",  "1h",   "60min"],
        "D1":    ["D1",  "1d",   "daily"],
    }
    candidates = tf_aliases.get(timeframe, [timeframe])

    for tf in candidates:
        for sym in [symbol, symbol.upper(), symbol.lower()]:
            p = cache_dir / f"{sym}_{tf}.csv"
            if p.exists():
                df = pd.read_csv(p)
                # Handle various date column names
                for col in ["datetime", "date", "time", "timestamp"]:
                    if col in df.columns:
                        df = df.rename(columns={col: "date"})
                        break
                df["date"] = pd.to_datetime(df["date"])
                for c in ["open", "high", "low", "close", "volume"]:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.sort_values("date").reset_index(drop=True)
                return df

    raise FileNotFoundError(
        f"No cached data for {symbol} {timeframe}. "
        f"Looked for: {[f'{symbol}_{tf}.csv' for tf in candidates]} in {cache_dir}"
    )


# ---------------------------------------------------------------------------
# Core backtest engine
# ---------------------------------------------------------------------------

def run_backtest(
    symbol: str,
    algo_name: str,
    algo_params: dict,
    min_rr: float = 1.5,
    entry_mode: str = "market",  # "market" = fill at next bar open; "limit" = fill if price revisits
) -> dict:
    """Run backtest and return metrics dict."""

    print(f"\n{'='*60}")
    print(f"  BACKTEST: {symbol} | algo={algo_name}")
    print(f"  entry={entry_mode}  min_rr={min_rr}  params={algo_params}")
    print(f"{'='*60}")

    # Load data
    try:
        m15 = _load_cache(symbol, "15min")
        h1  = _load_cache(symbol, "1h")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return {}

    print(f"  M15 bars: {len(m15)}  H1 bars: {len(h1)}")
    print(f"  Period: {m15['date'].iloc[0]} → {m15['date'].iloc[-1]}")

    # Instantiate algo
    cls = ALGO_REGISTRY.get(algo_name)
    if cls is None:
        print(f"  ERROR: unknown algo '{algo_name}'")
        return {}
    algo = cls(**algo_params) if algo_params else cls()

    # --- Bar-by-bar simulation ---
    trades = []
    warmup = max(100, algo.slow if hasattr(algo, "slow") else 50)

    for i in range(warmup, len(m15) - 1):
        m15_slice = m15.iloc[: i + 1].copy()

        # Align H1 slice to this M15 bar's timestamp
        bar_time = m15.iloc[i]["date"]
        h1_slice = h1[h1["date"] <= bar_time].copy()
        if len(h1_slice) < 2:
            continue

        try:
            sig = algo.generate(m15_slice, h1_slice, idx=len(m15_slice) - 1)
        except Exception:
            continue

        if sig is None:
            continue
        if sig.rr < min_rr:
            continue

        # Simulate fill
        next_bar = m15.iloc[i + 1]
        if entry_mode == "market":
            fill_price = float(next_bar["open"])
        else:
            # Limit: fill only if price touches entry within next 4 bars
            fill_price = None
            for j in range(i + 1, min(i + 5, len(m15))):
                bar = m15.iloc[j]
                if sig.direction == "long" and float(bar["low"]) <= sig.entry:
                    fill_price = sig.entry
                    break
                elif sig.direction == "short" and float(bar["high"]) >= sig.entry:
                    fill_price = sig.entry
                    break
            if fill_price is None:
                continue  # limit never filled — skip

        # Simulate exit: scan forward for SL or TP hit
        exit_price = None
        exit_reason = "open"
        hold_bars = 0
        max_hold = 50  # max 50 bars (~12.5 hours)

        for j in range(i + 1, min(i + max_hold + 1, len(m15))):
            bar = m15.iloc[j]
            hold_bars = j - i
            hi, lo = float(bar["high"]), float(bar["low"])

            if sig.direction == "long":
                if lo <= sig.sl:
                    exit_price = sig.sl
                    exit_reason = "sl"
                    break
                elif hi >= sig.tp:
                    exit_price = sig.tp
                    exit_reason = "tp"
                    break
            else:
                if hi >= sig.sl:
                    exit_price = sig.sl
                    exit_reason = "sl"
                    break
                elif lo <= sig.tp:
                    exit_price = sig.tp
                    exit_reason = "tp"
                    break

        if exit_price is None:
            # Still open at max_hold — exit at last close
            exit_price = float(m15.iloc[min(i + max_hold, len(m15) - 1)]["close"])
            exit_reason = "timeout"

        # P&L in price terms (normalised to entry)
        if sig.direction == "long":
            pnl_r = (exit_price - fill_price) / abs(fill_price - sig.sl) if abs(fill_price - sig.sl) > 0 else 0
        else:
            pnl_r = (fill_price - exit_price) / abs(sig.sl - fill_price) if abs(sig.sl - fill_price) > 0 else 0

        trades.append({
            "date":        str(bar_time),
            "direction":   sig.direction,
            "entry":       round(fill_price, 6),
            "sl":          round(sig.sl, 6),
            "tp":          round(sig.tp, 6),
            "exit":        round(exit_price, 6),
            "exit_reason": exit_reason,
            "rr":          sig.rr,
            "pnl_r":       round(pnl_r, 3),
            "hold_bars":   hold_bars,
        })

        # Skip ahead to avoid re-entering during this trade
        i_skip = i + hold_bars

    # --- Metrics ---
    if not trades:
        print("  No trades generated.")
        return {"symbol": symbol, "algo": algo_name, "trades": 0}

    df = pd.DataFrame(trades)
    wins   = df[df["pnl_r"] > 0]
    losses = df[df["pnl_r"] <= 0]

    win_rate   = len(wins) / len(df) * 100
    avg_win    = wins["pnl_r"].mean() if len(wins) else 0
    avg_loss   = losses["pnl_r"].mean() if len(losses) else 0
    total_r    = df["pnl_r"].sum()
    expectancy = df["pnl_r"].mean()

    # Sharpe (annualised, assuming ~4 trades/day × 252 days)
    if df["pnl_r"].std() > 0:
        sharpe = (expectancy / df["pnl_r"].std()) * np.sqrt(len(df))
    else:
        sharpe = 0.0

    # Max drawdown in R
    cumulative = df["pnl_r"].cumsum()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak)
    max_dd = drawdown.min()

    by_exit = df["exit_reason"].value_counts().to_dict()

    print(f"\n  RESULTS ({len(df)} trades)")
    print(f"  Win rate:    {win_rate:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"  Avg win:     +{avg_win:.2f}R")
    print(f"  Avg loss:    {avg_loss:.2f}R")
    print(f"  Expectancy:  {expectancy:.3f}R per trade")
    print(f"  Total P&L:   {total_r:.1f}R")
    print(f"  Sharpe:      {sharpe:.2f}")
    print(f"  Max DD:      {max_dd:.2f}R")
    print(f"  Exits:       {by_exit}")
    print(f"  Avg hold:    {df['hold_bars'].mean():.1f} bars ({df['hold_bars'].mean()*15/60:.1f}h)")

    # Save trade log
    out_dir = ROOT / "logs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"backtest_{symbol}_{algo_name}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Trade log → {out_path}")

    return {
        "symbol":     symbol,
        "algo":       algo_name,
        "trades":     len(df),
        "win_rate":   round(win_rate, 1),
        "expectancy": round(expectancy, 3),
        "total_r":    round(total_r, 2),
        "sharpe":     round(sharpe, 2),
        "max_dd":     round(max_dd, 2),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Backtest research algos")
    parser.add_argument("--symbol",  default=None, help="Symbol to test e.g. GBPUSD")
    parser.add_argument("--algo",    default=None, help="Algo name e.g. macd_momentum")
    parser.add_argument("--params",  default="{}", help="JSON algo params override")
    parser.add_argument("--min-rr",  type=float, default=1.5)
    parser.add_argument("--entry",   default="market", choices=["market", "limit"])
    parser.add_argument("--all",     action="store_true", help="Run all from best_algos.json")
    args = parser.parse_args()

    best_algos_path = ROOT / "data" / "research" / "best_algos.json"

    if args.all:
        with open(best_algos_path) as f:
            best = json.load(f)
        results = []
        for sym, info in best.items():
            r = run_backtest(
                symbol=sym,
                algo_name=info["algo"],
                algo_params=info.get("params", {}),
                min_rr=args.min_rr,
                entry_mode=args.entry,
            )
            if r:
                results.append(r)

        print(f"\n{'='*60}")
        print("  SUMMARY — ALL SYMBOLS")
        print(f"{'='*60}")
        print(f"  {'Symbol':<10} {'Algo':<22} {'Trades':>6} {'WinRate':>8} {'Expect':>8} {'Sharpe':>7} {'MaxDD':>7}")
        print(f"  {'-'*10} {'-'*22} {'-'*6} {'-'*8} {'-'*8} {'-'*7} {'-'*7}")
        for r in results:
            print(
                f"  {r['symbol']:<10} {r['algo']:<22} {r['trades']:>6} "
                f"{r['win_rate']:>7.1f}% {r['expectancy']:>8.3f} "
                f"{r['sharpe']:>7.2f} {r['max_dd']:>7.2f}R"
            )
    else:
        if not args.symbol or not args.algo:
            # Default: read from best_algos.json
            if best_algos_path.exists():
                with open(best_algos_path) as f:
                    best = json.load(f)
                sym   = args.symbol or list(best.keys())[0]
                info  = best.get(sym, {})
                algo  = args.algo or info.get("algo", "macd_momentum")
                params = info.get("params", {})
            else:
                parser.print_help()
                sys.exit(1)
        else:
            sym    = args.symbol
            algo   = args.algo
            params = json.loads(args.params)

        run_backtest(sym, algo, params, min_rr=args.min_rr, entry_mode=args.entry)


if __name__ == "__main__":
    main()
