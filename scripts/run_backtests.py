"""
Walk-Forward Backtest Runner
=============================
Loads cached CSV data and runs walk-forward backtests for all available symbols.

Usage:
    cd /path/to/IBKR
    python scripts/run_backtests.py
    python scripts/run_backtests.py --symbols XAUUSD EURUSD
    python scripts/run_backtests.py --equity 100000 --sharpe 1.0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from backtesting.walk_forward import WalkForwardBacktester
from config.settings import CONFIG


CACHE_DIR = Path(CONFIG.cache.cache_dir)


def load_csv(symbol: str, tf: str) -> pd.DataFrame | None:
    """Load a cached CSV file and return a clean OHLCV DataFrame, or None."""
    path = CACHE_DIR / f"{symbol}_{tf}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col="datetime", parse_dates=True)
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()
        return df
    except Exception as e:
        print(f"  [WARN] Could not load {path}: {e}")
        return None


def fetch_yfinance(yf_ticker: str, label: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Fetch OHLCV from Yahoo Finance. Returns (m15_df, h1_df)."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(yf_ticker)
        df_h1 = ticker.history(period="1y", interval="1h")
        df_m15 = ticker.history(period="60d", interval="15m")  # YF limits 15m to 60d

        def clean(df):
            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume"
            })
            df = df[["open", "high", "low", "close", "volume"]].dropna()
            df.index = pd.to_datetime(df.index, utc=True)
            return df.sort_index()

        m15 = clean(df_m15) if not df_m15.empty else None
        h1  = clean(df_h1)  if not df_h1.empty  else None
        return m15, h1
    except ImportError:
        print(f"  [WARN] yfinance not installed")
        return None, None
    except Exception as e:
        print(f"  [WARN] yfinance {label} ({yf_ticker}) failed: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Run walk-forward backtests from cached CSVs")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Symbols to backtest (default: all available)")
    parser.add_argument("--equity", type=float, default=50_000,
                        help="Initial equity per backtest run (default: 50000)")
    parser.add_argument("--sharpe", type=float, default=0.8,
                        help="Minimum val Sharpe to pass gate (default: 0.8)")
    parser.add_argument("--out", default="logs/walk_forward_results.csv",
                        help="CSV output path for summary results")
    args = parser.parse_args()

    all_symbols = CONFIG.active_symbols
    symbols = args.symbols or all_symbols

    print(f"\nWalk-Forward Backtest Runner")
    print(f"{'─'*50}")
    print(f"Cache dir   : {CACHE_DIR.resolve()}")
    print(f"Symbols     : {', '.join(symbols)}")
    print(f"Initial eq  : ${args.equity:,.0f}")
    print(f"Sharpe gate : {args.sharpe}")
    print(f"{'─'*50}\n")

    # Build data map
    data_map: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    skip_reasons: dict[str, str] = {}

    for sym in symbols:
        m15 = load_csv(sym, "M15")
        h1  = load_csv(sym, "H1")

        # Fallback: fetch from Yahoo Finance for symbols without IBKR data
        YF_TICKERS = {"BTC": "BTC-USD", "OIL": "CL=F"}
        if sym in YF_TICKERS and (m15 is None or h1 is None):
            yf_ticker = YF_TICKERS[sym]
            print(f"  {sym}: IBKR data unavailable. Trying Yahoo Finance ({yf_ticker})...")
            yf_m15, yf_h1 = fetch_yfinance(yf_ticker, sym)
            if m15 is None and yf_m15 is not None:
                m15 = yf_m15
                print(f"    ✓ yfinance M15: {len(yf_m15)} bars (60-day limit)")
            if h1 is None and yf_h1 is not None:
                h1 = yf_h1
                print(f"    ✓ yfinance H1 : {len(yf_h1)} bars")

        if m15 is None and h1 is not None:
            # Use H1 as proxy for M15 when M15 not available
            print(f"  {sym}: M15 not found — using H1 bars as M15 proxy (reduced accuracy)")
            m15 = h1.copy()

        if m15 is None or h1 is None:
            skip_reasons[sym] = "missing data (both M15 and H1)"
            continue

        min_bars = 1500
        if len(m15) < min_bars:
            skip_reasons[sym] = f"too few bars ({len(m15)} M15, need {min_bars})"
            continue

        data_map[sym] = (m15, h1)
        m15_bars = len(m15)
        h1_bars = len(h1)
        print(f"  {sym:8}: M15={m15_bars:>6} bars  H1={h1_bars:>5} bars  ✓ Ready")

    if skip_reasons:
        print()
        for sym, reason in skip_reasons.items():
            print(f"  [SKIP] {sym}: {reason}")

    if not data_map:
        print("\nNo symbols with sufficient data. Exiting.")
        sys.exit(1)

    print(f"\nRunning walk-forward for {len(data_map)} symbol(s)...\n")

    wf = WalkForwardBacktester(
        initial_equity=args.equity,
        min_sharpe=args.sharpe,
    )

    results = wf.run_all(list(data_map.keys()), data_map)

    if results:
        wf.to_csv(results, path=args.out)


if __name__ == "__main__":
    main()
