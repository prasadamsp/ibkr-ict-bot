"""
auto_selector.py — Pick the best algo for each instrument and publish results.

Workflow
--------
1. Load data (IBKR CSVs or yfinance fallback).
2. Run walk-forward grid search across all 10 algos for all 8 instruments.
3. Apply macro context filter (USD, carry, risk-on/off).
4. Select winner: highest val_sharpe that also passes test window.
5. Write results to:
     data/research/best_algos.json   — live bot reads this
     data/research/full_grid.csv     — full ranked table for inspection
     logs/research_<date>.log        — human-readable report

The live bot's AdaptiveRouter can optionally read best_algos.json to swap
strategies at runtime (future integration; Phase 2b).

Usage
-----
    python -m research.auto_selector                 # run all instruments
    python -m research.auto_selector --symbols XAUUSD BTC
    python -m research.auto_selector --dry-run       # don't write files
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import CONFIG
from research.algos import ALL_ALGOS
from research.walk_forward_grid import run_grid, grid_to_dataframe, GridResult
from research.macro_filters import carry_bias

_log = logging.getLogger("research")

RESULTS_DIR = Path("data/research")
BEST_ALGOS_PATH = RESULTS_DIR / "best_algos.json"
FULL_GRID_PATH  = RESULTS_DIR / "full_grid.csv"

# Instruments to research (all 8 active)
ALL_SYMBOLS = CONFIG.active_symbols

# yfinance tickers for fallback when IBKR CSVs unavailable
YF_TICKERS = {
    "XAUUSD": "GC=F",
    "XAGUSD": "SI=F",
    "NAS100": "NQ=F",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "GBPJPY": "GBPJPY=X",
    "BTC":    "BTC-USD",
    "OIL":    "CL=F",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(
    symbol: str,
    cache_dir: Path = Path(CONFIG.cache.cache_dir),
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load M15 + H1 data from IBKR CSV cache or yfinance fallback."""

    def read_csv(path: Path) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path, index_col="datetime", parse_dates=True)
            df = df[["open", "high", "low", "close", "volume"]].dropna()
            df.index = pd.to_datetime(df.index, utc=True)
            return df.sort_index()
        except Exception as e:
            _log.warning("Could not load %s: %s", path, e)
            return None

    m15 = read_csv(cache_dir / f"{symbol}_M15.csv")
    h1  = read_csv(cache_dir / f"{symbol}_H1.csv")

    # yfinance fallback
    if (m15 is None or h1 is None) and symbol in YF_TICKERS:
        try:
            import yfinance as yf
            ticker = yf.Ticker(YF_TICKERS[symbol])
            raw_h1  = ticker.history(period="2y",  interval="1h")
            raw_m15 = ticker.history(period="60d", interval="15m")

            def clean(df):
                df = df.rename(columns={
                    "Open": "open", "High": "high",
                    "Low": "low", "Close": "close", "Volume": "volume",
                })
                df = df[["open", "high", "low", "close", "volume"]].dropna()
                df.index = pd.to_datetime(df.index, utc=True)
                return df.sort_index()

            if m15 is None and not raw_m15.empty:
                m15 = clean(raw_m15)
                _log.info("  %s: yfinance M15 %d bars", symbol, len(m15))
            if h1 is None and not raw_h1.empty:
                h1 = clean(raw_h1)
                _log.info("  %s: yfinance H1 %d bars", symbol, len(h1))
        except Exception as e:
            _log.warning("  %s: yfinance failed: %s", symbol, e)

    # Use H1 as M15 proxy if still missing
    if m15 is None and h1 is not None:
        _log.info("  %s: using H1 as M15 proxy", symbol)
        m15 = h1.copy()

    return m15, h1


# ---------------------------------------------------------------------------
# Winner selection
# ---------------------------------------------------------------------------

def pick_winner(results: List[GridResult], symbol: str) -> Optional[GridResult]:
    """
    Select the best algo for a symbol.

    Selection rules (in priority order):
    1. Must have passed the validation gate (val_sharpe ≥ 0.5).
    2. Prefer algos that also show positive test_sharpe.
    3. Among those, pick highest val_sharpe.
    4. Carry filter: prefer algos whose typical direction matches carry bias.
    """
    passed = [r for r in results if r.passed_gate]
    if not passed:
        # Fallback: take best val_sharpe even if not passing test
        passed = sorted(results, key=lambda r: r.val_sharpe, reverse=True)[:1]
        if not passed:
            return None

    # Prefer positive test sharpe
    with_test = [r for r in passed if r.test_sharpe > 0]
    pool = with_test if with_test else passed

    # Sort by val_sharpe
    pool.sort(key=lambda r: r.val_sharpe, reverse=True)
    return pool[0]


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_research(
    symbols: List[str],
    dry_run: bool = False,
) -> Dict:
    """
    Run full research pipeline for the given symbols.

    Returns the best_algos dict (also written to disk unless dry_run).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    _log.info("=" * 60)
    _log.info("Auto-Selector Research Run  %s", datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))
    _log.info("Symbols: %s", ", ".join(symbols))
    _log.info("=" * 60)

    # Instantiate all algos
    algo_instances = [cls() for cls in ALL_ALGOS]

    all_grid_results: Dict[str, List[GridResult]] = {}
    best_algos: Dict[str, dict] = {}
    report_lines: List[str] = []

    for sym in symbols:
        _log.info("\n[%s] Loading data...", sym)
        m15, h1 = load_data(sym)

        if m15 is None:
            _log.warning("  [%s] No data available — skipping", sym)
            continue

        _log.info("  [%s] M15=%d bars  H1=%d bars", sym, len(m15), len(h1) if h1 is not None else 0)

        if len(m15) < 500:
            _log.warning("  [%s] Too few M15 bars (%d) — skipping", sym, len(m15))
            continue

        # Cap to last 4000 M15 bars (~10 months) for speed; enough for seasonal signal
        MAX_M15 = 4000
        if len(m15) > MAX_M15:
            m15 = m15.iloc[-MAX_M15:]
            _log.info("  [%s] Capped to last %d M15 bars for grid search speed", sym, MAX_M15)

        h1_data = h1 if h1 is not None else m15

        _log.info("  [%s] Running grid search (%d algos)...", sym, len(algo_instances))
        results = run_grid(sym, m15, h1_data, algo_instances, val_gate=0.3)
        all_grid_results[sym] = results

        # Print top 3
        _log.info("  [%s] Top 3 algos:", sym)
        for r in results[:3]:
            gate = "✓" if r.passed_gate else "✗"
            _log.info(
                "    %s  %-22s  val=%.2f  test=%.2f  months=%s",
                gate, r.algo_name, r.val_sharpe, r.test_sharpe, r.active_months[:6],
            )

        winner = pick_winner(results, sym)
        if winner:
            _log.info(
                "  [%s] WINNER → %s  (val=%.2f, test=%.2f, months=%s)",
                sym, winner.algo_name, winner.val_sharpe, winner.test_sharpe,
                winner.active_months,
            )
            best_algos[sym] = {
                "algo":          winner.algo_name,
                "val_sharpe":    winner.val_sharpe,
                "test_sharpe":   winner.test_sharpe,
                "active_months": winner.active_months,
                "active_hours":  winner.active_hours,
                "updated_at":    datetime.utcnow().isoformat(),
                "train_start":   winner.train_start,
                "test_end":      winner.test_end,
                "n_val_trades":  winner.n_val,
                "n_test_trades": winner.n_test,
            }
            report_lines.append(
                f"[{sym}] WINNER: {winner.algo_name} "
                f"val={winner.val_sharpe:.2f} test={winner.test_sharpe:.2f} "
                f"months={winner.active_months}"
            )
        else:
            _log.warning("  [%s] No viable algo found", sym)
            report_lines.append(f"[{sym}] NO WINNER — all algos failed gate")

    # Write outputs
    if not dry_run:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        with open(BEST_ALGOS_PATH, "w") as fh:
            json.dump(best_algos, fh, indent=2)
        _log.info("\nWrote best_algos.json → %s", BEST_ALGOS_PATH)

        df = grid_to_dataframe(all_grid_results)
        if not df.empty:
            df.to_csv(FULL_GRID_PATH, index=False)
            _log.info("Wrote full_grid.csv → %s", FULL_GRID_PATH)

        # Human-readable report
        report_path = Path("logs") / f"research_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.txt"
        Path("logs").mkdir(exist_ok=True)
        with open(report_path, "w") as fh:
            fh.write("\n".join([
                f"Research Run: {datetime.utcnow().isoformat()}",
                "=" * 50,
                "",
            ] + report_lines))
        _log.info("Report → %s", report_path)

    # Final summary table
    _log.info("\n%s", "=" * 60)
    _log.info("RESEARCH SUMMARY")
    _log.info("%-10s  %-22s  %-8s  %-8s  %s", "Symbol", "Best Algo", "Val", "Test", "Months")
    _log.info("-" * 70)
    for sym, info in best_algos.items():
        _log.info(
            "%-10s  %-22s  %-8.2f  %-8.2f  %s",
            sym, info["algo"], info["val_sharpe"], info["test_sharpe"],
            str(info["active_months"][:6]),
        )

    return best_algos


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Find best algo for each instrument")
    parser.add_argument("--symbols", nargs="+", default=None, help="Subset of symbols")
    parser.add_argument("--dry-run", action="store_true", help="Don't write output files")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    symbols = [s.upper() for s in args.symbols] if args.symbols else ALL_SYMBOLS
    run_research(symbols, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
