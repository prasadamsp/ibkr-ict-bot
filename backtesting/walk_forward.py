"""
Walk-Forward Backtester — temporal out-of-sample validation.

Splits each symbol's data into three non-overlapping windows:

    ┌──────────────────────────────────────────────────────┐
    │  TRAIN (60%)  │  VALIDATION (20%)  │  TEST (20%)     │
    └──────────────────────────────────────────────────────┘
    ← earliest                                  latest →

Workflow
--------
1. Run Backtester on the TRAIN window (strategy development).
2. Run Backtester on the VAL window   (hyper-parameter gate).
3. If val Sharpe ≥ min_sharpe  → run on TEST window  (true OOS estimate).
   Else                         → flag as FAIL, skip test.

The gate prevents test-set peeking: the test result is only revealed when the
strategy has already demonstrated acceptable val performance.

Usage
-----
    wf = WalkForwardBacktester(initial_equity=50_000, min_sharpe=0.8)
    result = wf.run("XAUUSD", m15_df, h1_df)
    wf.print_report(result)

    # Multi-symbol
    results = wf.run_all(symbols, data_map)
    wf.to_csv(results)
"""

import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtesting.backtester import Backtester, BacktestResult


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardResult:
    """Holds all three split results plus the gate verdict for one symbol."""

    symbol: str

    # Date boundary strings (YYYY-MM-DD) for each window
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str

    # Backtest outcomes
    train_result: BacktestResult
    val_result: BacktestResult
    test_result: Optional[BacktestResult]  # None when gate not passed

    # Gate
    passed_gate: bool   # True  → val Sharpe >= min_sharpe
    min_sharpe: float


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

class WalkForwardBacktester:
    """
    Temporal walk-forward validation wrapper around the existing Backtester.

    Parameters
    ----------
    initial_equity:
        Starting equity used for every split's Backtester run.
    min_sharpe:
        Minimum annualised Sharpe ratio on the validation window for a
        symbol to be considered "production-ready" (gate threshold).
    train_pct:
        Fraction of total bars assigned to the training window.
    val_pct:
        Fraction of total bars assigned to the validation window.
    test_pct:
        Fraction of total bars assigned to the test window.
        (train_pct + val_pct + test_pct must equal 1.0)
    """

    MIN_M15_BARS = 500  # minimum bars per split before we skip with a warning

    def __init__(
        self,
        initial_equity: float = 50_000,
        min_sharpe: float = 0.8,
        train_pct: float = 0.6,
        val_pct: float = 0.2,
        test_pct: float = 0.2,
    ):
        if not abs(train_pct + val_pct + test_pct - 1.0) < 1e-9:
            raise ValueError(
                f"train_pct + val_pct + test_pct must equal 1.0 "
                f"(got {train_pct + val_pct + test_pct:.4f})"
            )

        self.initial_equity = initial_equity
        self.min_sharpe = min_sharpe
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.test_pct = test_pct

    # ------------------------------------------------------------------
    # Core split logic
    # ------------------------------------------------------------------

    def _split_dataframe(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split a time-indexed DataFrame into train / val / test slices.

        The split is purely temporal: no shuffling, no leakage.
        Both the M15 and H1 DataFrames must be split at the *same* datetime
        boundaries, which is why callers pass datetime cutoffs and filter by
        index rather than using integer iloc slicing.

        Returns
        -------
        (train_df, val_df, test_df)
        """
        n = len(df)
        train_end_idx = int(n * self.train_pct)
        val_end_idx   = int(n * (self.train_pct + self.val_pct))

        train_df = df.iloc[:train_end_idx].copy()
        val_df   = df.iloc[train_end_idx:val_end_idx].copy()
        test_df  = df.iloc[val_end_idx:].copy()

        return train_df, val_df, test_df

    def _cutoff_datetimes(
        self, m15_df: pd.DataFrame
    ) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Derive the two boundary datetimes (val_start, test_start) from M15 data.

        Both H1 and M15 DataFrames will be sliced using these same boundaries so
        the splits are aligned in calendar time rather than bar count.
        """
        n = len(m15_df)
        train_end_idx = int(n * self.train_pct)
        val_end_idx   = int(n * (self.train_pct + self.val_pct))

        val_start_dt  = m15_df.index[train_end_idx]
        test_start_dt = m15_df.index[val_end_idx]

        return val_start_dt, test_start_dt

    def _split_by_datetime(
        self,
        df: pd.DataFrame,
        val_start_dt: pd.Timestamp,
        test_start_dt: pd.Timestamp,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Slice DataFrame using datetime boundaries derived from M15 data.

        This ensures M15 and H1 are cut at the same calendar moment.
        """
        train_df = df[df.index < val_start_dt].copy()
        val_df   = df[(df.index >= val_start_dt) & (df.index < test_start_dt)].copy()
        test_df  = df[df.index >= test_start_dt].copy()

        return train_df, val_df, test_df

    # ------------------------------------------------------------------
    # Single-symbol run
    # ------------------------------------------------------------------

    def run(
        self,
        symbol: str,
        m15_df: pd.DataFrame,
        h1_df: pd.DataFrame,
    ) -> WalkForwardResult:
        """
        Run walk-forward backtest for a single symbol.

        Parameters
        ----------
        symbol:
            Instrument ticker, e.g. "XAUUSD".
        m15_df:
            15-minute OHLCV DataFrame with a datetime index, sorted ascending.
        h1_df:
            1-hour OHLCV DataFrame with a datetime index, sorted ascending.

        Returns
        -------
        WalkForwardResult with train, val, and (optionally) test outcomes.

        Raises
        ------
        ValueError
            If the total M15 dataset has fewer than 3 × MIN_M15_BARS rows, making
            at least one split unusably small.
        """
        if not isinstance(m15_df.index, pd.DatetimeIndex):
            m15_df = m15_df.copy()
            m15_df.index = pd.to_datetime(m15_df.index)
        if not isinstance(h1_df.index, pd.DatetimeIndex):
            h1_df = h1_df.copy()
            h1_df.index = pd.to_datetime(h1_df.index)

        m15_df = m15_df.sort_index()
        h1_df  = h1_df.sort_index()

        total_bars = len(m15_df)
        if total_bars < self.MIN_M15_BARS * 3:
            raise ValueError(
                f"{symbol}: total M15 bars ({total_bars}) is below the minimum "
                f"required for a 3-way split ({self.MIN_M15_BARS * 3}). Skipping."
            )

        # Determine calendar cutoffs from M15; apply to both timeframes
        val_start_dt, test_start_dt = self._cutoff_datetimes(m15_df)

        m15_train, m15_val, m15_test = self._split_by_datetime(
            m15_df, val_start_dt, test_start_dt
        )
        h1_train, h1_val, h1_test = self._split_by_datetime(
            h1_df, val_start_dt, test_start_dt
        )

        # Validate individual split sizes
        for split_name, split_df in [
            ("TRAIN", m15_train),
            ("VAL",   m15_val),
            ("TEST",  m15_test),
        ]:
            if len(split_df) < self.MIN_M15_BARS:
                warnings.warn(
                    f"{symbol} {split_name} split has only {len(split_df)} M15 bars "
                    f"(minimum {self.MIN_M15_BARS}). Results may be unreliable.",
                    UserWarning,
                    stacklevel=2,
                )

        # --- Train ---
        bt_train = Backtester(initial_equity=self.initial_equity)
        train_result = bt_train.run(symbol, m15_train, h1_train)

        # --- Validation ---
        bt_val = Backtester(initial_equity=self.initial_equity)
        val_result = bt_val.run(symbol, m15_val, h1_val)

        # --- Gate check ---
        passed_gate = val_result.sharpe_ratio >= self.min_sharpe

        # --- Test (only if gate passed) ---
        test_result: Optional[BacktestResult] = None
        if passed_gate:
            bt_test = Backtester(initial_equity=self.initial_equity)
            test_result = bt_test.run(symbol, m15_test, h1_test)

        return WalkForwardResult(
            symbol=symbol,
            train_start=train_result.start_date,
            train_end=train_result.end_date,
            val_start=val_result.start_date,
            val_end=val_result.end_date,
            test_start=(test_result.start_date if test_result else str(m15_test.index[0].date())),
            test_end=(test_result.end_date   if test_result else str(m15_test.index[-1].date())),
            train_result=train_result,
            val_result=val_result,
            test_result=test_result,
            passed_gate=passed_gate,
            min_sharpe=self.min_sharpe,
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self, result: WalkForwardResult):
        """
        Print a formatted three-section report comparing train / val / test metrics.

        The gate verdict (PASS / FAIL) is prominently displayed with the val Sharpe
        vs the threshold.
        """
        r = result
        sep  = "═" * 72
        thin = "─" * 72

        # Gate verdict banner
        gate_label = "✔  PASS" if r.passed_gate else "✘  FAIL"
        gate_detail = (
            f"Val Sharpe {r.val_result.sharpe_ratio:.3f} "
            f"{'≥' if r.passed_gate else '<'} "
            f"threshold {r.min_sharpe:.2f}"
        )

        print(f"\n{sep}")
        print(f"  WALK-FORWARD REPORT — {r.symbol}")
        print(sep)
        print(f"  Gate verdict : {gate_label}  ({gate_detail})")
        print(thin)

        # Column headers
        col_w = 18
        print(
            f"  {'Metric':<24}"
            f"{'TRAIN':>{col_w}}"
            f"{'VAL':>{col_w}}"
            f"{'TEST':>{col_w}}"
        )
        print(
            f"  {'Period':<24}"
            f"{r.train_start+' → '+r.train_end:>{col_w}}"
            f"{r.val_start+' → '+r.val_end:>{col_w}}"
        )
        print(thin)

        tr = r.train_result
        vr = r.val_result
        te = r.test_result  # may be None

        def _fmt_test(value, fmt):
            return format(value, fmt) if te is not None else "  —"

        rows = [
            ("M15 bars",
             f"{tr.total_trades:>0}",       # reuse via trade count as proxy; actual bars not stored
             f"{vr.total_trades:>0}",
             _fmt_test(te.total_trades if te else 0, "")),
            ("Total Trades",
             f"{tr.total_trades}",
             f"{vr.total_trades}",
             _fmt_test(te.total_trades if te else 0, "")),
            ("Win Rate",
             f"{tr.win_rate*100:.1f}%",
             f"{vr.win_rate*100:.1f}%",
             _fmt_test(te.win_rate * 100 if te else 0.0, ".1f") + ("%" if te else "")),
            ("Net P&L",
             f"${tr.net_pnl:+,.0f}",
             f"${vr.net_pnl:+,.0f}",
             _fmt_test(te.net_pnl if te else 0.0, "+,.0f")),
            ("Net P&L %",
             f"{tr.net_pnl_pct:+.2f}%",
             f"{vr.net_pnl_pct:+.2f}%",
             _fmt_test(te.net_pnl_pct if te else 0.0, "+.2f") + ("%" if te else "")),
            ("Profit Factor",
             f"{tr.profit_factor:.2f}",
             f"{vr.profit_factor:.2f}",
             _fmt_test(te.profit_factor if te else 0.0, ".2f")),
            ("Max Drawdown %",
             f"{tr.max_drawdown_pct:.2f}%",
             f"{vr.max_drawdown_pct:.2f}%",
             _fmt_test(te.max_drawdown_pct if te else 0.0, ".2f") + ("%" if te else "")),
            ("Sharpe Ratio",
             f"{tr.sharpe_ratio:.3f}",
             f"{vr.sharpe_ratio:.3f}",
             _fmt_test(te.sharpe_ratio if te else 0.0, ".3f")),
            ("Sortino Ratio",
             f"{tr.sortino_ratio:.3f}",
             f"{vr.sortino_ratio:.3f}",
             _fmt_test(te.sortino_ratio if te else 0.0, ".3f")),
            ("Expectancy",
             f"${tr.expectancy:+,.2f}",
             f"${vr.expectancy:+,.2f}",
             _fmt_test(te.expectancy if te else 0.0, "+,.2f")),
        ]

        # Remove the duplicate M15 bars row (trade count isn't bar count — drop it)
        rows = rows[1:]

        for label, t_val, v_val, te_val in rows:
            print(
                f"  {label:<24}"
                f"{t_val:>{col_w}}"
                f"{v_val:>{col_w}}"
                f"{te_val:>{col_w}}"
            )

        print(sep)
        if not r.passed_gate:
            print(
                f"  NOTE: Test window not evaluated — val Sharpe "
                f"({r.val_result.sharpe_ratio:.3f}) did not meet "
                f"threshold ({r.min_sharpe:.2f})."
            )
            print(sep)
        print()

    # ------------------------------------------------------------------
    # Multi-symbol run
    # ------------------------------------------------------------------

    def run_all(
        self,
        symbols: List[str],
        data_map: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    ) -> Dict[str, WalkForwardResult]:
        """
        Run walk-forward for every symbol in *symbols*.

        Parameters
        ----------
        symbols:
            Ordered list of tickers to process.
        data_map:
            Mapping of ticker → (m15_df, h1_df).

        Returns
        -------
        Dict of ticker → WalkForwardResult.  Symbols that fail data-size checks
        are skipped with a printed warning and omitted from the output dict.
        """
        results: Dict[str, WalkForwardResult] = {}

        for sym in symbols:
            if sym not in data_map:
                warnings.warn(f"{sym}: no data found in data_map — skipping.", UserWarning)
                continue

            m15_df, h1_df = data_map[sym]
            try:
                result = self.run(sym, m15_df, h1_df)
                results[sym] = result
                self.print_report(result)
            except ValueError as exc:
                print(f"  [SKIP] {sym}: {exc}")
            except Exception as exc:  # noqa: BLE001
                print(f"  [ERROR] {sym}: {exc}")

        # Summary table
        self._print_summary_table(results)

        return results

    def _print_summary_table(self, results: Dict[str, WalkForwardResult]):
        """Print a compact summary table for all symbols."""
        if not results:
            print("No walk-forward results to summarise.")
            return

        sep  = "─" * 74
        print(f"\n{'═' * 74}")
        print("  WALK-FORWARD SUMMARY")
        print(f"{'═' * 74}")
        print(
            f"  {'SYMBOL':<12}"
            f"{'TRAIN SHARPE':>14}"
            f"{'VAL SHARPE':>12}"
            f"{'TEST SHARPE':>12}"
            f"{'VERDICT':>12}"
            f"  {'THRESHOLD':>10}"
        )
        print(sep)

        for sym, r in results.items():
            test_sharpe = (
                f"{r.test_result.sharpe_ratio:.3f}"
                if r.test_result is not None
                else "   —"
            )
            verdict = "PASS" if r.passed_gate else "FAIL"
            print(
                f"  {sym:<12}"
                f"{r.train_result.sharpe_ratio:>14.3f}"
                f"{r.val_result.sharpe_ratio:>12.3f}"
                f"{test_sharpe:>12}"
                f"{verdict:>12}"
                f"  {r.min_sharpe:>10.2f}"
            )

        n_pass = sum(1 for r in results.values() if r.passed_gate)
        print(sep)
        print(
            f"  {len(results)} symbol(s) evaluated — "
            f"{n_pass} PASS, {len(results) - n_pass} FAIL"
        )
        print(f"{'═' * 74}\n")

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def to_csv(
        self,
        results: Dict[str, WalkForwardResult],
        path: str = "logs/walk_forward_results.csv",
    ):
        """
        Save the summary table to a CSV file.

        Parameters
        ----------
        results:
            Dictionary returned by :meth:`run_all`.
        path:
            Destination file path.  Parent directories are created if they do
            not already exist.
        """
        if not results:
            warnings.warn("to_csv: no results to save.", UserWarning)
            return

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        rows = []
        for sym, r in results.items():
            rows.append(
                {
                    "symbol":         sym,
                    "train_start":    r.train_start,
                    "train_end":      r.train_end,
                    "val_start":      r.val_start,
                    "val_end":        r.val_end,
                    "test_start":     r.test_start,
                    "test_end":       r.test_end,
                    "train_trades":   r.train_result.total_trades,
                    "train_win_rate": round(r.train_result.win_rate, 4),
                    "train_pnl":      round(r.train_result.net_pnl, 2),
                    "train_pnl_pct":  round(r.train_result.net_pnl_pct, 4),
                    "train_sharpe":   round(r.train_result.sharpe_ratio, 4),
                    "train_sortino":  round(r.train_result.sortino_ratio, 4),
                    "train_maxdd_pct":round(r.train_result.max_drawdown_pct, 4),
                    "val_trades":     r.val_result.total_trades,
                    "val_win_rate":   round(r.val_result.win_rate, 4),
                    "val_pnl":        round(r.val_result.net_pnl, 2),
                    "val_pnl_pct":    round(r.val_result.net_pnl_pct, 4),
                    "val_sharpe":     round(r.val_result.sharpe_ratio, 4),
                    "val_sortino":    round(r.val_result.sortino_ratio, 4),
                    "val_maxdd_pct":  round(r.val_result.max_drawdown_pct, 4),
                    "test_trades":    r.test_result.total_trades if r.test_result else None,
                    "test_win_rate":  round(r.test_result.win_rate, 4) if r.test_result else None,
                    "test_pnl":       round(r.test_result.net_pnl, 2) if r.test_result else None,
                    "test_pnl_pct":   round(r.test_result.net_pnl_pct, 4) if r.test_result else None,
                    "test_sharpe":    round(r.test_result.sharpe_ratio, 4) if r.test_result else None,
                    "test_sortino":   round(r.test_result.sortino_ratio, 4) if r.test_result else None,
                    "test_maxdd_pct": round(r.test_result.max_drawdown_pct, 4) if r.test_result else None,
                    "passed_gate":    r.passed_gate,
                    "min_sharpe":     r.min_sharpe,
                }
            )

        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        print(f"Walk-forward results saved to: {path}")
