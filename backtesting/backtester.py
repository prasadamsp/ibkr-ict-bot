"""
Backtesting Framework — event-driven, no lookahead bias.

Design:
───────
- Processes bars strictly left to right.
- Strategy receives only bars up to and including bar[i-1] (closed bars).
- Signals generated at bar[i-1] close are executed at bar[i] open or limit.
- No peeking into future bars.

Metrics computed:
- Total trades, win rate, average win/loss
- Net P&L, P&L %
- Max drawdown (peak to trough on equity curve)
- Sharpe ratio (annualized)
- Expectancy (avg P&L per trade)
- Profit factor (gross profit / gross loss)
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import CONFIG, StrategyConfig, RiskConfig
from strategy.strategy import ICTStrategy, TradeSignal
from utils.logger import system_log


# ---------------------------------------------------------------------------
# Backtest result containers
# ---------------------------------------------------------------------------

@dataclass
class BacktestTrade:
    trade_id: str
    symbol: str
    direction: str
    entry_bar: int
    exit_bar: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    quantity: float
    pnl: float
    pnl_pct: float
    exit_reason: str       # "tp", "sl", "end_of_data"
    confluence_score: float
    hold_bars: int


@dataclass
class BacktestResult:
    symbol: str
    start_date: str
    end_date: str
    initial_equity: float
    final_equity: float
    net_pnl: float
    net_pnl_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    expectancy: float
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Simple backtester
# ---------------------------------------------------------------------------

class Backtester:
    """
    Event-driven backtester for the ICT strategy.

    Usage:
        bt = Backtester(initial_equity=50000)
        result = bt.run(symbol="XAGUSD", m15_df=df_m15, h1_df=df_h1)
        bt.print_report(result)
    """

    def __init__(
        self,
        initial_equity: float = 50_000,
        strategy_cfg: StrategyConfig = None,
        risk_cfg: RiskConfig = None,
    ):
        self.initial_equity = initial_equity
        self.strategy_cfg = strategy_cfg or CONFIG.strategy
        self.risk_cfg = risk_cfg or CONFIG.risk
        self.strategy = ICTStrategy(self.strategy_cfg)

    def run(
        self,
        symbol: str,
        m15_df: pd.DataFrame,
        h1_df: pd.DataFrame,
    ) -> BacktestResult:
        """
        Run backtest on provided OHLCV DataFrames.

        m15_df and h1_df must be indexed by datetime, sorted ascending.
        Both should cover the same date range.
        """
        system_log.info(
            f"Backtest started: {symbol} | "
            f"{m15_df.index[0].date()} → {m15_df.index[-1].date()} | "
            f"{len(m15_df)} M15 bars"
        )

        equity = self.initial_equity
        equity_curve = [equity]
        trades: List[BacktestTrade] = []

        sym_cfg = CONFIG.symbols.get(symbol)
        contract_size = sym_cfg.contract_size if sym_cfg else 1.0

        open_position: Optional[dict] = None  # one position at a time for simplicity

        # Minimum warmup bars needed for indicators
        warmup = max(50, self.strategy_cfg.swing_lookback * 4)

        for i in range(warmup, len(m15_df)):
            # Give strategy only CLOSED bars (up to i-1)
            m15_window = m15_df.iloc[:i].copy()

            # Get H1 bars up to current M15 time
            m15_time = m15_df.index[i - 1]
            h1_window = h1_df[h1_df.index <= m15_time].copy()

            if len(h1_window) < 20:
                continue

            current_bar = m15_df.iloc[i]   # this is the "execution bar"
            open_price = float(current_bar["open"])
            high_price = float(current_bar["high"])
            low_price = float(current_bar["low"])
            close_price = float(current_bar["close"])
            current_time = m15_df.index[i]

            # --- Check if open position hits TP or SL ---
            if open_position is not None:
                tp = open_position["take_profit"]
                sl = open_position["stop_loss"]
                direction = open_position["direction"]

                exit_price = None
                exit_reason = None

                if direction == "bullish":
                    if low_price <= sl:
                        exit_price = sl
                        exit_reason = "sl"
                    elif high_price >= tp:
                        exit_price = tp
                        exit_reason = "tp"
                else:
                    if high_price >= sl:
                        exit_price = sl
                        exit_reason = "sl"
                    elif low_price <= tp:
                        exit_price = tp
                        exit_reason = "tp"

                if exit_price is not None:
                    qty = open_position["quantity"]
                    entry_p = open_position["entry_price"]

                    if direction == "bullish":
                        pnl = (exit_price - entry_p) * qty * contract_size
                    else:
                        pnl = (entry_p - exit_price) * qty * contract_size

                    equity += pnl
                    equity_curve.append(equity)

                    trade = BacktestTrade(
                        trade_id=open_position["id"],
                        symbol=symbol,
                        direction=direction,
                        entry_bar=open_position["entry_bar"],
                        exit_bar=i,
                        entry_time=open_position["entry_time"],
                        exit_time=current_time,
                        entry_price=entry_p,
                        exit_price=exit_price,
                        stop_loss=sl,
                        take_profit=tp,
                        quantity=qty,
                        pnl=pnl,
                        pnl_pct=pnl / self.initial_equity * 100,
                        exit_reason=exit_reason,
                        confluence_score=open_position["confluence_score"],
                        hold_bars=i - open_position["entry_bar"],
                    )
                    trades.append(trade)
                    open_position = None

            # --- Generate signal (if no open position) ---
            if open_position is None:
                signal = self.strategy.on_bar(symbol, "M15", m15_window, h1_window)

                if signal and signal.confluence_score >= self.strategy_cfg.min_confluence_score:
                    # Position sizing
                    risk_amount = equity * self.risk_cfg.risk_per_trade
                    stop_dist = abs(signal.entry_price - signal.stop_loss)
                    if stop_dist > 0:
                        pnl_per_lot = stop_dist * contract_size
                        qty = min(risk_amount / pnl_per_lot, self.risk_cfg.max_position_size)
                        qty = round(qty, 2)
                    else:
                        qty = 0.0

                    if qty > 0 and signal.rr_ratio >= self.risk_cfg.min_rr_ratio:
                        # Determine actual entry (limit or market)
                        if signal.entry_type == "limit":
                            # Check if limit was hit on this bar
                            if signal.direction == "bullish" and low_price <= signal.entry_price:
                                actual_entry = signal.entry_price
                            elif signal.direction == "bearish" and high_price >= signal.entry_price:
                                actual_entry = signal.entry_price
                            else:
                                actual_entry = None  # limit not triggered
                        else:
                            actual_entry = open_price

                        if actual_entry is not None:
                            open_position = {
                                "id": str(uuid.uuid4())[:8],
                                "entry_bar": i,
                                "entry_time": current_time,
                                "entry_price": actual_entry,
                                "stop_loss": signal.stop_loss,
                                "take_profit": signal.take_profit,
                                "direction": signal.direction,
                                "quantity": qty,
                                "confluence_score": signal.confluence_score,
                            }

        # Close any remaining open position at last bar
        if open_position is not None:
            last_close = float(m15_df.iloc[-1]["close"])
            qty = open_position["quantity"]
            entry_p = open_position["entry_price"]
            direction = open_position["direction"]

            if direction == "bullish":
                pnl = (last_close - entry_p) * qty * contract_size
            else:
                pnl = (entry_p - last_close) * qty * contract_size

            equity += pnl
            equity_curve.append(equity)

            trades.append(BacktestTrade(
                trade_id=open_position["id"],
                symbol=symbol,
                direction=direction,
                entry_bar=open_position["entry_bar"],
                exit_bar=len(m15_df) - 1,
                entry_time=open_position["entry_time"],
                exit_time=m15_df.index[-1],
                entry_price=entry_p,
                exit_price=last_close,
                stop_loss=open_position["stop_loss"],
                take_profit=open_position["take_profit"],
                quantity=qty,
                pnl=pnl,
                pnl_pct=pnl / self.initial_equity * 100,
                exit_reason="end_of_data",
                confluence_score=open_position["confluence_score"],
                hold_bars=len(m15_df) - 1 - open_position["entry_bar"],
            ))

        result = self._compute_metrics(symbol, m15_df, trades, equity, equity_curve)
        self.print_report(result)
        return result

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        symbol: str,
        df: pd.DataFrame,
        trades: List[BacktestTrade],
        final_equity: float,
        equity_curve: List[float],
    ) -> BacktestResult:

        net_pnl = final_equity - self.initial_equity
        net_pnl_pct = net_pnl / self.initial_equity * 100

        if not trades:
            return BacktestResult(
                symbol=symbol,
                start_date=str(df.index[0].date()),
                end_date=str(df.index[-1].date()),
                initial_equity=self.initial_equity,
                final_equity=final_equity,
                net_pnl=net_pnl,
                net_pnl_pct=net_pnl_pct,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                max_drawdown=0.0,
                max_drawdown_pct=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                expectancy=0.0,
                trades=[],
                equity_curve=equity_curve,
            )

        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        win_rate = len(wins) / len(pnls) if pnls else 0.0
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        expectancy = float(np.mean(pnls)) if pnls else 0.0

        # Drawdown
        eq_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(eq_arr)
        drawdown = peak - eq_arr
        max_dd = float(np.max(drawdown))
        max_dd_pct = float(np.max(drawdown / peak)) * 100

        # Sharpe (annualized, assumes M15 bars — 96 bars/day)
        returns = np.diff(eq_arr) / eq_arr[:-1]
        bars_per_year = 96 * 252
        if len(returns) > 1 and returns.std() > 0:
            sharpe = float(returns.mean() / returns.std() * np.sqrt(bars_per_year))
        else:
            sharpe = 0.0

        # Sortino (downside deviation only)
        neg_returns = returns[returns < 0]
        if len(neg_returns) > 1 and neg_returns.std() > 0:
            sortino = float(returns.mean() / neg_returns.std() * np.sqrt(bars_per_year))
        else:
            sortino = 0.0

        return BacktestResult(
            symbol=symbol,
            start_date=str(df.index[0].date()),
            end_date=str(df.index[-1].date()),
            initial_equity=self.initial_equity,
            final_equity=final_equity,
            net_pnl=net_pnl,
            net_pnl_pct=net_pnl_pct,
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            expectancy=expectancy,
            trades=trades,
            equity_curve=equity_curve,
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self, r: BacktestResult):
        sep = "═" * 52
        print(f"\n{sep}")
        print(f"  BACKTEST REPORT — {r.symbol}")
        print(sep)
        print(f"  Period:          {r.start_date} → {r.end_date}")
        print(f"  Initial Equity:  ${r.initial_equity:>12,.2f}")
        print(f"  Final Equity:    ${r.final_equity:>12,.2f}")
        print(f"  Net P&L:         ${r.net_pnl:>+12,.2f}  ({r.net_pnl_pct:+.2f}%)")
        print(sep)
        print(f"  Total Trades:    {r.total_trades:>6}")
        print(f"  Winning:         {r.winning_trades:>6}  ({r.win_rate*100:.1f}%)")
        print(f"  Losing:          {r.losing_trades:>6}")
        print(f"  Avg Win:         ${r.avg_win:>+10,.2f}")
        print(f"  Avg Loss:        ${r.avg_loss:>+10,.2f}")
        print(f"  Profit Factor:   {r.profit_factor:>8.2f}")
        print(f"  Expectancy:      ${r.expectancy:>+10,.2f}")
        print(sep)
        print(f"  Max Drawdown:    ${r.max_drawdown:>10,.2f}  ({r.max_drawdown_pct:.2f}%)")
        print(f"  Sharpe Ratio:    {r.sharpe_ratio:>8.3f}")
        print(f"  Sortino Ratio:   {r.sortino_ratio:>8.3f}")
        print(sep)

    def trades_to_csv(self, result: BacktestResult, path: str = "logs/backtest_trades.csv"):
        if not result.trades:
            return
        rows = [vars(t) for t in result.trades]
        pd.DataFrame(rows).to_csv(path, index=False)
        system_log.info(f"Backtest trades saved to {path}")

    def equity_curve_to_csv(self, result: BacktestResult, path: str = "logs/equity_curve.csv"):
        pd.Series(result.equity_curve, name="equity").to_csv(path)
        system_log.info(f"Equity curve saved to {path}")
