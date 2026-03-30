"""
Main entry point — ICT Automated Trading System (IBKR).

Modes:
  paper    — Paper trading via TWS port 7497 (default)
  live     — Live trading via TWS port 7496  (requires 'CONFIRM' prompt)
  backtest — Run backtester from IBKR data (online) or local CSV (offline)
  status   — Connect, show account + positions, then exit

Usage examples:
  python main.py --mode paper
  python main.py --mode paper --symbols XAGUSD,XAUUSD
  python main.py --mode status
  python main.py --mode backtest --symbol XAGUSD --days 60
  python main.py --mode backtest --offline --m15 data/xag_m15.csv --h1 data/xag_h1.csv
  python main.py --mode live    (USE WITH EXTREME CAUTION)
"""

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

from config.settings import CONFIG
from data.data_handler import DataHandler
from execution.execution import ExecutionEngine
from risk.risk_manager import RiskManager
from strategy.router import StrategyRouter
from utils.logger import TradeLogger, system_log

# Phase 2: adaptive system — import lazily to avoid hard sklearn dependency
def _make_router(strategy_cfg, risk_cfg, adaptive: bool, router_type: str = "ict"):
    # Research router — uses validated algo per instrument from best_algos.json
    if router_type == "research":
        from strategy.research_router import ResearchRouter
        return ResearchRouter(strategy_cfg, risk_cfg)
    # ICT adaptive router
    if adaptive:
        try:
            from strategy.adaptive import AdaptiveRouter
            return AdaptiveRouter(strategy_cfg, risk_cfg)
        except ImportError as e:
            system_log.warning(f"AdaptiveRouter unavailable ({e}), using StrategyRouter")
    return StrategyRouter(strategy_cfg, risk_cfg)


# ---------------------------------------------------------------------------
# Live / Paper trading loop  (multi-symbol)
# ---------------------------------------------------------------------------

async def run_live(symbols: list[str], paper: bool = True, adaptive: bool = False, router_type: str = "ict") -> None:
    """
    Multi-symbol live/paper trading loop.

    Lifecycle per symbol:
      • start_live_feed() → pre-loads history + subscribes to live bars
      • M15 bar closes → strategy generates signal → execution engine acts
      • Status logged every minute; kill switch checked continuously
    """
    if not paper and not CONFIG.paper_trading:
        answer = input(
            "\n⚠  WARNING: You are about to run LIVE trading with REAL money.\n"
            "   Type 'CONFIRM' to proceed: "
        )
        if answer.strip() != "CONFIRM":
            print("Aborted.")
            return

    mode_label = "PAPER" if paper else "LIVE"
    # Port already set by CLI --port or IBKR_PORT env var; don't override here

    system_log.info(f"{'='*60}")
    system_log.info(f"Mode: {mode_label}  |  Symbols: {', '.join(symbols)}")
    system_log.info(f"IBKR: {CONFIG.ibkr.host}:{CONFIG.ibkr.port}")
    system_log.info(f"{'='*60}")

    # ── Shared components ────────────────────────────────────────────────────
    data_handler = DataHandler()
    risk_manager = RiskManager()
    # Research router uses separate log files to enable side-by-side comparison
    _sfx = "_research" if router_type == "research" else ""
    _trade_log  = CONFIG.logging.trade_log_file.replace(".csv",  f"{_sfx}.csv")
    _signal_log = CONFIG.logging.signal_log_file.replace(".csv", f"{_sfx}.csv")
    trade_logger = TradeLogger(_trade_log, _signal_log)
    execution = ExecutionEngine(
        data_handler.ib,
        data_handler,
        risk_manager,
        trade_logger,
    )

    # ── Connect ──────────────────────────────────────────────────────────────
    await data_handler.connect()

    equity = data_handler.get_account_value()
    if equity > 0:
        risk_manager.update_account(equity)
        system_log.info(f"Account equity: ${equity:,.2f}")
    else:
        system_log.warning(
            "Could not read account equity. "
            "Risk manager will wait for first equity update."
        )

    # ── Per-symbol setup ─────────────────────────────────────────────────────
    router = _make_router(CONFIG.strategy, CONFIG.risk, adaptive, router_type)
    system_log.info(f"Router: {router_type.upper()}  {'(adaptive)' if adaptive and router_type == 'ict' else ''}")

    for sym in symbols:

        def make_m15_callback(symbol: str):
            def on_m15_bar(s: str, tf: str, df):
                if tf != "M15":
                    return

                # Advance bar counter and cancel stale limit orders
                execution.tick_bar()
                asyncio.ensure_future(execution.cancel_stale_orders())

                # Refresh equity
                eq = data_handler.get_account_value()
                if eq > 0:
                    risk_manager.update_account(eq)

                # Kill switch
                if risk_manager.kill_switch_active:
                    system_log.critical(
                        f"[{symbol}] Kill switch active — closing all trades."
                    )
                    asyncio.ensure_future(execution.close_all_trades("kill_switch"))
                    return

                h1_df = data_handler.get_closed_bars(symbol, "H1")
                m15_df = data_handler.get_closed_bars(symbol, tf)
                d1_df = data_handler.get_closed_bars(symbol, "D1")

                if m15_df is None or h1_df is None or len(m15_df) < 50:
                    return

                current_dt = m15_df.index[-1].to_pydatetime()
                open_positions = list(execution.active_trades.values())

                # Build extra_data for GSR (XAGUSD needs gold price)
                extra_data = {}
                if symbol == "XAGUSD":
                    xau_bars = data_handler.get_closed_bars("XAUUSD", "M15")
                    xag_bars = data_handler.get_closed_bars("XAGUSD", "M15")
                    if xau_bars is not None and xag_bars is not None:
                        extra_data["xau_price"] = float(xau_bars["close"].iloc[-1])
                        extra_data["xag_price"] = float(xag_bars["close"].iloc[-1])

                signal = router.route(
                    symbol, m15_df, h1_df, current_dt, open_positions,
                    extra_data, d1_df=d1_df,
                )
                if signal:
                    system_log.info(
                        f"▶ Signal [{symbol}] {signal.direction.upper()}  "
                        f"entry={signal.entry_price:.5f}  "
                        f"sl={signal.stop_loss:.5f}  tp={signal.take_profit:.5f}  "
                        f"RR={signal.rr_ratio:.1f}  score={signal.confluence_score:.2f}"
                    )
                    asyncio.ensure_future(execution.execute(signal))

            return on_m15_bar

        # start_live_feed pre-loads history AND subscribes live in one call
        await data_handler.start_live_feed(
            sym,
            ["M15", "H1", "D1"],
            make_m15_callback(sym),
        )
        system_log.info(f"[{sym}] Ready.")

    system_log.info("Trading system running. Press Ctrl+C to stop.")

    # ── Keep-alive loop ───────────────────────────────────────────────────────
    try:
        while True:
            await asyncio.sleep(60)
            eq = data_handler.get_account_value()
            if eq > 0:
                risk_manager.update_account(eq)

            status = risk_manager.status_report()
            open_symbols = [t.symbol for t in execution.active_trades.values()]
            system_log.info(
                f"Status  equity=${status['equity']:,.2f}  "
                f"daily_pnl=${status['daily_pnl']:+,.2f} ({status['daily_pnl_pct']:+.2f}%)  "
                f"open={status['open_trades']} {open_symbols}  "
                f"kill_switch={status['kill_switch']}"
            )

    except KeyboardInterrupt:
        system_log.info("Shutdown requested.")
    finally:
        await execution.close_all_trades("shutdown")
        data_handler.disconnect()
        system_log.info("Trading system stopped.")


# ---------------------------------------------------------------------------
# Status check
# ---------------------------------------------------------------------------

async def run_status() -> None:
    """Connect to IBKR and display account + position summary, then exit."""
    system_log.info("Connecting to IBKR for status check...")
    data_handler = DataHandler()

    try:
        await data_handler.connect()
    except ConnectionError as e:
        system_log.error(str(e))
        print("\nCould not connect. Is TWS/Gateway running?")
        return

    ib = data_handler.ib

    print("\n" + "═" * 56)
    print("  IBKR CONNECTION STATUS")
    print("═" * 56)
    print(f"  Connected:   {data_handler.is_connected()}")
    print(f"  Host/Port:   {CONFIG.ibkr.host}:{CONFIG.ibkr.port}")
    print(f"  Client ID:   {CONFIG.ibkr.client_id}")

    # Account summary
    equity = data_handler.get_account_value()
    print(f"\n  Net Liquidation (USD):  ${equity:>14,.2f}")

    # Managed accounts
    try:
        accounts = ib.managedAccounts()
        print(f"  Managed Accounts:  {', '.join(accounts)}")
    except Exception:
        pass

    # Open positions
    positions = data_handler.get_positions()
    print(f"\n{'─'*56}")
    if positions.empty:
        print("  No open positions.")
    else:
        print(f"  {'Symbol':<12} {'SecType':<8} {'Qty':>10} {'AvgCost':>12}")
        print(f"  {'─'*44}")
        for _, row in positions.iterrows():
            print(
                f"  {row['symbol']:<12} {row['sec_type']:<8} "
                f"{row['quantity']:>10.2f} {row['avg_cost']:>12.4f}"
            )

    # Open orders
    open_orders = data_handler.get_open_orders()
    print(f"\n{'─'*56}")
    if not open_orders:
        print("  No open orders.")
    else:
        print(f"  Open Orders ({len(open_orders)}):")
        for o in open_orders:
            print(
                f"    orderId={o.orderId}  {o.action}  {o.totalQuantity}  "
                f"{o.orderType}  lmt={getattr(o, 'lmtPrice', '-')}"
            )

    # Contract qualification test
    print(f"\n{'─'*56}")
    print("  Contract qualification test:")
    for sym in CONFIG.active_symbols:
        try:
            contract = await data_handler._get_contract(sym)
            print(
                f"    ✓  {sym:<10}  conId={contract.conId}  "
                f"exchange={contract.exchange}"
            )
        except Exception as e:
            print(f"    ✗  {sym:<10}  FAILED: {e}")

    print("═" * 56 + "\n")
    data_handler.disconnect()


# ---------------------------------------------------------------------------
# Backtest — IBKR online
# ---------------------------------------------------------------------------

async def run_backtest_online(symbol: str, days: int = 60) -> None:
    """Download data from IBKR and run the backtester."""
    from backtesting.backtester import Backtester

    system_log.info(f"Backtest (online): {symbol}  {days} days")

    data_handler = DataHandler()
    data_handler.cfg.readonly = True

    try:
        await data_handler.connect()
        await data_handler.load_history(symbol, ["M15", "H1"], days=days)
    except Exception as e:
        system_log.error(f"Data load failed: {e}")

    m15_df = data_handler.get_bars(symbol, "M15")
    h1_df = data_handler.get_bars(symbol, "H1")

    if m15_df is None or h1_df is None:
        system_log.error("No data available. Aborting.")
        data_handler.disconnect()
        return

    system_log.info(f"M15={len(m15_df)} bars  H1={len(h1_df)} bars")
    bt = Backtester(50_000, CONFIG.strategy, CONFIG.risk)
    result = bt.run(symbol, m15_df, h1_df)

    Path("logs").mkdir(exist_ok=True)
    bt.trades_to_csv(result, "logs/backtest_trades.csv")
    bt.equity_curve_to_csv(result, "logs/equity_curve.csv")
    data_handler.disconnect()


# ---------------------------------------------------------------------------
# Backtest — offline CSV
# ---------------------------------------------------------------------------

def run_backtest_offline(
    symbol: str,
    m15_path: str,
    h1_path: str,
    initial_equity: float = 50_000,
) -> None:
    """Run backtester from local CSV files — no IBKR connection needed."""
    from backtesting.backtester import Backtester
    from data.csv_loader import load_pair

    system_log.info(f"Backtest (offline): {symbol}  m15={m15_path}  h1={h1_path}")

    try:
        m15_df, h1_df = load_pair(m15_path, h1_path)
    except (FileNotFoundError, ValueError) as e:
        system_log.error(f"CSV load error: {e}")
        sys.exit(1)

    system_log.info(f"M15={len(m15_df)} bars  H1={len(h1_df)} bars")
    bt = Backtester(initial_equity, CONFIG.strategy, CONFIG.risk)
    result = bt.run(symbol, m15_df, h1_df)

    Path("logs").mkdir(exist_ok=True)
    bt.trades_to_csv(result, "logs/backtest_trades.csv")
    bt.equity_curve_to_csv(result, "logs/equity_curve.csv")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ICT Automated Trading System — IBKR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["live", "paper", "backtest", "status"],
        default="paper",
        help="Operating mode (default: paper)",
    )
    parser.add_argument(
        "--symbol",
        default="XAGUSD",
        help="Primary symbol (default: XAGUSD)",
    )
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbols for multi-symbol live/paper mode",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=60,
        help="Days of history for online backtest (default: 60)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override IBKR TWS port (overrides .env and settings.py)",
    )

    og = parser.add_argument_group("Offline backtest")
    og.add_argument(
        "--offline",
        action="store_true",
        help="Use local CSV files instead of IBKR data",
    )
    og.add_argument("--m15", default=None, metavar="PATH", help="M15 CSV path")
    og.add_argument("--h1",  default=None, metavar="PATH", help="H1 CSV path")
    og.add_argument(
        "--equity",
        type=float,
        default=50_000,
        help="Starting equity for backtest (default: 50000)",
    )

    parser.add_argument(
        "--adaptive",
        action="store_true",
        help=(
            "Enable Phase 2 self-adaptive system: ML regime classifier, "
            "Sharpe-weighted position sizing, rolling parameter tuning. "
            "Requires scikit-learn. Falls back to standard router if unavailable."
        ),
    )

    parser.add_argument(
        "--router",
        choices=["ict", "research"],
        default="ict",
        help=(
            "Signal router to use. "
            "'ict' (default): hand-crafted ICT strategies per instrument. "
            "'research': validated research algos from data/research/best_algos.json. "
            "Run both simultaneously on different --client-id values to A/B compare."
        ),
    )

    parser.add_argument(
        "--client-id",
        type=int,
        default=None,
        metavar="N",
        help=(
            "IBKR API client ID (default: 1). Use 2 for the research router instance "
            "so both can connect to the same IB Gateway simultaneously."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # CLI overrides
    if args.port:
        CONFIG.ibkr.port = args.port
    if args.client_id:
        CONFIG.ibkr.client_id = args.client_id

    system_log.info("=" * 60)
    system_log.info(
        f"ICT Trading System  mode={args.mode}  symbol={args.symbol}"
    )
    system_log.info("=" * 60)

    if args.mode == "status":
        from ib_insync import util
        util.startLoop()
        asyncio.get_event_loop().run_until_complete(run_status())

    elif args.mode == "backtest":
        if args.offline:
            if not args.m15 or not args.h1:
                print("Error: --offline requires both --m15 and --h1 paths.")
                sys.exit(1)
            run_backtest_offline(args.symbol, args.m15, args.h1, args.equity)
        else:
            from ib_insync import util
            util.startLoop()
            asyncio.get_event_loop().run_until_complete(
                run_backtest_online(args.symbol, args.days)
            )

    elif args.mode in ("paper", "live"):
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        else:
            symbols = [args.symbol.upper()]

        unknown = [s for s in symbols if s not in CONFIG.symbols]
        if unknown:
            print(f"Error: unknown symbol(s): {unknown}")
            print(f"Configured: {list(CONFIG.symbols.keys())}")
            sys.exit(1)

        from ib_insync import util
        util.startLoop()
        asyncio.get_event_loop().run_until_complete(
            run_live(
                symbols,
                paper=(args.mode == "paper"),
                adaptive=args.adaptive,
                router_type=args.router,
            )
        )

    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
