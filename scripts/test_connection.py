"""
IBKR Connection Test Script
============================
Run this BEFORE starting live/paper trading to verify:
  1. TWS / IB Gateway is reachable
  2. API connections are enabled in TWS settings
  3. All configured contracts qualify successfully
  4. Account data is readable
  5. A test paper order can be placed (optional, --place-test-order)

Usage:
  cd /path/to/IBKR
  python scripts/test_connection.py
  python scripts/test_connection.py --port 7497
  python scripts/test_connection.py --place-test-order --symbol EURUSD
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Make sure project root is in the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ib_insync import IB, MarketOrder, util

from config.settings import CONFIG, SYMBOLS
from data.data_handler import DataHandler, bars_to_df
from utils.logger import system_log


async def test_connection(port: int, place_test_order: bool, test_symbol: str) -> bool:
    """
    Run all connection diagnostics. Returns True if everything passed.
    """
    print("\n" + "═" * 60)
    print("  IBKR CONNECTION DIAGNOSTIC")
    print("═" * 60)

    # ── 1. Connect ──────────────────────────────────────────────────────────
    print(f"\n[1/5] Connecting to {CONFIG.ibkr.host}:{port} ...")
    ib = IB()
    try:
        await ib.connectAsync(
            CONFIG.ibkr.host,
            port,
            clientId=CONFIG.ibkr.client_id,
            timeout=CONFIG.ibkr.timeout,
        )
        print(f"      ✓  Connected  (server version {ib.client.serverVersion()})")
    except Exception as e:
        print(f"      ✗  FAILED: {e}")
        print(
            "\n  Troubleshooting:\n"
            "  • Is TWS or IB Gateway running?\n"
            "  • In TWS: Edit → Global Configuration → API → Settings\n"
            "    – Enable 'Enable ActiveX and Socket Clients'\n"
            "    – Allow connections from localhost (127.0.0.1)\n"
            "    – Socket port must match (paper=7497, live=7496)\n"
            "    – Uncheck 'Read-Only API' if you want to place orders\n"
        )
        return False

    # ── 2. Account data ────────────────────────────────────────────────────
    print("\n[2/5] Reading account data ...")
    try:
        accounts = ib.managedAccounts()
        print(f"      ✓  Managed accounts: {', '.join(accounts)}")

        equity = 0.0
        equity_currency = CONFIG.risk.account_currency
        for av in ib.accountValues():
            if av.tag == "NetLiquidation" and av.currency == equity_currency:
                equity = float(av.value)
                break
        print(f"      ✓  Net Liquidation: {equity_currency} {equity:,.2f}")

        if equity == 0:
            print("      ⚠  Equity is 0. Is this a fresh paper account?")
    except Exception as e:
        print(f"      ✗  Account read failed: {e}")

    # ── 3. Contract qualification ─────────────────────────────────────────
    print("\n[3/5] Qualifying configured contracts ...")
    from data.data_handler import make_contract

    all_ok = True
    active = {k: v for k, v in SYMBOLS.items() if k in CONFIG.active_symbols}
    for sym_key, sym_cfg in active.items():
        try:
            contract = make_contract(sym_cfg)
            qualified = await ib.qualifyContractsAsync(contract)
            if qualified:
                q = qualified[0]
                print(
                    f"      ✓  {sym_key:<10}  conId={q.conId:<12}  "
                    f"exchange={q.exchange:<10}  currency={q.currency}"
                )
            else:
                print(f"      ✗  {sym_key:<10}  qualification returned empty list")
                all_ok = False
        except Exception as e:
            print(f"      ✗  {sym_key:<10}  ERROR: {e}")
            all_ok = False

    # ── 4. Historical data request ─────────────────────────────────────────
    print(f"\n[4/5] Requesting 3 days of M15 data for {test_symbol} ...")
    try:
        sym_cfg = SYMBOLS[test_symbol]
        contract = make_contract(sym_cfg)
        qualified = await ib.qualifyContractsAsync(contract)
        if qualified:
            bars = await ib.reqHistoricalDataAsync(
                qualified[0],
                endDateTime="",
                durationStr="3 D",
                barSizeSetting="15 mins",
                whatToShow=sym_cfg.what_to_show,
                useRTH=False,
                formatDate=1,
                keepUpToDate=False,
            )
            if bars:
                df = bars_to_df(bars)
                print(
                    f"      ✓  {len(df)} bars received  "
                    f"[{df.index[0]} → {df.index[-1]}]"
                )
                print(
                    f"         Last bar: O={df.iloc[-1].open:.4f}  "
                    f"H={df.iloc[-1].high:.4f}  "
                    f"L={df.iloc[-1].low:.4f}  "
                    f"C={df.iloc[-1].close:.4f}"
                )
            else:
                print(f"      ✗  No bars returned (check market hours / data subscriptions)")
                all_ok = False
        else:
            print(f"      ✗  Could not qualify {test_symbol}")
            all_ok = False
    except Exception as e:
        print(f"      ✗  Historical data request failed: {e}")
        all_ok = False

    # ── 5. Test order (optional) ───────────────────────────────────────────
    if place_test_order:
        print(f"\n[5/5] Placing a 1-unit MARKET order on {test_symbol} (paper) ...")
        print("      This will immediately open and close a tiny position.")
        confirm = input("      Proceed? [y/N]: ").strip().lower()

        if confirm == "y":
            try:
                sym_cfg = SYMBOLS[test_symbol]
                contract = make_contract(sym_cfg)
                qualified = await ib.qualifyContractsAsync(contract)
                if qualified:
                    # Buy minimum quantity
                    order = MarketOrder("BUY", sym_cfg.min_qty)
                    trade = ib.placeOrder(qualified[0], order)
                    await asyncio.sleep(3)
                    print(f"      ✓  Order placed: orderId={trade.order.orderId}  "
                          f"status={trade.orderStatus.status}")

                    # Close immediately
                    close_order = MarketOrder("SELL", sym_cfg.min_qty)
                    close_trade = ib.placeOrder(qualified[0], close_order)
                    await asyncio.sleep(3)
                    print(f"      ✓  Close order placed: orderId={close_trade.order.orderId}  "
                          f"status={close_trade.orderStatus.status}")
                else:
                    print(f"      ✗  Could not qualify contract")
            except Exception as e:
                print(f"      ✗  Order test failed: {e}")
        else:
            print("      Skipped.")
    else:
        print("\n[5/5] Test order: skipped (use --place-test-order to enable)")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    if all_ok:
        print("  ✓  All diagnostics passed. System is ready.")
    else:
        print("  ✗  Some diagnostics failed. Review errors above.")
    print("═" * 60 + "\n")

    ib.disconnect()
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="IBKR connection diagnostic tool"
    )
    parser.add_argument("--port", type=int, default=CONFIG.ibkr.port,
                        help="TWS port (default from config)")
    parser.add_argument("--symbol", default=CONFIG.active_symbols[0],
                        help="Symbol to test data download (default: first active)")
    parser.add_argument("--place-test-order", action="store_true",
                        help="Place a real (paper) test order to verify order routing")
    args = parser.parse_args()

    util.startLoop()
    ok = asyncio.get_event_loop().run_until_complete(
        test_connection(args.port, args.place_test_order, args.symbol)
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
