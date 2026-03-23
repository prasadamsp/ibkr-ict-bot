"""
Check exact IBKR contract specifications for all configured symbols.

Prints:
  - conId, exchange, currency
  - Minimum order size, lot multiplier
  - Tick size and tick value
  - Current bid/ask (if market is open)

Usage:
  python scripts/check_contract_specs.py
  python scripts/check_contract_specs.py --port 7497
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ib_insync import IB, util
from config.settings import CONFIG, SYMBOLS
from data.data_handler import make_contract


async def check_specs(port: int) -> None:
    ib = IB()
    try:
        await ib.connectAsync(
            CONFIG.ibkr.host, port,
            clientId=CONFIG.ibkr.client_id + 9,  # use a different clientId
            timeout=20,
        )
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    print("\n" + "═" * 70)
    print("  CONTRACT SPECIFICATIONS (from IBKR)")
    print("═" * 70)

    for sym_key, sym_cfg in SYMBOLS.items():
        print(f"\n── {sym_key} ──")
        try:
            contract = make_contract(sym_cfg)
            details_list = await ib.reqContractDetailsAsync(contract)

            if not details_list:
                print("  ✗  No contract details returned")
                continue

            for d in details_list:
                c = d.contract
                print(f"  conId          : {c.conId}")
                print(f"  localSymbol    : {c.localSymbol}")
                print(f"  secType        : {c.secType}")
                print(f"  exchange       : {c.exchange}")
                print(f"  currency       : {c.currency}")
                print(f"  multiplier     : {getattr(d, 'multiplier', 'n/a')}")
                print(f"  minTick        : {d.minTick}")
                print(f"  priceMagnifier : {getattr(d, 'priceMagnifier', 'n/a')}")
                print(f"  longName       : {d.longName}")
                print(f"  tradingHours   : {d.tradingHours[:80] if d.tradingHours else 'n/a'}...")

                # Settings.py comparison
                print(f"\n  settings.py values (check these match):")
                print(f"    contract_size  : {sym_cfg.contract_size}")
                print(f"    pip_value      : {sym_cfg.pip_value}")
                print(f"    tick_size      : {sym_cfg.tick_size}")
                print(f"    min_qty        : {sym_cfg.min_qty}")
                print(f"    what_to_show   : {sym_cfg.what_to_show}")

            # Try to get current market data (snapshot)
            try:
                qualified = await ib.qualifyContractsAsync(contract)
                if qualified:
                    ticker = ib.reqMktData(qualified[0], "", True, False)
                    await asyncio.sleep(2)
                    ib.cancelMktData(qualified[0])
                    bid = ticker.bid if ticker.bid and ticker.bid > 0 else "no data"
                    ask = ticker.ask if ticker.ask and ticker.ask > 0 else "no data"
                    last = ticker.last if ticker.last and ticker.last > 0 else "no data"
                    print(f"\n  Live quote     : bid={bid}  ask={ask}  last={last}")
                    if ticker.bid and ticker.ask and ticker.bid > 0:
                        spread = ticker.ask - ticker.bid
                        print(f"  Spread         : {spread:.5f}  ({spread/ticker.bid*100:.4f}%)")
            except Exception as e:
                print(f"\n  Live quote     : unavailable ({e})")

        except Exception as e:
            print(f"  ✗  Error: {e}")

    print("\n" + "═" * 70)
    print("\n  ► If 'multiplier' is shown (e.g. 100), update contract_size in settings.py")
    print("    to match. Rule: contract_size = multiplier × lot_size")
    print("  ► If minTick differs from tick_size, update tick_size.")
    print("═" * 70 + "\n")

    ib.disconnect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=CONFIG.ibkr.port)
    args = parser.parse_args()

    util.startLoop()
    asyncio.get_event_loop().run_until_complete(check_specs(args.port))


if __name__ == "__main__":
    main()
