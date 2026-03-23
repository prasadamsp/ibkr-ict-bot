"""
Historical Data Downloader
===========================
Downloads OHLCV bar data from IBKR and saves to CSV.
Use these CSVs for offline backtesting without a live IBKR connection.

Usage:
  cd /path/to/IBKR
  python scripts/download_data.py --symbol XAGUSD --days 90
  python scripts/download_data.py --symbol XAGUSD --days 90 --tfs M15,H1
  python scripts/download_data.py --all --days 60

Output: data/cache/<SYMBOL>_<TF>.csv (overwrites existing)
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ib_insync import util

from config.settings import CONFIG, SYMBOLS
from data.data_handler import DataHandler, _save_to_cache, bars_to_df
from utils.logger import system_log


async def download(
    symbols: list[str],
    timeframes: list[str],
    days: int,
    output_dir: str,
) -> None:
    """Download bars for all symbol × timeframe combinations."""
    data_handler = DataHandler()
    data_handler.cfg.readonly = True

    # Temporarily disable cache so we always re-download fresh data
    data_handler.cache_cfg.enabled = False

    await data_handler.connect()

    total = len(symbols) * len(timeframes)
    done = 0

    for sym in symbols:
        if sym not in SYMBOLS:
            print(f"⚠  Unknown symbol {sym} — skipping")
            continue

        for tf in timeframes:
            done += 1
            print(f"\n[{done}/{total}] Downloading {sym} {tf} ({days} days) ...")

            try:
                await data_handler.load_history(sym, [tf], days=days)
                df = data_handler.get_bars(sym, tf)

                if df is not None and not df.empty:
                    # Save to specified output directory
                    out_path = Path(output_dir) / f"{sym}_{tf}.csv"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(out_path, index_label="datetime")
                    print(
                        f"      ✓  {len(df)} bars saved → {out_path}\n"
                        f"         Range: {df.index[0]} → {df.index[-1]}"
                    )
                else:
                    print(f"      ✗  No data returned — check market hours and data subscriptions")
            except Exception as e:
                print(f"      ✗  Error: {e}")

    data_handler.disconnect()
    print(f"\nDone. Files saved to: {Path(output_dir).resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Download IBKR historical bar data to CSV"
    )
    parser.add_argument(
        "--symbol", default=None,
        help="Symbol to download (e.g. XAGUSD). Omit to use --all."
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Download all configured symbols"
    )
    parser.add_argument(
        "--days", type=int, default=60,
        help="Days of history to download (default: 60)"
    )
    parser.add_argument(
        "--tfs", default="M15,H1",
        help="Comma-separated timeframes (default: M15,H1)"
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Override TWS port (e.g. 7497 for paper)"
    )
    parser.add_argument(
        "--out", default=CONFIG.cache.cache_dir,
        help=f"Output directory (default: {CONFIG.cache.cache_dir})"
    )
    args = parser.parse_args()

    if args.port:
        CONFIG.ibkr.port = args.port

    if args.all:
        symbols = list(SYMBOLS.keys())
    elif args.symbol:
        symbols = [args.symbol.upper()]
    else:
        parser.error("Specify --symbol <SYM> or --all")

    timeframes = [t.strip().upper() for t in args.tfs.split(",") if t.strip()]

    print(f"Symbols:    {', '.join(symbols)}")
    print(f"Timeframes: {', '.join(timeframes)}")
    print(f"Days:       {args.days}")
    print(f"Output:     {args.out}")
    print(f"IBKR port:  {CONFIG.ibkr.port}")

    util.startLoop()
    asyncio.get_event_loop().run_until_complete(
        download(symbols, timeframes, args.days, args.out)
    )


if __name__ == "__main__":
    main()
