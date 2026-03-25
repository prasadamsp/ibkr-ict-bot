"""
Fetch BTC/USD historical OHLCV data from Binance public API.
No API key required. Saves to data/cache/BTC_M15.csv and BTC_H1.csv.

Binance returns up to 1000 bars per request. We paginate backwards
from now to collect a full year (or more) of history.

Usage:
    python scripts/fetch_btc_binance.py
    python scripts/fetch_btc_binance.py --days 730   # 2 years
    python scripts/fetch_btc_binance.py --tfs M15     # one timeframe only
"""

import argparse
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import CONFIG

BINANCE_API = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"

TF_MAP = {
    "M15": "15m",
    "H1":  "1h",
    "H4":  "4h",
    "D1":  "1d",
}

CACHE_DIR = Path(CONFIG.cache.cache_dir)


def fetch_klines(
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
) -> list:
    """Fetch one page of klines from Binance."""
    params = {
        "symbol":    SYMBOL,
        "interval":  interval,
        "startTime": start_ms,
        "endTime":   end_ms,
        "limit":     limit,
    }
    resp = requests.get(BINANCE_API, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def klines_to_df(klines: list) -> pd.DataFrame:
    """Convert Binance kline list to OHLCV DataFrame with UTC datetime index."""
    if not klines:
        return pd.DataFrame()
    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time")
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df


def download(tf_key: str, days: int) -> pd.DataFrame | None:
    """Download *days* of history for *tf_key* (e.g. 'M15')."""
    interval = TF_MAP.get(tf_key)
    if interval is None:
        print(f"  Unknown timeframe: {tf_key}")
        return None

    now_ms  = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    start_ms = int((datetime.now(tz=timezone.utc) - timedelta(days=days)).timestamp() * 1000)

    all_frames = []
    cursor = start_ms
    page = 0

    while cursor < now_ms:
        page += 1
        try:
            klines = fetch_klines(interval, cursor, now_ms, limit=1000)
        except requests.RequestException as exc:
            print(f"  [ERROR] page {page}: {exc}")
            time.sleep(5)
            continue

        if not klines:
            break

        df_page = klines_to_df(klines)
        all_frames.append(df_page)

        # Advance cursor to after the last bar returned
        last_open_ms = klines[-1][0]
        cursor = last_open_ms + 1

        bars_so_far = sum(len(f) for f in all_frames)
        print(f"    page {page:>3}: {len(df_page)} bars   total: {bars_so_far}", end="\r")

        # Binance rate limit: ~1200 requests/min; 100ms sleep is safe
        time.sleep(0.1)

        if len(klines) < 1000:
            break  # last page

    print()  # newline after progress

    if not all_frames:
        return None

    result = pd.concat(all_frames)
    result = result[~result.index.duplicated(keep="first")]
    result = result.sort_index()
    return result


def main():
    parser = argparse.ArgumentParser(description="Fetch BTC OHLCV from Binance")
    parser.add_argument("--days", type=int, default=500,
                        help="Days of history to fetch (default: 500)")
    parser.add_argument("--tfs", default="M15,H1",
                        help="Comma-separated timeframes (default: M15,H1)")
    parser.add_argument("--out", default=str(CACHE_DIR),
                        help=f"Output directory (default: {CACHE_DIR})")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    timeframes = [t.strip().upper() for t in args.tfs.split(",") if t.strip()]

    print(f"\nBinance BTC Downloader")
    print(f"{'─'*40}")
    print(f"Symbol : {SYMBOL}")
    print(f"Days   : {args.days}")
    print(f"TFs    : {', '.join(timeframes)}")
    print(f"Output : {out_dir.resolve()}\n")

    for tf in timeframes:
        print(f"[{tf}] Downloading {args.days} days...")
        df = download(tf, args.days)

        if df is None or df.empty:
            print(f"  [FAIL] No data returned for {tf}\n")
            continue

        out_path = out_dir / f"BTC_{tf}.csv"
        df.to_csv(out_path, index_label="datetime")
        print(
            f"  ✓  {len(df)} bars saved → {out_path}\n"
            f"     Range: {df.index[0]} → {df.index[-1]}\n"
        )


if __name__ == "__main__":
    main()
