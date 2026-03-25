"""
Alternative market data feeds for instruments not available via IBKR.

BinanceWSFeed  — BTC real-time M15/H1 bars via Binance WebSocket kline stream.
                 History loaded from local CSV cache (scripts/fetch_btc_binance.py).
                 No auth required.

TwelveDataFeed — OIL (WTI) real-time M15/H1 bars via Twelve Data REST API.
                 History fetched on startup; new bars polled every bar period.
                 Requires TWELVE_DATA_API_KEY in environment / .env file.

Both feeds implement the same interface as DataHandler's IBKR feed:
  - load history into DataHandler._store(symbol, tf, df)
  - fire bar-close callbacks via DataHandler._fire_callbacks(symbol, tf)
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import aiohttp
import pandas as pd
import websockets

from utils.logger import data_log


# ---------------------------------------------------------------------------
# Timeframe mappings
# ---------------------------------------------------------------------------

TF_TO_BINANCE = {
    "M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m",
    "H1": "1h", "H4": "4h", "D1": "1d",
}

TF_TO_TWELVE = {
    "M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
    "H1": "1h", "H4": "4h", "D1": "1day",
}

TF_SECONDS = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "D1": 86400,
}


# ---------------------------------------------------------------------------
# Binance WebSocket feed  (BTC)
# ---------------------------------------------------------------------------

class BinanceWSFeed:
    """
    Streams real-time OHLCV bars for BTC from Binance public kline WebSocket.

    On startup:
      - Loads historical bars from CSV cache (data/cache/BTC_M15.csv etc.)
      - Opens a WebSocket per timeframe: btcusdt@kline_15m, btcusdt@kline_1h
      - On each closed kline (x=true), stores bar and fires callbacks

    Usage:
        feed = BinanceWSFeed(handler._store, handler._fire_callbacks)
        await feed.start("BTC", ["M15", "H1"])
    """

    WS_BASE    = "wss://stream.binance.com:9443/ws"
    SYMBOL_WS  = "btcusdt"

    def __init__(
        self,
        store_fn:    Callable,
        fire_fn:     Callable,
        cache_dir:   str = "data/cache",
    ):
        self._store     = store_fn
        self._fire      = fire_fn
        self._cache_dir = Path(cache_dir)
        self._tasks:    List[asyncio.Task] = []

    # ------------------------------------------------------------------

    def load_history(self, symbol: str, timeframes: List[str]) -> None:
        """Load bars from local CSV cache into DataHandler storage."""
        for tf in timeframes:
            csv_path = self._cache_dir / f"{symbol}_{tf}.csv"
            if not csv_path.exists():
                data_log.warning(
                    f"[{symbol} {tf}] No Binance cache at {csv_path}. "
                    f"Run: python scripts/fetch_btc_binance.py"
                )
                continue

            try:
                df = pd.read_csv(csv_path, index_col="datetime", parse_dates=True)
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                else:
                    df.index = df.index.tz_convert("UTC")
                df.sort_index(inplace=True)
                self._store(symbol, tf, df)
                data_log.info(
                    f"[{symbol} {tf}] Loaded {len(df)} bars from Binance cache  "
                    f"[{df.index[0]} → {df.index[-1]}]"
                )
            except Exception as e:
                data_log.error(f"[{symbol} {tf}] Cache load failed: {e}")

    async def start(self, symbol: str, timeframes: List[str]) -> None:
        """Load history, then launch one WebSocket listener per timeframe."""
        self.load_history(symbol, timeframes)
        for tf in timeframes:
            binance_tf = TF_TO_BINANCE.get(tf)
            if not binance_tf:
                data_log.warning(f"[{symbol} {tf}] No Binance mapping — skipping.")
                continue
            task = asyncio.ensure_future(self._listen(symbol, tf, binance_tf))
            self._tasks.append(task)
            data_log.info(f"[{symbol} {tf}] Binance WebSocket feed active.")

    async def _listen(self, symbol: str, tf: str, binance_tf: str) -> None:
        """Connect to Binance kline stream; fire callback on each closed bar."""
        url = f"{self.WS_BASE}/{self.SYMBOL_WS}@kline_{binance_tf}"
        while True:
            try:
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    data_log.info(f"[{symbol} {tf}] WS connected → {url}")
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        k = msg.get("k", {})
                        if not k.get("x"):   # x=true → bar closed
                            continue

                        bar = self._kline_to_df(k)
                        self._store(symbol, tf, bar)
                        self._append_cache(symbol, tf, bar)

                        data_log.debug(
                            f"[{symbol} {tf}] Bar closed @ {bar.index[0]}  "
                            f"C={k['c']}"
                        )
                        self._fire(symbol, tf)

            except (websockets.ConnectionClosed, ConnectionError) as e:
                data_log.warning(
                    f"[{symbol} {tf}] WS disconnected ({e}) — reconnecting in 5s"
                )
                await asyncio.sleep(5)
            except Exception as e:
                data_log.error(
                    f"[{symbol} {tf}] WS unexpected error: {e} — reconnecting in 10s"
                )
                await asyncio.sleep(10)

    @staticmethod
    def _kline_to_df(k: dict) -> pd.DataFrame:
        """Convert a Binance kline dict to a single-row OHLCV DataFrame."""
        ts = pd.Timestamp(k["t"], unit="ms", tz="UTC")
        return pd.DataFrame(
            [{
                "open":   float(k["o"]),
                "high":   float(k["h"]),
                "low":    float(k["l"]),
                "close":  float(k["c"]),
                "volume": float(k["v"]),
            }],
            index=pd.DatetimeIndex([ts], name="datetime"),
        )

    def _append_cache(self, symbol: str, tf: str, bar: pd.DataFrame) -> None:
        """Append the new bar to the local CSV cache."""
        try:
            csv_path = self._cache_dir / f"{symbol}_{tf}.csv"
            bar.to_csv(
                csv_path,
                mode="a",
                header=not csv_path.exists(),
                index_label="datetime",
            )
        except Exception as e:
            data_log.debug(f"[{symbol} {tf}] Cache append failed: {e}")


# ---------------------------------------------------------------------------
# Twelve Data polling feed  (OIL / WTI)
# ---------------------------------------------------------------------------

# Twelve Data symbol map
# OIL → USO (United States Oil Fund ETF, NYSE) — free tier, WTI proxy, ~0.95 correlation
# WTI/USD requires Grow/Venture plan; USO is available on Basic (free)
TD_SYMBOLS: Dict[str, str] = {
    "OIL":    "USO",
    "XAUUSD": "XAU/USD",
    "XAGUSD": "XAG/USD",
}

TWELVE_DATA_BASE = "https://api.twelvedata.com"


class TwelveDataFeed:
    """
    Polls Twelve Data REST API for real-time OHLCV bars.

    On startup:
      - Fetches up to `history_days` of historical M15/H1 bars
      - Polls every bar-period (900s for M15, 3600s for H1)
      - Detects new closed bar by comparing timestamps
      - Stores bar and fires callbacks

    Requires env var: TWELVE_DATA_API_KEY

    Usage:
        feed = TwelveDataFeed(api_key, handler._store, handler._fire_callbacks)
        await feed.start("OIL", ["M15", "H1"])
    """

    def __init__(
        self,
        api_key:      str,
        store_fn:     Callable,
        fire_fn:      Callable,
        history_days: int = 30,
    ):
        self._api_key      = api_key
        self._store        = store_fn
        self._fire         = fire_fn
        self._history_days = history_days
        self._last_bar:    Dict[str, pd.Timestamp] = {}   # "SYMBOL_TF" → timestamp
        self._tasks:       List[asyncio.Task] = []

    # ------------------------------------------------------------------

    async def load_history(self, symbol: str, timeframes: List[str]) -> None:
        """Fetch historical bars from Twelve Data on startup."""
        td_sym = TD_SYMBOLS.get(symbol, symbol)
        async with aiohttp.ClientSession() as session:
            for tf in timeframes:
                bars_per_day = 86400 // TF_SECONDS.get(tf, 900)
                outputsize   = min(self._history_days * bars_per_day, 5000)
                url = (
                    f"{TWELVE_DATA_BASE}/time_series"
                    f"?symbol={td_sym}"
                    f"&interval={TF_TO_TWELVE[tf]}"
                    f"&outputsize={outputsize}"
                    f"&timezone=UTC"
                    f"&apikey={self._api_key}"
                )
                try:
                    async with session.get(
                        url, timeout=aiohttp.ClientTimeout(total=20)
                    ) as resp:
                        data = await resp.json()

                    if data.get("status") == "error":
                        data_log.error(
                            f"[{symbol} {tf}] Twelve Data error: {data.get('message')}"
                        )
                        continue

                    values = data.get("values", [])
                    if not values:
                        data_log.warning(
                            f"[{symbol} {tf}] Twelve Data returned no values"
                        )
                        continue

                    df = self._parse_values(values)
                    self._store(symbol, tf, df)
                    key = f"{symbol}_{tf}"
                    self._last_bar[key] = df.index[-1]
                    data_log.info(
                        f"[{symbol} {tf}] Loaded {len(df)} bars from Twelve Data  "
                        f"[{df.index[0]} → {df.index[-1]}]"
                    )

                except Exception as e:
                    data_log.error(
                        f"[{symbol} {tf}] Twelve Data history failed: {e}"
                    )

    async def start(self, symbol: str, timeframes: List[str]) -> None:
        """Load history then start polling tasks."""
        await self.load_history(symbol, timeframes)
        for tf in timeframes:
            interval_secs = TF_SECONDS.get(tf, 900)
            task = asyncio.ensure_future(
                self._poll(symbol, tf, interval_secs)
            )
            self._tasks.append(task)
            data_log.info(
                f"[{symbol} {tf}] Twelve Data polling active "
                f"(every {interval_secs}s)."
            )

    async def _poll(self, symbol: str, tf: str, interval_secs: int) -> None:
        """Poll every interval_secs; detect and emit new closed bars."""
        # Align to next bar boundary before polling
        await self._sleep_to_next_bar(interval_secs)

        td_sym = TD_SYMBOLS.get(symbol, symbol)
        td_tf  = TF_TO_TWELVE[tf]
        key    = f"{symbol}_{tf}"
        url    = (
            f"{TWELVE_DATA_BASE}/time_series"
            f"?symbol={td_sym}"
            f"&interval={td_tf}"
            f"&outputsize=3"
            f"&timezone=UTC"
            f"&apikey={self._api_key}"
        )

        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    async with session.get(
                        url, timeout=aiohttp.ClientTimeout(total=15)
                    ) as resp:
                        data = await resp.json()

                    values = data.get("values", [])
                    if len(values) >= 2:
                        # TD returns newest-first: [0]=forming, [1]=last closed
                        closed_values = values[1:]   # exclude forming bar
                        df_new = self._parse_values(closed_values)

                        last_known = self._last_bar.get(key)
                        new_bars   = (
                            df_new[df_new.index > last_known]
                            if last_known is not None
                            else df_new
                        )

                        if not new_bars.empty:
                            self._store(symbol, tf, new_bars)
                            self._last_bar[key] = new_bars.index[-1]
                            data_log.debug(
                                f"[{symbol} {tf}] New bar @ {new_bars.index[-1]}  "
                                f"C={new_bars.iloc[-1]['close']:.4f}"
                            )
                            self._fire(symbol, tf)

                except Exception as e:
                    data_log.error(f"[{symbol} {tf}] Poll error: {e}")

                await asyncio.sleep(interval_secs)

    @staticmethod
    def _parse_values(values: list) -> pd.DataFrame:
        """Parse Twelve Data 'values' list (newest-first) into UTC DataFrame."""
        rows, index = [], []
        for v in reversed(values):   # reverse → oldest first
            rows.append({
                "open":   float(v["open"]),
                "high":   float(v["high"]),
                "low":    float(v["low"]),
                "close":  float(v["close"]),
                "volume": float(v.get("volume") or 0),
            })
            index.append(pd.Timestamp(v["datetime"], tz="UTC"))

        df = pd.DataFrame(
            rows,
            index=pd.DatetimeIndex(index, name="datetime"),
        )
        df = df[~df.index.duplicated(keep="last")]
        df.sort_index(inplace=True)
        return df

    @staticmethod
    async def _sleep_to_next_bar(interval_secs: int) -> None:
        """Sleep until just after the next bar close boundary (+30s buffer)."""
        now      = datetime.now(timezone.utc).timestamp()
        into_bar = now % interval_secs
        wait     = (interval_secs - into_bar) + 30
        if wait > interval_secs:
            wait -= interval_secs
        data_log.debug(f"Sleeping {wait:.0f}s to next {interval_secs}s bar boundary")
        await asyncio.sleep(wait)
