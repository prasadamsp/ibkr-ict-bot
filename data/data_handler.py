"""
Data Handler — IBKR connection, historical + live bar management.

Responsibilities:
- Connect/reconnect to TWS or IB Gateway via ib_insync
- Fetch historical OHLCV bars (M15, H1) for any configured symbol
- Optionally cache bars to CSV to avoid redundant downloads
- Subscribe to real-time bar updates (via keepUpToDate)
- Store and maintain rolling DataFrames per symbol per timeframe
- Emit callbacks when a new bar closes (used by strategy engine)

Live feed design
────────────────
A single reqHistoricalData call with keepUpToDate=True handles BOTH
historical pre-load and live updates. No separate "load" + "subscribe"
step is needed. The updateEvent fires:
  hasNewBar=False → forming bar ticked (ignore)
  hasNewBar=True  → previous bar finalized (trigger strategy)
"""

import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd
from ib_insync import IB, BarData, Contract, util

from config.settings import CONFIG, CacheConfig, IBKRConfig, SymbolConfig
from utils.logger import data_log


# ---------------------------------------------------------------------------
# Contract factory
# ---------------------------------------------------------------------------

def make_contract(sym_cfg: SymbolConfig) -> Contract:
    """Build an ib_insync Contract from SymbolConfig."""
    from ib_insync import Forex, Stock

    st = sym_cfg.sec_type

    if st == "CFD":
        c = Contract()
        c.symbol = sym_cfg.symbol
        c.secType = "CFD"
        c.exchange = sym_cfg.exchange
        c.currency = sym_cfg.currency
        return c

    elif st == "CASH":
        # Forex — symbol is the base currency (e.g. "EUR" for EURUSD)
        return Forex(sym_cfg.symbol)

    elif st == "CMDTY":
        c = Contract()
        c.symbol = sym_cfg.symbol
        c.secType = "CMDTY"
        c.exchange = sym_cfg.exchange
        c.currency = sym_cfg.currency
        return c

    elif st == "IND":
        c = Contract()
        c.symbol = sym_cfg.symbol
        c.secType = "IND"
        c.exchange = sym_cfg.exchange
        c.currency = sym_cfg.currency
        return c

    elif st == "STK":
        return Stock(sym_cfg.symbol, sym_cfg.exchange, sym_cfg.currency)

    elif st == "FUT":
        c = Contract()
        c.symbol = sym_cfg.symbol
        c.secType = "FUT"
        c.exchange = sym_cfg.exchange
        c.currency = sym_cfg.currency
        # lastTradeDateOrContractMonth must be set by caller for futures
        return c

    else:
        raise ValueError(f"Unsupported sec_type: {st!r}")


# ---------------------------------------------------------------------------
# Bar DataFrame helpers
# ---------------------------------------------------------------------------

BAR_COLUMNS = ["open", "high", "low", "close", "volume"]

# Our label → IBKR barSizeSetting
TF_MAP = {
    "M1":  "1 min",
    "M5":  "5 mins",
    "M15": "15 mins",
    "M30": "30 mins",
    "H1":  "1 hour",
    "H4":  "4 hours",
    "D1":  "1 day",
}

# Bars per day per timeframe (approximate, used for duration calculation)
BARS_PER_DAY = {
    "M1": 1440, "M5": 288, "M15": 96, "M30": 48,
    "H1": 24, "H4": 6, "D1": 1,
}


def bars_to_df(bars) -> pd.DataFrame:
    """Convert ib_insync BarData list to a clean OHLCV DataFrame (UTC-indexed)."""
    rows = []
    index = []

    for b in bars:
        rows.append({
            "open":   float(b.open),
            "high":   float(b.high),
            "low":    float(b.low),
            "close":  float(b.close),
            "volume": float(b.volume) if b.volume not in (None, -1) else 0.0,
        })
        dt = b.date
        if isinstance(dt, str):
            dt = pd.to_datetime(dt)
        if hasattr(dt, "tzinfo") and dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        index.append(dt)

    if not rows:
        return pd.DataFrame(columns=BAR_COLUMNS)

    df = pd.DataFrame(rows, index=pd.DatetimeIndex(index, name="datetime"))
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)
    return df


def _days_to_duration_str(days: int) -> str:
    """Convert an integer number of days to an IBKR durationStr."""
    if days <= 7:
        return f"{days} D"
    elif days <= 365:
        return f"{days} D"
    else:
        return f"{days // 365 + 1} Y"


# ---------------------------------------------------------------------------
# CSV cache helpers
# ---------------------------------------------------------------------------

def _cache_path(cache_dir: str, symbol: str, tf: str) -> Path:
    return Path(cache_dir) / f"{symbol}_{tf}.csv"


def _load_from_cache(
    cache_dir: str, symbol: str, tf: str, max_age_hours: float
) -> Optional[pd.DataFrame]:
    """Return cached DataFrame if it exists and is fresh enough."""
    p = _cache_path(cache_dir, symbol, tf)
    if not p.exists():
        return None

    age_hours = (datetime.utcnow() - datetime.utcfromtimestamp(p.stat().st_mtime)).total_seconds() / 3600
    if age_hours > max_age_hours:
        data_log.debug(f"Cache expired for {symbol} {tf} (age={age_hours:.1f}h)")
        return None

    try:
        df = pd.read_csv(p, index_col="datetime", parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df.sort_index(inplace=True)
        data_log.info(
            f"[cache] Loaded {len(df)} {symbol} {tf} bars from {p.name}"
        )
        return df
    except Exception as e:
        data_log.warning(f"Cache read failed for {symbol} {tf}: {e}")
        return None


def _save_to_cache(cache_dir: str, symbol: str, tf: str, df: pd.DataFrame) -> None:
    """Persist a bar DataFrame to cache CSV."""
    p = _cache_path(cache_dir, symbol, tf)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index_label="datetime")
    data_log.debug(f"[cache] Saved {len(df)} {symbol} {tf} bars → {p.name}")


# ---------------------------------------------------------------------------
# DataHandler
# ---------------------------------------------------------------------------

class DataHandler:
    """
    Manages all market data subscriptions and storage.

    Backtest usage:
        handler = DataHandler()
        await handler.connect()
        await handler.load_history("XAGUSD", ["M15", "H1"])
        df = handler.get_bars("XAGUSD", "M15")

    Live / paper trading usage:
        handler = DataHandler()
        await handler.connect()
        await handler.start_live_feed("XAGUSD", ["M15", "H1"], on_bar_callback)
        # system runs until KeyboardInterrupt
    """

    def __init__(self, cfg: IBKRConfig = None, cache_cfg: CacheConfig = None):
        self.cfg = cfg or CONFIG.ibkr
        self.cache_cfg = cache_cfg or CONFIG.cache
        self.ib = IB()

        # Storage: symbol → timeframe → DataFrame
        self._bars: Dict[str, Dict[str, pd.DataFrame]] = {}

        # Live subscriptions: symbol → timeframe → BarDataList
        self._subscriptions: Dict[str, Dict[str, object]] = {}

        # Bar-close callbacks: symbol → timeframe → [callable, ...]
        self._callbacks: Dict[str, Dict[str, List[Callable]]] = {}

        # Qualified contracts cache
        self._contracts: Dict[str, Contract] = {}

        # Reconnect task handle
        self._reconnect_task: Optional[asyncio.Task] = None

        self.ib.disconnectedEvent += self._on_disconnect

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def connect(self):
        """Connect to TWS/Gateway with exponential-back-off retry."""
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                data_log.info(
                    f"Connecting to IBKR {self.cfg.host}:{self.cfg.port} "
                    f"clientId={self.cfg.client_id} "
                    f"(attempt {attempt}/{self.cfg.max_retries})"
                )
                await self.ib.connectAsync(
                    self.cfg.host,
                    self.cfg.port,
                    clientId=self.cfg.client_id,
                    timeout=self.cfg.timeout,
                    readonly=self.cfg.readonly,
                )
                data_log.info("Connected to IBKR.")
                return
            except Exception as e:
                data_log.warning(f"Connection attempt {attempt} failed: {e}")
                if attempt < self.cfg.max_retries:
                    await asyncio.sleep(self.cfg.retry_delay * attempt)  # back-off
        raise ConnectionError(
            f"Failed to connect to IBKR after {self.cfg.max_retries} attempts."
        )

    def disconnect(self):
        self.ib.disconnect()
        data_log.info("Disconnected from IBKR.")

    def is_connected(self) -> bool:
        return self.ib.isConnected()

    def _on_disconnect(self):
        data_log.warning("IBKR disconnected. Scheduling reconnect...")
        if self._reconnect_task is None or self._reconnect_task.done():
            self._reconnect_task = asyncio.ensure_future(self._reconnect())

    async def _reconnect(self):
        await asyncio.sleep(self.cfg.retry_delay)
        try:
            await self.connect()
            # Re-subscribe live feeds
            for symbol, tfs in list(self._subscriptions.items()):
                for tf in list(tfs.keys()):
                    data_log.info(f"Re-subscribing {symbol} {tf} after reconnect.")
                    cbs = self._callbacks.get(symbol, {}).get(tf, [])
                    await self.start_live_feed(symbol, [tf], *cbs)
        except Exception as e:
            data_log.error(f"Reconnect failed: {e}")

    # ------------------------------------------------------------------
    # Contract management
    # ------------------------------------------------------------------

    async def _get_contract(self, symbol: str) -> Contract:
        """Qualify and cache the IBKR contract for a symbol."""
        if symbol in self._contracts:
            return self._contracts[symbol]

        sym_cfg = CONFIG.symbols[symbol]
        contract = make_contract(sym_cfg)

        data_log.info(f"Qualifying contract for {symbol} ...")
        qualified = await self.ib.qualifyContractsAsync(contract)
        if not qualified:
            raise ValueError(
                f"IBKR could not qualify contract for {symbol}. "
                f"Check symbol name, sec_type, exchange, and account permissions."
            )

        self._contracts[symbol] = qualified[0]
        q = qualified[0]
        data_log.info(
            f"Contract qualified: {symbol} | conId={q.conId} "
            f"secType={q.secType} exchange={q.exchange} currency={q.currency}"
        )
        return self._contracts[symbol]

    # ------------------------------------------------------------------
    # Historical data (backtest / one-shot)
    # ------------------------------------------------------------------

    async def load_history(
        self,
        symbol: str,
        timeframes: List[str],
        days: Optional[int] = None,
    ) -> None:
        """
        Fetch historical bars for symbol × timeframes.
        Uses cache if enabled and fresh; otherwise downloads from IBKR.

        Args:
            symbol:     Configured symbol key (e.g. "XAGUSD").
            timeframes: List of timeframe keys (e.g. ["M15", "H1"]).
            days:       Override lookback days (default: from StrategyConfig).
        """
        contract = await self._get_contract(symbol)
        sym_cfg = CONFIG.symbols[symbol]

        for tf in timeframes:
            # --- Try cache first ---
            if self.cache_cfg.enabled:
                cached = _load_from_cache(
                    self.cache_cfg.cache_dir, symbol, tf,
                    self.cache_cfg.max_age_hours
                )
                if cached is not None:
                    self._store(symbol, tf, cached)
                    continue

            # --- Download from IBKR ---
            tf_days = days or (
                CONFIG.strategy.m15_lookback // 96
                if tf == "M15"
                else CONFIG.strategy.h1_lookback // 24
            )
            duration_str = _days_to_duration_str(tf_days)
            bar_size = TF_MAP[tf]

            data_log.info(
                f"Downloading {symbol} {tf} history ({duration_str}) ..."
            )

            try:
                bars = await self.ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime="",
                    durationStr=duration_str,
                    barSizeSetting=bar_size,
                    whatToShow=sym_cfg.what_to_show,
                    useRTH=False,
                    formatDate=1,
                    keepUpToDate=False,
                )
            except Exception as e:
                data_log.error(f"History request failed for {symbol} {tf}: {e}")
                continue

            if not bars:
                data_log.warning(f"No historical data returned for {symbol} {tf}")
                continue

            df = bars_to_df(bars)
            self._store(symbol, tf, df)

            if self.cache_cfg.enabled:
                _save_to_cache(self.cache_cfg.cache_dir, symbol, tf, df)

            data_log.info(
                f"Loaded {len(df)} bars for {symbol} {tf}  "
                f"[{df.index[0]} → {df.index[-1]}]"
            )

    # ------------------------------------------------------------------
    # Live feed  (paper / live trading)
    # ------------------------------------------------------------------

    async def start_live_feed(
        self,
        symbol: str,
        timeframes: List[str],
        *callbacks: Callable,
    ) -> None:
        """
        Pre-load history AND subscribe to live updates in one call.

        Uses a single reqHistoricalData with keepUpToDate=True per timeframe.
        The updateEvent fires hasNewBar=True when a new bar period starts,
        meaning the previous bar is now closed and confirmed.

        Args:
            symbol:     Symbol key.
            timeframes: E.g. ["M15", "H1"].
            callbacks:  Zero or more callables(symbol, tf, df) to fire on each
                        M15 bar close. Equivalent to calling on_bar_close()
                        before start_live_feed.
        """
        contract = await self._get_contract(symbol)
        sym_cfg = CONFIG.symbols[symbol]

        # Register any provided callbacks
        for cb in callbacks:
            for tf in timeframes:
                self.on_bar_close(symbol, tf, cb)

        lookback_days = CONFIG.strategy.live_lookback_days
        duration_str = _days_to_duration_str(lookback_days)

        for tf in timeframes:
            if symbol in self._subscriptions and tf in self._subscriptions[symbol]:
                data_log.debug(f"Already subscribed to {symbol} {tf}")
                continue

            bar_size = TF_MAP[tf]
            data_log.info(
                f"Starting live feed: {symbol} {tf}  "
                f"(pre-loading {lookback_days} days of history)"
            )

            # This single call handles both historical pre-load and live updates
            bars_list = self.ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=duration_str,
                barSizeSetting=bar_size,
                whatToShow=sym_cfg.what_to_show,
                useRTH=False,
                formatDate=1,
                keepUpToDate=True,   # ← live updates enabled
            )

            if symbol not in self._subscriptions:
                self._subscriptions[symbol] = {}
            self._subscriptions[symbol][tf] = bars_list

            # Seed initial bars from the historical portion
            if bars_list:
                df = bars_to_df(bars_list)
                self._store(symbol, tf, df)
                data_log.info(
                    f"[{symbol} {tf}] Pre-loaded {len(df)} bars  "
                    f"[{df.index[0]} → {df.index[-1]}]"
                )

            # Wire update handler
            bars_list.updateEvent += self._make_update_handler(symbol, tf)
            data_log.info(f"[{symbol} {tf}] Live subscription active.")

    # ------------------------------------------------------------------
    # Legacy subscribe_live  (kept for backward compatibility)
    # ------------------------------------------------------------------

    async def subscribe_live(self, symbol: str, timeframes: List[str]) -> None:
        """Backward-compatible wrapper. Prefer start_live_feed() for new code."""
        await self.start_live_feed(symbol, timeframes)

    # ------------------------------------------------------------------
    # Bar update handler (fires on each IBKR update tick)
    # ------------------------------------------------------------------

    def _make_update_handler(self, symbol: str, tf: str):
        def handler(bars_list, has_new_bar: bool):
            if not has_new_bar:
                return  # forming bar update — ignore

            # The new bar is bars_list[-1] (forming); bars_list[-2] just closed
            if len(bars_list) < 2:
                return

            new_df = bars_to_df(bars_list)
            self._store(symbol, tf, new_df)

            # Log the just-closed bar
            closed = new_df.iloc[-2]
            data_log.debug(
                f"[{symbol} {tf}] Bar closed  "
                f"O={closed.open:.4f} H={closed.high:.4f} "
                f"L={closed.low:.4f} C={closed.close:.4f}"
            )

            # Fire registered callbacks (strategy runs here)
            self._fire_callbacks(symbol, tf)

        return handler

    def _store(self, symbol: str, tf: str, df: pd.DataFrame) -> None:
        """Merge new bars into storage, deduplicate, keep sorted."""
        if symbol not in self._bars:
            self._bars[symbol] = {}

        existing = self._bars[symbol].get(tf)
        if existing is not None and not existing.empty:
            combined = pd.concat([existing, df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
            self._bars[symbol][tf] = combined
        else:
            self._bars[symbol][tf] = df.copy()

    def _fire_callbacks(self, symbol: str, tf: str) -> None:
        cbs = self._callbacks.get(symbol, {}).get(tf, [])
        df = self.get_bars(symbol, tf)
        if df is None:
            return
        for cb in cbs:
            try:
                cb(symbol, tf, df)
            except Exception as e:
                data_log.error(
                    f"Callback error for {symbol} {tf}: {e}", exc_info=True
                )

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def on_bar_close(self, symbol: str, tf: str, callback: Callable) -> None:
        """
        Register a callable to fire when a bar closes.
        Signature: callback(symbol: str, tf: str, df: pd.DataFrame)
        """
        if symbol not in self._callbacks:
            self._callbacks[symbol] = {}
        if tf not in self._callbacks[symbol]:
            self._callbacks[symbol][tf] = []
        if callback not in self._callbacks[symbol][tf]:
            self._callbacks[symbol][tf].append(callback)

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def get_bars(self, symbol: str, tf: str) -> Optional[pd.DataFrame]:
        """Return current OHLCV DataFrame (includes forming bar). None if not loaded."""
        return self._bars.get(symbol, {}).get(tf)

    def get_closed_bars(self, symbol: str, tf: str) -> Optional[pd.DataFrame]:
        """
        Return only confirmed closed bars — excludes the current forming bar.
        ALL strategy logic must use this — never pass the live (last) bar.
        """
        df = self.get_bars(symbol, tf)
        if df is None or len(df) < 2:
            return None
        return df.iloc[:-1].copy()

    def latest_price(self, symbol: str) -> Optional[float]:
        """Best available price (close of most recent bar)."""
        for tf in ("M1", "M5", "M15"):
            df = self.get_bars(symbol, tf)
            if df is not None and not df.empty:
                return float(df.iloc[-1]["close"])
        return None

    # ------------------------------------------------------------------
    # Account data
    # ------------------------------------------------------------------

    def get_account_value(self) -> float:
        """Return Net Liquidation Value in account currency from IBKR."""
        try:
            currency = CONFIG.risk.account_currency
            for av in self.ib.accountValues():
                if av.tag == "NetLiquidation" and av.currency == currency:
                    return float(av.value)
        except Exception as e:
            data_log.error(f"Failed to get account value: {e}")
        return 0.0

    def get_positions(self) -> pd.DataFrame:
        """Return current open positions as DataFrame."""
        rows = [
            {
                "symbol":   p.contract.symbol,
                "sec_type": p.contract.secType,
                "quantity": p.position,
                "avg_cost": p.avgCost,
            }
            for p in self.ib.positions()
        ]
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["symbol", "sec_type", "quantity", "avg_cost"]
        )

    def get_open_orders(self) -> list:
        """Return list of open orders from IBKR."""
        return self.ib.openOrders()
