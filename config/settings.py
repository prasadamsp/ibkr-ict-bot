"""
Central configuration for the IBKR ICT trading system.
All parameters in one place — change here, applies everywhere.

Environment overrides:  Copy .env.example → .env and set values there.
  IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID
  RISK_PER_TRADE, MAX_DAILY_LOSS, LOG_LEVEL
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

# Load .env file if present (silently ignored if missing)
load_dotenv(Path(__file__).parent.parent / ".env")


# ---------------------------------------------------------------------------
# IBKR Connection
# ---------------------------------------------------------------------------

@dataclass
class IBKRConfig:
    # TWS Paper: 7497 | TWS Live: 7496 | Gateway Paper: 4002 | Gateway Live: 4001
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    timeout: int = 30           # seconds to wait for connection
    readonly: bool = False      # set True for data-only, no orders
    max_retries: int = 5
    retry_delay: float = 5.0    # seconds between reconnect attempts


# ---------------------------------------------------------------------------
# Symbol Definitions
# ---------------------------------------------------------------------------

@dataclass
class SymbolConfig:
    symbol: str
    sec_type: str               # CFD, CASH, STK, FUT, IND
    exchange: str
    currency: str
    what_to_show: str = "MIDPOINT"   # MIDPOINT (CFD/FX), TRADES (stocks/futures)
    lot_size: float = 1.0            # 1 lot = how many units
    tick_size: float = 0.001         # minimum price movement
    pip_value: float = 1.0           # P&L per 1-unit price move per lot
    contract_size: float = 1.0       # units per lot (used in position sizing)
    max_position_size: float = 500.0 # hard cap in lots (per-symbol)
    min_qty: float = 1.0             # minimum order quantity
    qty_step: float = 1.0            # order quantity increment


# ---------------------------------------------------------------------------
# Instrument specs
#
# XAGUSD (Silver CFD) — IBKR CFD, 1 lot = 100 oz
#   $1 move × 100 oz = $100/lot. At $25/oz with $0.25 stop & $250 risk:
#   qty = 250 / (0.25 × 100) = 10 lots = 1 000 oz notional ($25 000).
#
# XAUUSD (Gold CFD) — IBKR CFD, 1 lot = 100 oz (mini)
#   $1 move × 100 oz = $100/lot. At $2 000/oz with $2 stop & $250 risk:
#   qty = 250 / (2 × 100) = 1.25 → 1 lot = 100 oz notional ($200 000 → heavy
#   margin; reduce lot size / raise stop distance in live use).
#
# EURUSD (Forex) — 1 standard lot = 100 000 units
#   $1 move = $100 000/lot. Use fractional lots (0.01 = micro).
#
# ⚠  Verify contract specs against your IBKR account type and region.
#    Use scripts/test_connection.py to qualify contracts before trading.
# ---------------------------------------------------------------------------

SYMBOLS: Dict[str, SymbolConfig] = {
    "XAGUSD": SymbolConfig(
        symbol="XAGUSD",
        sec_type="CFD",
        exchange="SMART",
        currency="USD",
        what_to_show="MIDPOINT",
        lot_size=1.0,
        tick_size=0.001,
        contract_size=100.0,    # 100 oz per lot
        pip_value=100.0,        # $100 per $1 move per lot
        max_position_size=50.0, # 50 lots = 5 000 oz max
        min_qty=1.0,
        qty_step=1.0,
    ),
    "XAUUSD": SymbolConfig(
        symbol="XAUUSD",
        sec_type="CFD",
        exchange="SMART",
        currency="USD",
        what_to_show="MIDPOINT",
        lot_size=1.0,
        tick_size=0.01,
        contract_size=100.0,    # 100 oz per lot (mini gold)
        pip_value=100.0,        # $100 per $1 move per lot
        max_position_size=5.0,  # 5 lots = 500 oz max
        min_qty=1.0,
        qty_step=1.0,
    ),
    "EURUSD": SymbolConfig(
        symbol="EUR",
        sec_type="CASH",
        exchange="IDEALPRO",
        currency="USD",
        what_to_show="MIDPOINT",
        lot_size=1.0,
        tick_size=0.00001,
        contract_size=100_000.0,  # 1 standard lot = 100k units
        pip_value=10.0,            # $10 per pip (0.0001) per lot
        max_position_size=10.0,   # 10 standard lots max
        min_qty=20_000.0,          # IBKR IDEALPRO minimum 20k units
        qty_step=1.0,
    ),
    # NAS100 CFD — IBKR's Nasdaq 100 CFD (symbol: IBUST100)
    # $1 move = $1 per CFD. At ~18000, 1% move = $180/CFD.
    # EU ESMA leverage: 20:1 on major indices.
    # ICT works well on indices: very clean FVGs and OBs.
    "NAS100": SymbolConfig(
        symbol="IBUST100",
        sec_type="CFD",
        exchange="SMART",
        currency="USD",
        what_to_show="MIDPOINT",
        lot_size=1.0,
        tick_size=0.25,            # 0.25 index point minimum move
        contract_size=1.0,         # 1 CFD = $1 per point
        pip_value=1.0,             # $1 per 1-point move per CFD
        max_position_size=20.0,    # 20 CFDs max
        min_qty=1.0,
        qty_step=1.0,
    ),
    # GBPUSD — Forex via IDEALPRO, same spec structure as EURUSD
    "GBPUSD": SymbolConfig(
        symbol="GBP",
        sec_type="CASH",
        exchange="IDEALPRO",
        currency="USD",
        what_to_show="MIDPOINT",
        lot_size=1.0,
        tick_size=0.00001,
        contract_size=100_000.0,
        pip_value=10.0,
        max_position_size=10.0,
        min_qty=20_000.0,
        qty_step=1.0,
    ),
    # GBPJPY — highest-vol major cross, ICT liquidity sweep strategy
    # 1 standard lot = 100 000 GBP; P&L in JPY.
    # pip_value ≈ ¥1 000/lot = ~$7/lot (at USD/JPY 145).
    # Calibrate pip_value in JPY terms; risk manager uses USD equity.
    # ⚠  Verify contract specs for your account region before going live.
    "GBPJPY": SymbolConfig(
        symbol="GBP",
        sec_type="CASH",
        exchange="IDEALPRO",
        currency="JPY",
        what_to_show="MIDPOINT",
        lot_size=1.0,
        tick_size=0.01,            # 1 pip = 0.01 JPY
        contract_size=100_000.0,   # 1 standard lot = 100k GBP
        pip_value=1_000.0,         # ¥1 000 per pip per lot (approx at USDJPY 145)
        max_position_size=5.0,     # 5 lots max (smaller than GBPUSD — higher vol)
        min_qty=20_000.0,          # IBKR IDEALPRO minimum 20k units
        qty_step=1.0,
    ),
    # BTC — Crypto via PAXOS. 1 lot = 1 BTC, $1 move = $1 P&L.
    "BTC": SymbolConfig(
        symbol="BTC",
        sec_type="CRYPTO",
        exchange="PAXOS",
        currency="USD",
        what_to_show="AGGTRADES",
        lot_size=1.0,
        tick_size=0.01,
        contract_size=1.0,    # 1 BTC per lot
        pip_value=1.0,         # $1 per $1 move
        max_position_size=0.5, # max 0.5 BTC
        min_qty=0.0001,
        qty_step=0.0001,
    ),
    # OIL (WTI Crude) — Continuous futures contract on NYMEX
    # ContFuture rolls automatically — no expiry management needed.
    # 1 CL contract = 1,000 barrels. $0.01 tick = $10/contract.
    # At ~$70/bbl, 1 contract notional = ~$70,000. Use fractional qty.
    # Verified: ContFuture('CL','NYMEX') qualifies as 'Light Sweet Crude Oil'
    "OIL": SymbolConfig(
        symbol="CL",
        sec_type="CONTFUT",
        exchange="NYMEX",
        currency="USD",
        what_to_show="TRADES",
        lot_size=1.0,
        tick_size=0.01,
        contract_size=1000.0,  # 1,000 barrels per contract
        pip_value=10.0,         # $10 per $0.01 tick per contract
        max_position_size=2.0,  # max 2 contracts (2,000 barrels)
        min_qty=1.0,
        qty_step=1.0,
    ),
}


# ---------------------------------------------------------------------------
# Regime & Seasonality Configuration
# ---------------------------------------------------------------------------

@dataclass
class RegimeConfig:
    adx_trending_threshold: float = 25.0
    adx_ranging_threshold: float = 20.0
    atr_high_vol_pct: float = 0.015
    atr_low_vol_pct: float = 0.005
    ema_fast: int = 50
    ema_slow: int = 200


@dataclass
class SeasonalityConfig:
    enabled: bool = True
    min_multiplier: float = 0.5
    max_multiplier: float = 1.5


# ---------------------------------------------------------------------------
# Risk Parameters  — hard limits, not suggestions
# ---------------------------------------------------------------------------

@dataclass
class RiskConfig:
    account_currency: str = "EUR"   # EU IBKR account is denominated in EUR

    # Per-trade risk (fraction of equity)
    risk_per_trade: float = 0.005     # 0.5% of account equity
    max_risk_per_trade: float = 0.01  # hard cap 1%

    # Daily limits
    max_daily_loss: float = 0.03      # 3% → daily halt
    max_daily_trades: int = 10

    # Portfolio limits
    max_concurrent_trades: int = 3
    max_correlated_exposure: float = 0.02  # max 2% in correlated pairs

    # Trade quality filters
    min_rr_ratio: float = 2.0         # minimum 1:2 reward/risk
    max_slippage_pct: float = 0.001   # 0.1% — reject if worse

    # Kill switch triggers
    kill_switch_drawdown: float = 0.05       # 5% equity drawdown → halt
    kill_switch_consecutive_losses: int = 5  # 5 losses in a row → halt


# ---------------------------------------------------------------------------
# Strategy Parameters
# ---------------------------------------------------------------------------

@dataclass
class StrategyConfig:
    # Data warmup bars
    m15_lookback: int = 500    # bars to fetch for M15 (historical/backtest)
    h1_lookback: int = 200     # bars to fetch for H1

    # Live feed lookback (days of history pre-loaded on startup)
    live_lookback_days: int = 30   # 30 days = ~2 880 M15 bars

    # Swing detection
    swing_lookback: int = 5    # bars left and right to confirm swing H/L

    # Fair Value Gap
    fvg_min_size_pct: float = 0.0002   # FVG must be >= 0.02% of price
    fvg_max_age_bars: int = 50         # FVG expires after N bars

    # Liquidity sweeps
    equal_hl_tolerance_pct: float = 0.0003  # within 0.03% = "equal"
    equal_hl_lookback: int = 30             # bars to look for equal H/L

    # Premium / Discount zones
    pd_zone_pct: float = 0.10   # top/bottom 10% of range = extreme P/D

    # Sessions (UTC)
    london_open: str = "07:00"
    london_close: str = "11:00"
    ny_open: str = "12:00"
    ny_close: str = "17:00"
    asian_open: str = "23:00"
    asian_close: str = "03:00"

    # Take-profit cap — prevents absurdly wide TPs from ICT swing-high targets
    max_rr_ratio: float = 5.0   # if natural TP gives RR > 5, clamp to 5R

    # Signal confluence
    min_confluence_score: float = 0.55
    score_weights: Dict[str, float] = field(default_factory=lambda: {
        "bos": 0.20,
        "fvg": 0.20,
        "order_block": 0.20,
        "liquidity_sweep": 0.20,
        "session": 0.10,
        "pd_zone": 0.10,
    })

    # Execution
    entry_type: str = "limit"         # "limit" or "market"
    limit_offset_pct: float = 0.0001  # limit entry slightly inside FVG
    order_timeout_bars: int = 4       # cancel unfilled limit after N M15 bars (~1 hour)


# ---------------------------------------------------------------------------
# Data caching
# ---------------------------------------------------------------------------

@dataclass
class CacheConfig:
    enabled: bool = True
    cache_dir: str = "data/cache"
    max_age_hours: float = 25.0  # regenerate if cache older than N hours (25h covers overnight)


# ---------------------------------------------------------------------------
# Logging & Storage
# ---------------------------------------------------------------------------

@dataclass
class LogConfig:
    log_dir: str = "logs"
    trade_log_file: str = "logs/trades.csv"
    signal_log_file: str = "logs/signals.csv"
    system_log_file: str = "logs/system.log"
    log_level: str = "INFO"


# ---------------------------------------------------------------------------
# Master config — import this everywhere
# ---------------------------------------------------------------------------

@dataclass
class Config:
    ibkr: IBKRConfig = field(default_factory=IBKRConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    logging: LogConfig = field(default_factory=LogConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    seasonality: SeasonalityConfig = field(default_factory=SeasonalityConfig)
    symbols: Dict[str, SymbolConfig] = field(default_factory=lambda: SYMBOLS)
    active_symbols: List[str] = field(
        default_factory=lambda: ["XAUUSD", "NAS100", "EURUSD", "GBPUSD", "BTC", "XAGUSD", "OIL", "GBPJPY"]
    )
    paper_trading: bool = True   # NEVER set False without review


CONFIG = Config()

# ---------------------------------------------------------------------------
# Environment overrides  (from .env file or shell environment)
# ---------------------------------------------------------------------------

def _apply_env_overrides(cfg: Config) -> None:
    """Apply environment variable overrides to the config instance."""
    # IBKR connection
    if v := os.getenv("IBKR_HOST"):
        cfg.ibkr.host = v
    if v := os.getenv("IBKR_PORT"):
        cfg.ibkr.port = int(v)
    if v := os.getenv("IBKR_CLIENT_ID"):
        cfg.ibkr.client_id = int(v)

    # Risk
    if v := os.getenv("RISK_PER_TRADE"):
        cfg.risk.risk_per_trade = float(v)
    if v := os.getenv("MAX_DAILY_LOSS"):
        cfg.risk.max_daily_loss = float(v)
    if v := os.getenv("MIN_RR_RATIO"):
        cfg.risk.min_rr_ratio = float(v)

    # Strategy
    if v := os.getenv("MIN_CONFLUENCE_SCORE"):
        cfg.strategy.min_confluence_score = float(v)
    if v := os.getenv("ENTRY_TYPE"):
        cfg.strategy.entry_type = v.lower()  # "market" or "limit"

    # Logging
    if v := os.getenv("LOG_LEVEL"):
        cfg.logging.log_level = v.upper()


_apply_env_overrides(CONFIG)
