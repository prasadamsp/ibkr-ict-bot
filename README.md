# IBKR Algorithmic Trading Bot

> **Status: LIVE (Paper Trading)** — Running 24/7 on Hetzner server since March 23, 2026
> Paper account equity: **$1,003,429** | AdaptiveRouter: ON | Top 5 instruments active

A fully autonomous algorithmic trading system that combines **ICT (Inner Circle Trader)** concepts with a **self-adaptive research engine** that finds the best strategy for each instrument automatically — running 24/7 on a cloud server via Interactive Brokers API.

---

## What Makes This Different

Most algo systems pick one strategy and apply it to everything. This bot does the opposite:

1. **Sunday research run** — Automatically tests 10 different algorithms across all 8 instruments using walk-forward validation
2. **Picks the winner per instrument** — Selects the algo with the best out-of-sample (test) Sharpe ratio for each symbol
3. **Adaptive routing at runtime** — ML regime classifier + portfolio optimiser adjusts position sizing based on recent performance
4. **Seasonal filtering** — Only trades in months/hours that have historically shown positive expectancy

---

## Instruments & Current Best Algos

Updated every Sunday by the automated research daemon:

| # | Symbol | Instrument | Best Algo (researched) | Val Sharpe | Test Sharpe |
|---|--------|-----------|------------------------|-----------|------------|
| ✅ | **XAUUSD** | Gold CFD | `donchian_breakout` | 0.49 | **14.40** |
| ✅ | **OIL** | WTI Crude CFD | `ema_pullback` | 1.03 | **12.40** |
| ✅ | **BTC** | Bitcoin | `macd_momentum` | 2.88 | **3.64** |
| ✅ | **GBPUSD** | GBP/USD Forex | `macd_momentum` | 2.28 | **1.75** |
| ✅ | **EURUSD** | Euro/USD Forex | `ema_pullback` | 4.08 | **0.83** |
| 🔬 | NAS100 | Nasdaq 100 CFD | `vwap_reversion` | 8.43 | -0.18 (overfit) |
| 🔬 | XAGUSD | Silver CFD | `rsi_extreme` | 13.44 | 0.00 (overfit) |
| 🔬 | GBPJPY | GBP/JPY Forex | `ma_crossover` | 0.00 | 0.00 (no signal) |

✅ = currently trading (top 5 by out-of-sample test Sharpe) | 🔬 = benched pending next research run

---

## Algorithm Library (10 Strategies)

The research engine tests all of these every Sunday and picks the best one per instrument:

| Algorithm | Type | Description |
|-----------|------|-------------|
| `ema_pullback` | Trend-following | Price retraces to EMA20 while above EMA200 |
| `macd_momentum` | Momentum | MACD histogram zero-cross with trend filter |
| `donchian_breakout` | Breakout | N-bar channel breakout, SL = channel width |
| `bb_rsi` | Mean reversion | Bollinger Band touch + RSI confirmation |
| `rsi_extreme` | Mean reversion | RSI crosses back from oversold/overbought |
| `zscore_reversion` | Mean reversion | Z-score deviation from rolling mean |
| `vwap_reversion` | Mean reversion | Daily VWAP standard-deviation bands |
| `keltner_reversion` | Mean reversion | Keltner Channel boundary reversion |
| `ma_crossover` | Trend-following | EMA golden/death cross |
| `ict_fvg` | ICT | Simplified Fair Value Gap with H1 bias filter |

---

## Architecture

```
main.py
├── DataHandler          ← IBKR ib_insync + Binance WS (BTC) + Twelve Data (OIL)
│   ├── Live M15/H1/D1 bars (keepUpToDate=True)
│   ├── CSV cache seeding (25h max age) → zero cold-start gaps
│   └── M15 bar close → fires strategy pipeline
│
├── AdaptiveRouter       ← Drop-in upgrade over base StrategyRouter
│   ├── MLRegimeClassifier
│   │   ├── 13 features: ADX14/28, ATR%, EMA50/200 slope, RSI14, vol ratio,
│   │   │   volume ratio, day-of-week sin/cos, hour-of-day sin/cos
│   │   ├── RandomForest + StandardScaler pipeline
│   │   ├── Teacher-model bootstrapping from rule-based regime detector
│   │   ├── Confidence threshold 0.55 → falls back to rule-based if uncertain
│   │   └── Weekly auto-retrain on accumulated live data
│   │
│   ├── PortfolioOptimiser
│   │   ├── Rolling 30-trade Sharpe per instrument
│   │   ├── Piecewise linear Sharpe → size multiplier (0× to 2×)
│   │   ├── EWM smoothing (α=0.3) to prevent thrashing
│   │   └── Portfolio risk cap at 4% total
│   │
│   └── ParameterTuner (background thread)
│       ├── Weekly grid search for optimal parameters per instrument
│       ├── SharedParamStore → live param reads without restart
│       └── Persisted to data/adaptive/params.json
│
├── ICT Strategy Engine  ← Per-instrument signal generation (ICT mode)
│   ├── MarketStructure  (swing H/L, BOS, CHoCH)
│   ├── FairValueGap     (3-candle imbalance, ≥0.02% of price)
│   ├── OrderBlock       (last opposing candle before BOS)
│   ├── LiquiditySweep   (equal H/L stop hunts, ±0.03% tolerance)
│   ├── CBDR             (Central Bank Dealers Range + AMD phases)
│   ├── IPDA             (20/40/60-day delivery targets for TP)
│   └── Confluence score ≥ 0.55 required to generate signal
│
├── RiskManager
│   ├── 0.5% risk per trade (hard cap 1%)
│   ├── 3% daily loss halt
│   ├── 5% drawdown kill switch
│   └── Max 3 concurrent trades
│
└── ExecutionEngine      ← IBKR order placement + management
    ├── Limit orders inside FVG midpoint
    ├── Auto-cancel stale orders after 4 bars (~1h)
    └── Paper / Live mode toggle (paper=4002, live=7496)
```

---

## Self-Adaptive Research System

```
research/
├── auto_selector.py        ← Orchestrates full research pipeline
├── walk_forward_grid.py    ← Train/Val/Test split per algo per instrument
├── seasonality_optimizer.py← Per-month and per-hour Sharpe filtering
├── macro_filters.py        ← USD strength, carry bias, risk-on/off, sessions
└── algos/
    ├── base.py             ← BaseAlgo abstract class + AlgoSignal dataclass
    ├── ema_pullback.py
    ├── macd_momentum.py
    ├── donchian.py
    ├── bb_rsi.py
    ├── rsi_extreme.py
    ├── zscore.py
    ├── vwap_revert.py
    ├── keltner.py
    ├── ma_crossover.py
    └── ict_fvg.py
```

### How It Works

Every Sunday at 22:00 UTC, the research daemon:

1. **Loads data** — IBKR CSV cache (up to 33,000 M15 bars), falls back to yfinance
2. **Caps to last 4,000 bars** (~10 weeks) for grid search speed
3. **Splits: TRAIN 60% / VAL 20% / TEST 20%** — strict time ordering
4. **Seasonality optimisation on TRAIN only** — finds best months + hours
5. **Evaluates all 10 algos on VAL** with seasonal filter applied
6. **Gate check**: val Sharpe ≥ 0.3 to proceed to TEST
7. **Picks winner**: highest test Sharpe with both val + test positive
8. **Writes** `data/research/best_algos.json` — live bot reads this on next restart

```bash
# Run manually
python research/run_research.py --once

# Daemon mode (Sunday 22:00 UTC recurring)
python research/run_research.py

# Specific symbols only
python -m research.auto_selector --symbols XAUUSD BTC
```

---

## Risk Controls

| Parameter | Paper | Live Week 1 | Live Week 2+ |
|-----------|-------|-------------|--------------|
| Risk per trade | 0.5% | **0.3%** | 0.5% |
| Max daily loss | 3% | **2%** | 3% |
| Kill switch drawdown | 5% | **3%** | 5% |
| Max concurrent trades | 3 | **2** | 3 |
| AdaptiveRouter | OFF | **ON** | ON |
| Instruments | 8 | **5** | 5→8 |

---

## Live Launch Plan (1-Week Sprint)

| Day | Date | Task |
|-----|------|------|
| ✅ Day 1 | Sun Mar 29 | Research run complete, top 5 confirmed, AdaptiveRouter live |
| Day 2 | Mon Mar 30 | Monitor signals, check confluence scores |
| Day 3 | Tue Mar 31 | Risk param review (tighten to 2% daily max for live) |
| Day 4 | Wed Apr 1 | Technical pre-flight — test live port 7496 |
| Day 5 | Thu Apr 2 | Read-only dry run at live port |
| Day 6 | Fri Apr 3 | Position sizing calc + create live env file |
| **Day 7** | **Sat Apr 4** | **Go/No-Go decision** |
| **Launch** | **Sun Apr 5** | **Live money — 0.3% risk/trade** |

See [LIVE_LAUNCH_PLAN.md](./LIVE_LAUNCH_PLAN.md) for full checklist and Go/No-Go criteria.

---

## Infrastructure

```
Server:       Hetzner CX22 (AMD, 2 vCPU, 4GB RAM, Ubuntu 22.04)
Broker:       Interactive Brokers — IB Gateway 10.30
Runtime:      Python 3.12, ib_insync 0.9.86, pandas, numpy, scikit-learn
Process:      systemd (trading-bot.service + research-daemon.service)
Logging:      logs/system.log, logs/signals.csv, logs/trades.csv
Data:         IBKR bars + Binance WS (BTC) + Twelve Data (OIL) + yfinance (research)
```

### Resilience Features
- **CSV cache seeding** — pre-loads 25h of history before IBKR request
- **Pacing fix** — requests only 2 days from IBKR when cache is fresh (avoids Error 162)
- **Error 1100/1102** — IBKR connectivity blips auto-recover without restart
- **AdaptiveRouter fallback** — gracefully falls back to StrategyRouter if ML unavailable
- **Research daemon failure** — previous `best_algos.json` remains valid until next run

---

## Project Structure

```
IBKR/
├── main.py                        # Entry point — paper/live/status/backtest modes
├── config/settings.py             # All parameters (risk, symbols, data paths)
├── data/
│   ├── data_handler.py            # IBKR connection, bar subscriptions, CSV cache
│   ├── alt_feed.py                # Binance WebSocket (BTC), Twelve Data (OIL)
│   └── csv_loader.py              # Offline backtest CSV loader
├── strategy/
│   ├── strategy.py                # Core ICT engine (FVG, OB, BOS, CBDR, AMD, IPDA)
│   ├── router.py                  # Standard strategy router (one strategy per symbol)
│   ├── adaptive/
│   │   ├── router.py              # AdaptiveRouter — ML + portfolio + param tuning
│   │   ├── ml_regime.py           # MLRegimeClassifier (RandomForest + 13 features)
│   │   ├── portfolio.py           # PortfolioOptimiser (Sharpe → size multiplier)
│   │   └── param_tuner.py         # Background parameter grid search
│   └── instruments/
│       ├── xauusd.py  xagusd.py   # Gold, Silver
│       ├── nas100.py  eurusd.py   # Nasdaq, EUR/USD
│       ├── gbpusd.py  gbpjpy.py   # GBP/USD, GBP/JPY
│       ├── btc.py     oil.py      # Bitcoin, WTI Oil
│       └── base.py                # BaseInstrumentStrategy
├── research/
│   ├── auto_selector.py           # Full research pipeline orchestrator
│   ├── walk_forward_grid.py       # Walk-forward train/val/test grid search
│   ├── seasonality_optimizer.py   # Monthly + hourly Sharpe bucket filtering
│   ├── macro_filters.py           # USD strength, carry, risk-on/off, sessions
│   ├── run_research.py            # Daemon (Sunday 22:00 UTC) + --once manual
│   └── algos/                     # 10 algorithm implementations
├── execution/execution.py         # IBKR order placement + management
├── risk/risk_manager.py           # Position sizing, kill switch, daily halt
├── backtesting/backtester.py      # Walk-forward backtester
├── scripts/
│   ├── setup_research_service.sh  # Install research-daemon.service
│   └── check_contract_specs.py    # Verify IBKR contracts for all instruments
├── LIVE_LAUNCH_PLAN.md            # 1-week paper → live timeline
└── utils/logger.py                # Structured logging
```

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env: set IBKR_HOST, IBKR_PORT, etc.

# Paper trading — top 5 instruments with adaptive routing
python main.py --mode paper --port 4002 --symbols XAUUSD,OIL,BTC,GBPUSD,EURUSD --adaptive

# Status check (no orders placed)
python main.py --mode status --port 4002

# Run research to find best algos
python research/run_research.py --once

# Run specific instruments only
python -m research.auto_selector --symbols XAUUSD BTC --verbose
```

Requires: IB Gateway or TWS running on the configured port.

---

## Build Log

| Date | Milestone |
|------|-----------|
| Mar 22 | Initial bot: XAGUSD + XAUUSD + NAS100, basic ICT strategy |
| Mar 23 | Multi-instrument: 7 strategies, CBDR + AMD + IPDA, portfolio risk cap |
| Mar 23 | Deployed to Hetzner, IB Gateway live, paper trading active |
| Mar 24 | Fixed IBKR pacing (Error 162) + BTC Binance WebSocket feed |
| Mar 25 | Fixed duplicate logs; all 7 symbols loading clean (2600–2800 bars each) |
| Mar 29 | Added GBPJPY (8th instrument); built self-adaptive research system |
| Mar 29 | Built AdaptiveRouter: ML regime + portfolio optimiser + param tuner |
| Mar 29 | First research run complete — top 5 confirmed, AdaptiveRouter live |

---

## ICT Concepts Implemented

| Concept | Implementation |
|---------|---------------|
| **Fair Value Gap (FVG)** | 3-candle imbalance ≥ 0.02% of price, expires after 50 bars |
| **Order Block (OB)** | Last opposing candle before BOS, invalidated on full body trade-through |
| **Liquidity Sweep** | Equal H/L within ±0.03% tolerance — institutional stop hunt zones |
| **Break of Structure (BOS)** | Swing high/low broken — directional confirmation |
| **Change of Character (CHoCH)** | First opposing BOS — potential reversal signal |
| **CBDR** | Asian session range → 4 delivery profiles (classic, reversal, inside, directional) |
| **AMD Cycle** | Accumulation → Manipulation → Distribution phase alignment |
| **IPDA Targets** | 20/40/60-day price delivery ranges for TP placement |
| **Premium/Discount** | Entries in discount zone for longs, premium for shorts |

---

*Built with Python + ib_insync. ICT concepts by Michael J. Huddleston.*
