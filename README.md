# IBKR ICT Automated Trading Bot

> **Status: LIVE (Paper Trading)** — Running 24/7 on Hetzner server since March 23, 2026
> Paper account equity: **$1,003,274** | Daily P&L: $0.00 | Open positions: 0

An algorithmic trading system built on **ICT (Inner Circle Trader)** concepts — Fair Value Gaps, Order Blocks, Liquidity Sweeps, Break of Structure, CBDR (Central Bank Dealers Range), AMD cycles, and IPDA logic — executed automatically via Interactive Brokers API.

---

## What It Trades

7 instruments, each with a **tailored strategy** (not one-size-fits-all):

| Symbol | Instrument | Strategy Approach | Data Feed |
|--------|-----------|-------------------|-----------|
| **XAUUSD** | Gold CFD | Pure ICT: FVG + OB + BOS + CBDR | IBKR |
| **XAGUSD** | Silver CFD | ICT + Gold-Silver Ratio (GSR) gate + seasonal filter | IBKR |
| **NAS100** | Nasdaq 100 CFD | ICT on index structure — clean FVG/OB setups | IBKR |
| **EURUSD** | Euro/USD Forex | ICT + London/NY session discipline | IBKR |
| **GBPUSD** | GBP/USD Forex | ICT + London/NY session discipline | IBKR |
| **BTC** | Bitcoin | ICT + Binance WebSocket feed (crypto CFD unavailable on paper) | Binance WS |
| **OIL** | WTI Crude CFD | ICT normal days + EIA Wednesday momentum mode | Twelve Data |

---

## Architecture

```
main.py
├── DataHandler          ← IBKR ib_insync + Binance WS + Twelve Data polling
│   ├── live bars (keepUpToDate=True) for M15, H1, D1
│   ├── CSV cache seeding (25h max age) → zero cold-start gaps
│   └── M15 bar close → callback fires strategy pipeline
│
├── StrategyRouter       ← Routes each symbol to its instrument strategy
│   ├── ICTStrategy.on_bar()
│   │   ├── MarketStructure  (swing H/L, BOS, CHoCH)
│   │   ├── FairValueGap     (FVG detection + freshness)
│   │   ├── OrderBlock       (OB identification + invalidation)
│   │   ├── LiquiditySweep   (equal H/L, stop hunts)
│   │   ├── CBDR             (Central Bank Dealers Range, AMD phases)
│   │   ├── IPDA             (Interbank Price Delivery Algorithm targets)
│   │   ├── RegimeDetector   (ADX trending/ranging + ATR volatility)
│   │   └── Seasonality      (monthly directional bias per instrument)
│   └── per-instrument pre/post filters (GSR, EIA, ATR gates)
│
├── RiskManager          ← Position sizing + kill switch
│   ├── 0.5% risk per trade (hard cap 1%)
│   ├── 3% daily loss halt
│   ├── 5% drawdown kill switch
│   └── max 3 concurrent trades
│
└── ExecutionEngine      ← IBKR order placement + management
    ├── Limit orders (offset inside FVG)
    ├── Auto-cancel stale orders after 4 bars (~1h)
    └── Paper trading mode active
```

---

## ICT Concepts Implemented

### Signal Generation Pipeline
Each M15 bar close runs through:

1. **Market Structure** — Identify swing highs/lows, detect Break of Structure (BOS) and Change of Character (CHoCH). Higher timeframe H1 bias gates lower timeframe entries.

2. **Fair Value Gap (FVG)** — 3-candle imbalance detection. Must be ≥0.02% of price. Expires after 50 bars. Entry limit placed inside the FVG midpoint.

3. **Order Block (OB)** — Last opposing candle before a BOS. Invalidated when price trades through the full OB body.

4. **Liquidity Sweep** — Equal highs/lows within 0.03% tolerance (stop hunt zones). Sweeps above/below = institutional footprint.

5. **CBDR (Central Bank Dealers Range)** — Asian session range defines the day's delivery framework. 4 profiles:
   - `A_classic` — price delivers from CBDR into London/NY
   - `B_reversal` — false break then reversal
   - `C_inside` — consolidation inside range
   - `D_directional` — strong continuation

6. **AMD Cycle** — Accumulation (Asia) → Manipulation (early London) → Distribution (NY). Entries aligned with Distribution phase for highest probability.

7. **IPDA Targets** — 20/40/60-day price delivery ranges for take-profit placement.

8. **Premium/Discount** — Entries only in discount zone for longs, premium for shorts (bottom/top 10% of structure range).

### Confluence Scoring
```
Signal score = weighted sum of active confluences (threshold: 0.55)

  BOS              20%  ← structural confirmation
  FVG              20%  ← imbalance to fill
  Order Block      20%  ← institutional footprint
  Liquidity Sweep  20%  ← stop hunt completed
  Session          10%  ← London/NY timing
  P/D Zone         10%  ← discount/premium positioning
```

---

## Backtest Results

Walk-forward validation on **$50,000 starting equity**, split into 3 non-overlapping windows:
- **Train (60%)** — strategy run on historical data
- **Validation (20%)** — out-of-sample gate check (must pass Sharpe ≥ 0.8 to reveal test)
- **Test (20%)** — untouched final verdict, only revealed if validation passes

Data ranges extended to **2 years** (Mar 2024 → Mar 2026) after downloading full history from Binance (BTC), Yahoo Finance (OIL), and IBKR cache (all others).

### Extended Results (Mar 2024 → Mar 2026)

| Symbol | Window | Period | Trades | Win Rate | Return | Sharpe | Max DD | Gate |
|--------|--------|--------|--------|----------|--------|--------|--------|------|
| **XAGUSD** | Train | Oct 24–Aug 25 | 89 | 28.1% | **+7.9%** | 25.0 | 3.6% | — |
| **XAGUSD** | Validation | Aug–Dec 25 | 27 | 18.5% | +2.0% | 13.8 | 2.9% | — |
| **XAGUSD** | **Test** | **Dec 25–Mar 26** | **26** | **42.3%** | **+19.1%** | **76.9** | **2.5%** | ✅ **PASS** |
| **NAS100** | Train | Oct 24–Aug 25 | 17 | 41.2% | +10.2% | 76.1 | 1.1% | — |
| **NAS100** | Validation | Aug–Dec 25 | 8 | 37.5% | +2.7% | 44.8 | 1.0% | — |
| **NAS100** | **Test** | **Dec 25–Mar 26** | **8** | **50.0%** | **+7.2%** | **101.0** | **0.7%** | ✅ **PASS** |
| **GBPUSD** | Train | Oct 24–Aug 25 | 52 | 19.2% | +2.5% | 7.3 | 6.8% | — |
| **GBPUSD** | Validation | Aug–Dec 25 | 26 | 23.1% | +4.6% | 22.6 | 4.4% | — |
| **GBPUSD** | **Test** | **Dec 25–Mar 26** | **14** | **14.3%** | **-2.5%** | **-32.3** | **4.4%** | ⚠️ PASS gate, negative test |
| XAUUSD | Train | Oct 24–Aug 25 | 67 | 19.4% | +5.2% | 10.8 | 9.6% | — |
| XAUUSD | Validation | Aug–Dec 25 | 24 | 8.3% | -5.9% | -47.0 | 9.6% | ❌ FAIL |
| EURUSD | Train | Oct 24–Aug 25 | 58 | 13.8% | -6.8% | -18.6 | 11.9% | — |
| EURUSD | Validation | Aug–Dec 25 | 21 | 19.1% | -3.2% | -30.5 | 3.7% | ❌ FAIL |
| OIL | Train | Jan–Feb 26 | 9 | 0.0% | -0.5% | -310 | 0.5% | — |
| OIL | Validation | Feb–Mar 26 | 4 | 0.0% | -0.5% | -469 | 0.5% | ❌ FAIL |
| **BTC** | Train | Mar 24–Jun 25 | 164 | 23.2% | +11.3% | 29.5 | 1.4% | — |
| **BTC** | Validation | Jun–Oct 25 | 78 | 19.2% | +0.7% | 6.7 | 2.7% | — |
| **BTC** | **Test** | **Oct 25–Mar 26** | **59** | **27.1%** | **+1.1%** | **16.4** | **1.2%** | ✅ **PASS** |

### Key Findings

**XAGUSD** is the standout performer — 42.3% win rate on unseen test data, +19.1% return, Sharpe 76.9, with only 2.5% max drawdown on a minimum 2:1 RR strategy. Consistent positive P&L across all three windows across the full 17-month period.

**NAS100** delivers clean results — 50% win rate and Sharpe 101 on test, <1% drawdown. Solid across all windows. (Note: uses H1 bars as M15 proxy — M15 NAS100 feed coming.)

**GBPUSD** is mixed — passes the validation gate (Sharpe 22.6) but then goes negative on the test window (-2.5%, Sharpe -32.3). This suggests the strategy over-fit to the Aug–Dec 25 period. Treat as unconfirmed.

**XAUUSD** improved significantly with more data (training went from -9.3% → +5.2%) but validation still fails. The ICT setup frequency for gold may be too low for reliable statistics in the 4-month val window.

**EURUSD & OIL** consistently negative — the current ICT parameter set doesn't suit these instruments. OIL backtest window is still only 2 months (limited Twelve Data cache); extending to 1 year is pending.

> **Methodology:** Walk-forward split 60/20/20 by time. Starting equity $50,000.
> Gate: validation Sharpe ≥ 0.8. ICT strategy is low-frequency by design — small
> trade counts per window are expected. Test result only shown when validation passes.

### Before vs After (Extended Data)

| Symbol | Old Test Return | New Test Return | Old Gate | New Gate |
|--------|----------------|----------------|----------|----------|
| XAGUSD | +13.3% | **+19.1%** | ✅ PASS | ✅ PASS |
| NAS100 | +4.7% | **+7.2%** | ✅ PASS | ✅ PASS |
| GBPUSD | failed val | +4.6% val | ❌ FAIL | ⚠️ gate pass, test fail |
| XAUUSD | -8.2% val | +5.2% train | ❌ FAIL | ❌ FAIL |
| EURUSD | -5.4% val | -3.2% val | ❌ FAIL | ❌ FAIL |
| **BTC** | -0.2% (2 months) | **+1.1% test (1yr)** | ❌ FAIL | ✅ **PASS** |

---

## Live Stats (as of March 25, 2026)

### Completed Trades
```
None yet — system went live March 23, 2026.
ICT strategy is highly selective (min score 0.55, min RR 2:1).
Active trading hours: London 07:00–11:00 UTC, NY 12:00–17:00 UTC.
```

### Paper Signals Generated (all-time)

| Date/Time UTC | Symbol | Direction | Entry | SL | TP | RR | Score | Status |
|---------------|--------|-----------|-------|----|----|----|-------|--------|
| 2026-03-23 20:00 | BTC | BULLISH | 70,396 | 70,032 | 76,640 | 17.2x | 1.00 | placed |
| 2026-03-24 02:45 | BTC | BULLISH | 70,396 | 70,032 | 76,640 | 17.2x | 1.00 | paper |
| 2026-03-24 03:00 | BTC | BULLISH | 70,370 | 70,194 | 76,640 | 35.6x | 1.00 | paper |
| 2026-03-24 03:15 | BTC | BULLISH | 70,057 | 69,651 | 76,640 | 16.2x | 1.00 | paper |
| 2026-03-24 03:30 | BTC | BULLISH | 70,057 | 69,651 | 76,640 | 16.2x | 1.00 | paper |
| 2026-03-24 08:30 | BTC | BULLISH | 70,681 | 70,559 | 71,292 | 5.0x | 0.68 | paper |
| 2026-03-24 08:45 | BTC | BULLISH | 70,681 | 70,559 | 71,292 | 5.0x | 0.83 | paper |
| 2026-03-24 09:30 | BTC | BULLISH | 70,681 | 70,559 | 71,292 | 5.0x | 0.68 | paper |
| 2026-03-24 09:45 | BTC | BULLISH | 70,681 | 70,559 | 71,292 | 5.0x | 0.84 | paper |

> **Note:** BTC signals use `paper_logged` status — BTC trades via Binance WS feed while paper
> account uses IBKR CFD market. XAUUSD/NAS100/XAGUSD/EURUSD/GBPUSD/OIL are live on IBKR paper.

---

## Infrastructure

```
Server:       Hetzner CX22 (AMD, 2 vCPU, 4GB RAM, Ubuntu 22.04)
              IP: 89.167.102.41
Broker:       Interactive Brokers — IB Gateway 10.30 (paper account)
Runtime:      Python 3.11, ib_insync 0.9.86, pandas, numpy
Process:      systemd service (trading-bot.service) — auto-restarts on failure
Logging:      logs/system.log (rotating), logs/signals.csv, logs/trades.csv
Data:         IBKR keepUpToDate bars + Binance WS (BTC) + Twelve Data (OIL)
```

### Startup Resilience
- **CSV cache seeding** — pre-loads up to 25h of history before IBKR request
- **Pacing fix** — requests only 2 days from IBKR when cache is fresh (avoids Error 162 throttle)
- **Error 1100/1102** — IBKR connectivity blips auto-recover without restart

---

## Risk Parameters

```python
risk_per_trade        = 0.5%    # of account equity per trade
max_risk_per_trade    = 1.0%    # hard cap
max_daily_loss        = 3.0%    # daily halt trigger
max_concurrent_trades = 3
kill_switch_drawdown  = 5.0%    # equity drawdown halt
kill_switch_losses    = 5       # consecutive losses halt
min_rr_ratio          = 2.0     # minimum reward:risk
min_confluence_score  = 0.55    # minimum signal quality
```

---

## Build Log

| Date | Milestone |
|------|-----------|
| Mar 22 | Initial bot: XAGUSD + XAUUSD + NAS100, basic ICT strategy |
| Mar 23 | Multi-instrument system: 7 strategies, portfolio intelligence, CBDR + AMD + IPDA added |
| Mar 23 | Deployed to Hetzner server, IB Gateway running, paper trading live |
| Mar 24 | Fixed IBKR pacing (Error 162) with CSV cache seeding; BTC Binance feed added |
| Mar 25 | Fixed duplicate log lines (StreamHandler + systemd stdout redirect conflict) |
| Mar 25 | All 7 symbols loading clean on startup: 2600–2800 M15 bars each, zero timeouts |

---

## Project Structure

```
IBKR/
├── main.py                    # Entry point, live/paper/backtest modes
├── config/settings.py         # All parameters (risk, strategy, symbols)
├── data/
│   ├── data_handler.py        # IBKR connection, bar subscriptions, cache
│   ├── alt_feed.py            # Binance WebSocket (BTC), Twelve Data (OIL)
│   └── csv_loader.py          # Offline backtest CSV loader
├── strategy/
│   ├── strategy.py            # Core ICT engine (FVG, OB, BOS, CBDR, AMD)
│   ├── router.py              # Routes symbols to instrument strategies
│   └── instruments/
│       ├── base.py            # BaseInstrumentStrategy (pre/post filters)
│       ├── xagusd.py          # Silver: GSR gate + seasonal filter
│       ├── xauusd.py          # Gold: pure ICT
│       ├── nas100.py          # Nasdaq: index structure ICT
│       ├── eurusd.py          # EUR/USD: session-gated ICT
│       ├── gbpusd.py          # GBP/USD: session-gated ICT
│       ├── btc.py             # Bitcoin: ICT on crypto
│       └── oil.py             # WTI: ICT + EIA Wednesday event mode
├── execution/execution.py     # IBKR order placement + management
├── risk/risk_manager.py       # Position sizing + kill switch
├── backtesting/backtester.py  # Walk-forward backtester
└── utils/logger.py            # Structured logging (CSV trades, signals)
```

---

## Run It Yourself

```bash
# Paper trading (default)
python main.py --mode paper --symbols XAUUSD,NAS100,EURUSD

# Status check (no trading)
python main.py --mode status

# Offline backtest from CSV
python main.py --mode backtest --offline --m15 data/xag_m15.csv --h1 data/xag_h1.csv

# Online backtest (needs IBKR connection)
python main.py --mode backtest --symbol XAGUSD --days 60
```

Requires: IB Gateway or TWS running, `pip install -r requirements.txt`, copy `.env.example` → `.env`.

---

*Built with Python + ib_insync. ICT concepts by Michael J. Huddleston.*
