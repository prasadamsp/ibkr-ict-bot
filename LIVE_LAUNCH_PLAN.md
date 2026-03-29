# Live Money Launch Plan — 1-Week Sprint
**Target:** Switch from paper to live trading on the best 5 instruments.
**Generated:** 2026-03-29

---

## Selection Criteria — Top 5 Instruments

Instruments are scored on:
1. Walk-forward test Sharpe (from research/auto_selector.py)
2. Strategy confidence (how well the algo fits the instrument character)
3. IBKR data quality and execution risk
4. Macro + seasonality window (are we in a good period?)

Current standings (updated after each Sunday research run):
→ See `data/research/best_algos.json` for latest scores.

**CONFIRMED top 5 from research run 2026-03-29 08:46 UTC:**
| # | Instrument | Best Algo           | Val Sharpe | Test Sharpe | Why chosen |
|---|-----------|---------------------|-----------|------------|------------|
| 1 | XAUUSD    | donchian_breakout   | 0.49 | **14.40** | Strongest OOS test |
| 2 | OIL       | ema_pullback        | 1.03 | **12.40** | Strong OOS test |
| 3 | BTC       | macd_momentum       | 2.88 | **3.64**  | Best balanced (test > val) |
| 4 | GBPUSD    | macd_momentum       | 2.28 | **1.75**  | Both positive, consistent |
| 5 | EURUSD    | ema_pullback        | 4.08 | **0.83**  | Both positive, consistent |

**Excluded:** NAS100 (test=-0.18 overfit), XAGUSD (test=0.00 overfit), GBPJPY (0/0 no signal)

(NAS100, XAGUSD, GBPJPY → bench until next research run)

---

## Week Timeline

### Day 1 (Sunday 2026-03-29) ← COMPLETE ✓
- [x] Research daemon running (first results in ~30 min)
- [x] AdaptiveRouter + PortfolioOptimiser built
- [x] Read research results → confirmed top 5 at 08:46 UTC
- [x] Switch active_symbols to top 5 only in paper mode (XAUUSD,OIL,BTC,GBPUSD,EURUSD)
- [x] Confirm `data/research/best_algos.json` has all 5 winners ✓
- [x] AdaptiveRouter started: ml=True, portfolio_opt=True, param_tuner=True

### Day 2 (Monday 2026-03-30) — Paper Validation
- [ ] Monitor paper trading: all 5 symbols generating signals?
- [ ] Check signal quality: confluence scores, RR ratios
- [ ] Verify execution: limit orders filling, SL/TP placing correctly
- [ ] Check `logs/signals.csv` — minimum 3 signals per symbol today
- [ ] No errors in `logs/system.log`
- [ ] Equity curve stable (no unexpected drawdown)

### Day 3 (Tuesday 2026-03-31) — Risk Parameter Review
- [ ] Review risk params in `config/settings.py`:
  - `risk_per_trade = 0.005` (0.5%) — KEEP for live launch
  - `max_daily_loss = 0.03` (3%) — TIGHTEN to 2% for first week live
  - `kill_switch_drawdown = 0.05` — TIGHTEN to 3% for first week live
  - `max_concurrent_trades = 3` — KEEP
- [ ] Confirm IBKR account has sufficient margin for 5 instruments
- [ ] Check `max_position_size` per instrument is appropriate for account size
- [ ] Run `python scripts/check_contract_specs.py` — all 5 qualify

### Day 4 (Wednesday 2026-04-01) — Technical Pre-flight
- [ ] Test IBKR live port (7496): `python main.py --mode status --port 7496`
- [ ] Confirm paper → live port switch works without code changes
- [ ] Verify no EIA event this Wednesday (OIL not in top 5, but check anyway)
- [ ] Check IBKR account type supports CFD + CASH + CRYPTO orders
- [ ] Confirm TWS API permissions: "Enable ActiveX and Socket Clients", "Read-Only API" OFF
- [ ] Verify `CONFIG.paper_trading = True` will require 'CONFIRM' input → change to False requires manual confirmation
- [ ] Run `python scripts/run_backtests.py --symbols XAUUSD NAS100 BTC XAGUSD EURUSD`
  - Expected: all 5 pass walk-forward gate (val Sharpe ≥ 0.8)

### Day 5 (Thursday 2026-04-02) — Dry Run at Live Port
- [ ] Connect to TWS Live (port 7496) in **read-only mode** first
  ```
  python main.py --mode status --port 7496
  ```
- [ ] Verify equity reads correctly from live account
- [ ] Verify positions show correctly
- [ ] Check contract qualification for all 5 live instruments
- [ ] Confirm IB Gateway live is running and stable

### Day 6 (Friday 2026-04-03) — Final Checklist + Position Sizing
- [ ] Calculate max loss for first week:
  - Account equity × 3% daily max × 5 days = 15% max drawdown budget
  - Reduce `risk_per_trade` to 0.003 (0.3%) for first 2 weeks live
  - After 2 weeks if profitable, increase to 0.005 (0.5%)
- [ ] Prepare `config/live_settings_overrides.env`:
  ```
  IBKR_PORT=7496
  RISK_PER_TRADE=0.003
  MAX_DAILY_LOSS=0.02
  ```
- [ ] Update systemd service for live:
  ```
  --mode live --port 7496 --symbols XAUUSD,NAS100,BTC,XAGUSD,EURUSD
  ```
- [ ] Brief stop on Friday before NY close (markets quiet weekend)

### Day 7 (Saturday 2026-04-04) — Go/No-Go Decision
**GO criteria (all must pass):**
- [ ] Paper week: net P&L ≥ 0 (not losing on paper)
- [ ] No kill switch triggers this week
- [ ] Signals CSV shows ≥ 15 signals across the 5 symbols this week
- [ ] System log: 0 critical errors
- [ ] All 5 instruments passed walk-forward backtest
- [ ] IBKR live account has ≥ $10,000 margin buffer above required

**NO-GO criteria (any one blocks):**
- [ ] Kill switch triggered even once
- [ ] Any instrument showing consecutive losses ≥ 3
- [ ] System errors on data feed or order placement
- [ ] Account equity dropped > 1% in paper mode this week

---

## Live Launch (Sunday 2026-04-05 or Monday 2026-04-06)

### Pre-launch commands:
```bash
# On VM
systemctl stop trading-bot

# Edit service to live mode
# ExecStart line becomes:
# --mode live --port 7496 --symbols XAUUSD,NAS100,BTC,XAGUSD,EURUSD

systemctl daemon-reload
systemctl start trading-bot

# Type 'CONFIRM' when prompted
tail -f /opt/trading/IBKR/logs/system.log
```

### First-hour monitoring:
- Watch `logs/system.log` live for first 2 hours
- Confirm first signal executes correctly (order placed, filled, SL/TP set)
- Check account equity in TWS matches system log
- Monitor spread/slippage on first fills

### Week 1 live: reduced sizing
- 0.3% risk per trade (not 0.5%)
- Max 2 concurrent trades (not 3)
- Manual review of any trade > $500 P&L

### Week 2+: if profitable
- Increase to 0.5% risk per trade
- Enable 3 concurrent trades
- Enable AdaptiveRouter: `python main.py --mode live --adaptive`

---

## Risk Controls Summary

| Control | Paper | Live Week 1 | Live Week 2+ |
|---------|-------|-------------|--------------|
| Risk/trade | 0.5% | 0.3% | 0.5% |
| Max daily loss | 3% | 2% | 3% |
| Kill switch drawdown | 5% | 3% | 5% |
| Max concurrent trades | 3 | 2 | 3 |
| AdaptiveRouter | OFF | OFF | ON |
| Instruments | 8 | 5 | 5→8 |

---

## Rollback Plan

If live trading shows unexpected losses in first 3 days:
1. `systemctl stop trading-bot`
2. Close all positions manually in TWS
3. Switch back to paper: `--mode paper`
4. Review `logs/trades.csv` for loss pattern
5. Run research again: `python research/run_research.py --once`
6. Re-evaluate instrument selection

**Hard stop:** If account drops 5% in live mode → automatic kill switch fires.
Manual override: ssh to VM, `systemctl stop trading-bot`, close all in TWS.
