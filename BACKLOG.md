# IBKR Bot — Backlog

Items debated and decided. Ordered by priority. Pick up after live launch stabilises.

---

## Week 2 (post-live-launch validation)

### Ensemble algo voting
**What:** Require agreement from 3 out of 4 algorithms drawn from different families
(trend-following, mean-reversion, breakout, structural/ICT) before a signal is sent.

**Why:** The research engine picks the *best single algo* per instrument, but a signal
confirmed by multiple independent methods is materially more reliable. Reduces false
positives without raising the confluence threshold further.

**How:**
- Define 4 families: trend (`ema_pullback`, `ma_crossover`), momentum (`macd_momentum`),
  breakout (`donchian_breakout`), mean-reversion (`bb_rsi`, `rsi_extreme`, `zscore_reversion`)
- For each bar, run the top algo + 3 peers from different families
- Gate: ≥ 3/4 must agree on direction before emitting a signal
- Effort: ~1 day (human) / ~30 min (CC)

**Files:** `strategy/instruments/*.py`, new `strategy/ensemble_gate.py`

---

### BTC halving cycle phase awareness
**What:** Track which phase of the 4-year halving cycle BTC is in (accumulation,
markup, distribution, markdown) and adjust confluence threshold or size multiplier
accordingly.

**Why:** BTC has a well-documented cycle driven by supply halving (last halving: Apr 2024).
Trading markup phase long-biased vs distribution phase short-biased improves edge
significantly. The current carry_bias=1 is static; this makes it dynamic.

**How:**
- Hard-code halving dates; derive cycle phase from days-since-halving
- Phase map: 0–12 months post-halving = markup (long bias), 12–24 = late bull/distribution,
  24–36 = bear, 36–48 = accumulation
- Feed into macro_allows_signal() or as a size multiplier in BTCStrategy
- Effort: ~2 hours (human) / ~10 min (CC)

**Files:** `research/macro_filters.py`, `strategy/instruments/btc.py`

---

## Month 2 (after 3–4 weeks of live data)

### HMM regime engine
**What:** Replace the rule-based `RegimeDetector` with a Hidden Markov Model trained on
daily OHLCV data (3+ years). Discovers 4 latent market states (trending-bull,
trending-bear, ranging, volatile/crisis) from price data rather than hard-coded rules.

**Why:** Hamilton (1989) HMM is the gold standard for regime detection. The current
ATR/EMA rule-based detector has fixed thresholds that break in unusual regimes.
HMM adds a FLAT state that prevents trading entirely — the most valuable output.

**How:**
- Use `hmmlearn` library (`GaussianHMM`, 4 components)
- Features: log-returns, ATR%, volume ratio, EMA slope normalised
- Train on 3yr daily data per instrument; retrain weekly alongside the research daemon
- Output: current regime (0–3) + confidence; map to TREND/RANGE/VOLATILE/FLAT
- FLAT state → no signals regardless of confluence
- Effort: ~3 days (human) / ~2 hours (CC)

**Files:** new `strategy/hmm_regime.py`, `strategy/regime.py` (refactor),
`research/run_research.py` (add weekly HMM retrain step)

**Dependency:** Needs 3+ weeks of live daily bar cache before first meaningful train.

---

## Parking lot (discussed, not prioritised)

| Item | Decision | Reason |
|------|----------|--------|
| EWA / online learning | Skip | Insufficient trade frequency (10–15/week/instrument) |
| Full Kelly sizing | Skip | Estimation error too high with small samples |
| Instrument character DB | Defer indefinitely | Adds manual upkeep; carry_bias already covers it |
