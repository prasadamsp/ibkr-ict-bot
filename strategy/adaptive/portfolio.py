"""
portfolio.py — Portfolio-Level Sharpe-Weighted Position Sizer

Rebalances per-instrument position-size multipliers daily based on their
rolling realised Sharpe ratios.

Motivation
----------
Not all 8 instruments contribute equally at all times. A fixed 1.0× size
multiplier ignores:
  - Which instruments are in drawdown vs outperforming
  - Changing regime suitability (Gold mean-reversion working; BTC trending off)
  - Correlation clusters that inflate portfolio volatility

Algorithm
---------
1. After every closed trade, update the instrument's rolling P&L log.
2. Once per day (or on demand), compute rolling Sharpe for each instrument
   over the last N=30 trades (or 20 if fewer available, min 10).
3. Map Sharpe → size multiplier via a piecewise linear transfer function:
     Sharpe ≥ 2.0  → 2.0× (reward outperformance)
     Sharpe = 1.0  → 1.0× (neutral)
     Sharpe = 0.0  → 0.5× (losing money, still trade but small)
     Sharpe < -0.5 → 0.0× (pause — instrument in losing streak)
4. Cap total portfolio exposure: sum of multipliers × base_risk ≤ max_portfolio_risk.
   If cap exceeded, scale all multipliers down proportionally.
5. Multipliers are smoothed via EWM(alpha=0.3) to avoid whipsaw.

Usage
-----
    opt = PortfolioOptimiser(base_risk_per_trade=0.005)
    opt.record_trade("XAUUSD", pnl=250.0)  # call after each close
    mults = opt.get_multipliers()           # call before each signal
    size_mult = mults.get("XAUUSD", 1.0)
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from datetime import datetime, date
from typing import Dict, List, Optional

import numpy as np

_log = logging.getLogger("strategy.adaptive")

_ROLLING_WINDOW   = 30    # trades in rolling Sharpe window
_MIN_WINDOW       = 10    # minimum trades to compute Sharpe (else 1.0×)
_REBALANCE_DAILY  = True  # rebalance once per day; set False for per-trade
_MAX_TOTAL_RISK   = 0.04  # 4% portfolio risk cap (8 instruments × 0.5% base)
_EWM_ALPHA        = 0.3   # smoothing factor for multiplier updates

# Piecewise linear transfer: Sharpe → raw_multiplier
# [(sharpe_breakpoint, multiplier), ...]  — linearly interpolated between
_TRANSFER_CURVE = [
    (-1.0,  0.0),
    (-0.5,  0.0),
    ( 0.0,  0.5),
    ( 1.0,  1.0),
    ( 1.5,  1.5),
    ( 2.0,  2.0),
    ( 3.0,  2.0),   # cap at 2×
]


def _sharpe_to_multiplier(sharpe: float) -> float:
    """Piecewise-linear map from Sharpe ratio to size multiplier."""
    xs = [p[0] for p in _TRANSFER_CURVE]
    ys = [p[1] for p in _TRANSFER_CURVE]
    return float(np.interp(sharpe, xs, ys))


def _rolling_sharpe(returns: List[float]) -> float:
    """Annualised Sharpe from a list of per-trade returns (fraction of equity)."""
    if len(returns) < _MIN_WINDOW:
        return 0.0
    arr  = np.array(returns, dtype=float)
    mean = arr.mean()
    std  = arr.std()
    if std < 1e-10:
        return 1.0 if mean > 0 else 0.0
    # Assume ~4 trades/day × 250 days = 1000 trades/year
    return float(mean / std * np.sqrt(1000))


class PortfolioOptimiser:
    """
    Sharpe-weighted position sizer.

    Parameters
    ----------
    base_risk_per_trade : float
        Fraction of equity risked per trade at 1.0× (matches RiskConfig.risk_per_trade).
    max_portfolio_risk : float
        Maximum combined risk fraction across all open multipliers.
    rolling_window : int
        Number of recent trades used to compute Sharpe.
    """

    def __init__(
        self,
        base_risk_per_trade: float = 0.005,
        max_portfolio_risk:  float = _MAX_TOTAL_RISK,
        rolling_window:      int   = _ROLLING_WINDOW,
    ) -> None:
        self._base_risk    = base_risk_per_trade
        self._max_risk     = max_portfolio_risk
        self._window       = rolling_window

        # Rolling trade P&L per instrument (as fraction of equity at time of trade)
        self._returns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=rolling_window))

        # Current smoothed multipliers
        self._multipliers: Dict[str, float] = {}

        # Last rebalance date
        self._last_rebalance: Optional[date] = None

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def record_trade(
        self,
        symbol: str,
        pnl: float,
        account_equity: float = 50_000.0,
    ) -> None:
        """
        Record a closed trade's P&L.

        Parameters
        ----------
        symbol : str
        pnl : float
            Realised P&L in account currency.
        account_equity : float
            Account equity at time of close (for normalisation).
        """
        ret = pnl / max(account_equity, 1.0)
        self._returns[symbol].append(ret)
        _log.debug("PortfolioOptimiser: %s recorded return %.4f (n=%d)",
                   symbol, ret, len(self._returns[symbol]))

    # ------------------------------------------------------------------
    # Rebalancing
    # ------------------------------------------------------------------

    def rebalance(self, current_dt: Optional[datetime] = None, force: bool = False) -> Dict[str, float]:
        """
        Recompute per-instrument size multipliers.

        Runs at most once per day (unless force=True).

        Returns
        -------
        Dict mapping symbol → size_multiplier (float in [0.0, 2.0]).
        """
        today = (current_dt or datetime.utcnow()).date()
        if not force and self._last_rebalance == today:
            return dict(self._multipliers)

        raw: Dict[str, float] = {}
        for sym, returns in self._returns.items():
            sharpe = _rolling_sharpe(list(returns))
            raw_mult = _sharpe_to_multiplier(sharpe)
            # Apply EWM smoothing against previous value
            prev = self._multipliers.get(sym, 1.0)
            smoothed = _EWM_ALPHA * raw_mult + (1.0 - _EWM_ALPHA) * prev
            raw[sym] = round(smoothed, 3)
            _log.debug(
                "PortfolioOptimiser: %s sharpe=%.2f raw_mult=%.2f smoothed=%.2f",
                sym, sharpe, raw_mult, smoothed,
            )

        # Portfolio risk cap: scale down if sum × base_risk > max_risk
        total_risk = sum(raw.values()) * self._base_risk
        if total_risk > self._max_risk and raw:
            scale = self._max_risk / total_risk
            raw   = {s: round(m * scale, 3) for s, m in raw.items()}
            _log.info(
                "PortfolioOptimiser: risk cap triggered — scaling all by %.2f", scale
            )

        self._multipliers     = raw
        self._last_rebalance  = today

        _log.info("PortfolioOptimiser: rebalanced → %s", raw)
        return dict(raw)

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def get_multipliers(self, current_dt: Optional[datetime] = None) -> Dict[str, float]:
        """
        Return current multipliers, triggering a daily rebalance if due.

        Instruments with no trade history return 1.0× (neutral).
        """
        self.rebalance(current_dt)
        return {sym: self._multipliers.get(sym, 1.0)
                for sym in list(self._returns.keys()) or []}

    def get_multiplier(self, symbol: str, current_dt: Optional[datetime] = None) -> float:
        """Return the size multiplier for a single instrument (1.0 if unknown)."""
        self.rebalance(current_dt)
        return self._multipliers.get(symbol, 1.0)

    def sharpe_summary(self) -> Dict[str, float]:
        """Return current rolling Sharpe per instrument for diagnostics."""
        return {sym: _rolling_sharpe(list(returns))
                for sym, returns in self._returns.items()}

    def reset(self, symbol: Optional[str] = None) -> None:
        """Reset trade history. If symbol is None, reset all instruments."""
        if symbol:
            self._returns[symbol].clear()
            self._multipliers.pop(symbol, None)
        else:
            self._returns.clear()
            self._multipliers.clear()
