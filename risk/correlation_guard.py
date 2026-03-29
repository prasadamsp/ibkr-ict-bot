"""
CorrelationGuard — prevents correlated position concentration.

Exposure groups:
- USD_PAIRS:  instruments that move inversely with the USD (XAUUSD, XAGUSD,
              EURUSD, GBPUSD, OIL).  Max 2 concurrent positions.
- METALS:     XAUUSD + XAGUSD (highly correlated sub-group).  Max 1 position.
- EQUITY:     NAS100.  Max 1 position.
- CRYPTO:     BTC.     Max 1 position.

Special rules:
  Never hold XAUUSD long AND XAGUSD long simultaneously — both are USD-negative
  metals and virtually move in lock-step.

  Concurrent FX gate: EURUSD and GBPUSD move at ~0.85 correlation.  When both
  generate a signal within a 4-hour window, only the higher-confluence signal
  is allowed through.  Use record_signal() after approval and
  should_skip_concurrent_fx() before passing a signal downstream.

Usage:
    guard = CorrelationGuard()
    ok, reason = guard.can_open("XAUUSD", "long", open_positions)
    if not ok:
        print(f"Blocked: {reason}")

    # Concurrent FX gate
    skip, reason = guard.should_skip_concurrent_fx("GBPUSD", "long", 0.72, now)
    if not skip:
        guard.record_signal("GBPUSD", "long", 0.72, now)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class OpenPosition:
    """Represents a single live position held by the portfolio."""
    symbol: str
    direction: str   # "long" or "short"
    size: float
    opened_at: datetime


@dataclass
class _RecentSignal:
    """Tracks a recently generated signal for the concurrent-FX gate."""
    symbol:          str
    direction:       str
    confluence:      float
    generated_at:    datetime


# ---------------------------------------------------------------------------
# CorrelationGuard
# ---------------------------------------------------------------------------

class CorrelationGuard:
    """
    Enforces correlation-based position limits across the portfolio.

    All checks are stateless — caller passes the current list of open
    positions on every call so there is no hidden mutable state to
    sync with the live order book.
    """

    # ----- Exposure group definitions ----------------------------------------

    USD_PAIRS: frozenset = frozenset({"XAUUSD", "XAGUSD", "EURUSD", "GBPUSD", "OIL"})
    """Instruments negatively correlated with USD strength."""

    EQUITY: frozenset = frozenset({"NAS100"})
    """Equity indices."""

    CRYPTO: frozenset = frozenset({"BTC"})
    """Crypto instruments."""

    METALS: frozenset = frozenset({"XAUUSD", "XAGUSD"})
    """Precious metals — highly correlated sub-group within USD_PAIRS."""

    # Highly correlated FX pairs — only the highest-confluence signal wins
    # within a 4-hour window when both fire simultaneously.
    FX_CORRELATED: frozenset = frozenset({"EURUSD", "GBPUSD"})
    """EUR/USD and GBP/USD move together ~0.85 correlation. One at a time."""

    _FX_SIGNAL_WINDOW_H = 4   # hours — recency window for concurrent FX gate

    # ----- Group limits -------------------------------------------------------

    _GROUP_LIMITS: Dict[str, int] = {
        "USD_PAIRS": 2,
        "METALS":    1,
        "EQUITY":    1,
        "CRYPTO":    1,
    }

    def __init__(self) -> None:
        # Recent-signal store for the concurrent FX gate (stateful)
        self._recent_fx_signals: List[_RecentSignal] = []

    # ----- Public API ---------------------------------------------------------

    def can_open(
        self,
        symbol: str,
        direction: str,
        open_positions: List[OpenPosition],
    ) -> Tuple[bool, str]:
        """
        Decide whether opening a new position is allowed under correlation rules.

        Parameters
        ----------
        symbol:
            Instrument ticker, e.g. "XAUUSD".
        direction:
            "long" or "short".
        open_positions:
            All currently open positions in the portfolio.

        Returns
        -------
        (True, "")
            Position may be opened.
        (False, reason)
            Position is blocked; *reason* explains which rule triggered.
        """
        # Rule 1 — Max 2 concurrent USD_PAIRS positions
        if symbol in self.USD_PAIRS:
            usd_count = sum(1 for p in open_positions if p.symbol in self.USD_PAIRS)
            if usd_count >= self._GROUP_LIMITS["USD_PAIRS"]:
                return (
                    False,
                    f"USD_PAIRS limit reached ({usd_count}/{self._GROUP_LIMITS['USD_PAIRS']} open): "
                    f"cannot add {symbol}.",
                )

        # Rule 2 — Max 1 concurrent METALS position
        if symbol in self.METALS:
            metals_count = sum(1 for p in open_positions if p.symbol in self.METALS)
            if metals_count >= self._GROUP_LIMITS["METALS"]:
                existing = [p.symbol for p in open_positions if p.symbol in self.METALS]
                return (
                    False,
                    f"METALS limit reached ({metals_count}/{self._GROUP_LIMITS['METALS']} open, "
                    f"holding {existing}): cannot add {symbol}.",
                )

        # Rule 3 — Max 1 concurrent EQUITY position
        if symbol in self.EQUITY:
            equity_count = sum(1 for p in open_positions if p.symbol in self.EQUITY)
            if equity_count >= self._GROUP_LIMITS["EQUITY"]:
                return (
                    False,
                    f"EQUITY limit reached ({equity_count}/{self._GROUP_LIMITS['EQUITY']} open): "
                    f"cannot add {symbol}.",
                )

        # Rule 4 — Max 1 concurrent CRYPTO position
        if symbol in self.CRYPTO:
            crypto_count = sum(1 for p in open_positions if p.symbol in self.CRYPTO)
            if crypto_count >= self._GROUP_LIMITS["CRYPTO"]:
                return (
                    False,
                    f"CRYPTO limit reached ({crypto_count}/{self._GROUP_LIMITS['CRYPTO']} open): "
                    f"cannot add {symbol}.",
                )

        # Rule 5 — Never hold XAUUSD long AND XAGUSD long simultaneously
        if symbol in self.METALS and direction == "long":
            other_metal = "XAGUSD" if symbol == "XAUUSD" else "XAUUSD"
            conflict = [
                p for p in open_positions
                if p.symbol == other_metal and p.direction == "long"
            ]
            if conflict:
                return (
                    False,
                    f"Dual long metals rule: already long {other_metal}; "
                    f"cannot also go long {symbol} (same USD-negative exposure).",
                )

        return True, ""

    def should_skip_concurrent_fx(
        self,
        symbol: str,
        direction: str,
        confluence: float,
        now: datetime,
    ) -> Tuple[bool, str]:
        """
        Concurrent FX gate: if a correlated FX peer already signaled in the
        last *_FX_SIGNAL_WINDOW_H* hours with EQUAL OR HIGHER confluence,
        block this signal.

        Parameters
        ----------
        symbol:
            Instrument that just generated a signal (e.g. "GBPUSD").
        direction:
            Signal direction ("long" or "short").
        confluence:
            Confluence score of the incoming signal.
        now:
            Timestamp of the current M15 bar close.

        Returns
        -------
        (True, reason)
            Block this signal — a peer with ≥ confluence already fired.
        (False, "")
            Allow this signal.
        """
        if symbol not in self.FX_CORRELATED:
            return False, ""

        self._purge_stale_fx_signals(now)

        for sig in self._recent_fx_signals:
            if sig.symbol == symbol:
                continue   # don't compare an instrument against itself
            # A different FX_CORRELATED peer fired recently with ≥ confluence
            if sig.confluence >= confluence:
                return (
                    True,
                    f"Concurrent FX gate: {sig.symbol} already signaled "
                    f"({sig.direction}, confluence={sig.confluence:.2f}) within "
                    f"{self._FX_SIGNAL_WINDOW_H}h window; {symbol} blocked "
                    f"(confluence={confluence:.2f} ≤ peer).",
                )

        return False, ""

    def record_signal(
        self,
        symbol: str,
        direction: str,
        confluence: float,
        generated_at: datetime,
    ) -> None:
        """
        Record a generated signal so future concurrent-FX gate checks can
        compare against it.

        Call this AFTER a signal has passed all guards and will be acted on,
        so only real approved signals compete for the FX slot.

        Parameters
        ----------
        symbol:
            Instrument that produced the signal.
        direction:
            "long" or "short".
        confluence:
            Confluence score.
        generated_at:
            Timestamp of the bar that generated the signal.
        """
        if symbol not in self.FX_CORRELATED:
            return
        self._purge_stale_fx_signals(generated_at)
        self._recent_fx_signals.append(
            _RecentSignal(
                symbol=symbol,
                direction=direction,
                confluence=confluence,
                generated_at=generated_at,
            )
        )

    # ----- Internal helpers --------------------------------------------------

    def _purge_stale_fx_signals(self, now: datetime) -> None:
        """Remove signals older than _FX_SIGNAL_WINDOW_H from the store."""
        cutoff = now - timedelta(hours=self._FX_SIGNAL_WINDOW_H)
        self._recent_fx_signals = [
            s for s in self._recent_fx_signals
            if s.generated_at >= cutoff
        ]

    def get_group_exposure(
        self,
        symbol: str,
        open_positions: List[OpenPosition],
    ) -> Dict[str, int]:
        """
        Return the count of open positions in each group that *symbol* belongs to.

        Parameters
        ----------
        symbol:
            Instrument ticker used to determine relevant groups.
        open_positions:
            All currently open positions.

        Returns
        -------
        Dict mapping group name → open position count.
        Only groups that contain *symbol* are included; returns an empty dict if
        *symbol* is not in any defined group.
        """
        groups: Dict[str, int] = {}

        if symbol in self.USD_PAIRS:
            groups["USD_PAIRS"] = sum(1 for p in open_positions if p.symbol in self.USD_PAIRS)
        if symbol in self.METALS:
            groups["METALS"] = sum(1 for p in open_positions if p.symbol in self.METALS)
        if symbol in self.EQUITY:
            groups["EQUITY"] = sum(1 for p in open_positions if p.symbol in self.EQUITY)
        if symbol in self.CRYPTO:
            groups["CRYPTO"] = sum(1 for p in open_positions if p.symbol in self.CRYPTO)

        return groups

    def summary(self, open_positions: List[OpenPosition]) -> str:
        """
        Return a human-readable summary of current group exposure.

        Parameters
        ----------
        open_positions:
            All currently open positions.

        Returns
        -------
        Multi-line string suitable for logging or console output.
        """
        usd_positions  = [p for p in open_positions if p.symbol in self.USD_PAIRS]
        metals_positions = [p for p in open_positions if p.symbol in self.METALS]
        equity_positions = [p for p in open_positions if p.symbol in self.EQUITY]
        crypto_positions = [p for p in open_positions if p.symbol in self.CRYPTO]

        def _fmt_group(name: str, limit: int, positions: List[OpenPosition]) -> str:
            count = len(positions)
            detail = (
                ", ".join(f"{p.symbol}({p.direction})" for p in positions)
                if positions else "none"
            )
            bar = "█" * count + "░" * (limit - count)
            return f"  {name:<12} [{bar}] {count}/{limit}  — {detail}"

        lines = [
            "CorrelationGuard — Portfolio Exposure",
            "─" * 50,
            _fmt_group("USD_PAIRS",  self._GROUP_LIMITS["USD_PAIRS"],  usd_positions),
            _fmt_group("METALS",     self._GROUP_LIMITS["METALS"],     metals_positions),
            _fmt_group("EQUITY",     self._GROUP_LIMITS["EQUITY"],     equity_positions),
            _fmt_group("CRYPTO",     self._GROUP_LIMITS["CRYPTO"],     crypto_positions),
            "─" * 50,
            f"  Total open: {len(open_positions)}",
        ]
        return "\n".join(lines)
