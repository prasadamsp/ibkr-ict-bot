"""
btc_cycle.py — Bitcoin Halving Cycle Clock

Tracks Bitcoin's ~4-year halving cycle and maps the current date to a named
phase, directional bias, and position-size multiplier.

Halving cycle phases (days since last halving):
  - Year 1 (0–364):   "accumulation"  — market digesting the supply shock
  - Year 2 (365–729): "bull"          — strong uptrend phase
  - Year 3 (730–1094):"distribution"  — euphoria / blow-off tops
  - Year 4 (1095+):   "bear"          — pre-halving contraction

As of 2026-03-23 we are ~703 days post the April 2024 halving → "bull" phase.

Usage:
    from strategy.btc_cycle import HalvingClock

    clock = HalvingClock()
    print(clock.get_phase())           # "bull"
    print(clock.get_size_multiplier()) # 1.5
    clock.summary()
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Phase definitions — maps (year_number: int) → phase label
# Year number is computed as floor(days_since_halving / 365) + 1, capped at 4
# ---------------------------------------------------------------------------

_YEAR_TO_PHASE: dict[int, str] = {
    1: "accumulation",
    2: "bull",
    3: "distribution",
    4: "bear",
}

_PHASE_TO_BIAS: dict[str, str] = {
    "accumulation": "long_preferred",
    "bull":         "long_only",
    "distribution": "short_preferred",
    "bear":         "short_only",
}

_PHASE_TO_MULTIPLIER: dict[str, float] = {
    "accumulation": 0.8,
    "bull":         1.5,
    "distribution": 0.6,
    "bear":         0.3,
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class HalvingClock:
    """
    Tracks Bitcoin's 4-year halving cycle.

    All public methods accept an optional `dt` (datetime) argument.
    When `dt` is None, the current UTC wall-clock time is used.

    Class constants
    ---------------
    HALVING_DATES : list[datetime]
        All known Bitcoin block-reward halving dates.

    CYCLE_DAYS : int
        Approximate days in one full cycle (4 * 365 = 1460).
    """

    HALVING_DATES: list[datetime] = [
        datetime(2012, 11, 28, tzinfo=timezone.utc),
        datetime(2016,  7,  9, tzinfo=timezone.utc),
        datetime(2020,  5, 11, tzinfo=timezone.utc),
        datetime(2024,  4, 20, tzinfo=timezone.utc),  # most recent halving
    ]

    CYCLE_DAYS: int = 4 * 365  # ~1460 days

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _last_halving(self, dt: datetime) -> datetime:
        """Return the most recent halving date that is <= `dt`."""
        past = [h for h in self.HALVING_DATES if h <= dt]
        if not past:
            # Before the first known halving — return the earliest one
            return self.HALVING_DATES[0]
        return max(past)

    def _resolve(self, dt: Optional[datetime]) -> datetime:
        """Return `dt` if provided, else UTC now (timezone-aware)."""
        if dt is not None:
            # Ensure tz-aware for comparison with HALVING_DATES
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt
        return datetime.now(tz=timezone.utc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def days_since_halving(self, dt: Optional[datetime] = None) -> int:
        """
        Return the number of whole days elapsed since the last halving.

        Parameters
        ----------
        dt : datetime, optional  Reference date (default: now UTC).

        Returns
        -------
        int  Number of days (>= 0).
        """
        dt = self._resolve(dt)
        last = self._last_halving(dt)
        delta = dt - last
        return max(0, delta.days)

    def get_phase(self, dt: Optional[datetime] = None) -> str:
        """
        Return the current cycle phase name.

        Phases by year (365-day buckets):
          Year 1 (days   0–364): "accumulation"
          Year 2 (days 365–729): "bull"
          Year 3 (days 730–1094):"distribution"
          Year 4 (days 1095+):   "bear"

        Parameters
        ----------
        dt : datetime, optional

        Returns
        -------
        str  One of: "accumulation", "bull", "distribution", "bear"
        """
        days = self.days_since_halving(dt)
        # Determine which year of the cycle we're in (1-indexed, max 4)
        year = min(days // 365 + 1, 4)
        return _YEAR_TO_PHASE[year]

    def get_direction_bias(self, dt: Optional[datetime] = None) -> str:
        """
        Return the directional trading bias for the current phase.

        Mapping:
          "bull"         → "long_only"
          "accumulation" → "long_preferred"
          "distribution" → "short_preferred"
          "bear"         → "short_only"

        Parameters
        ----------
        dt : datetime, optional

        Returns
        -------
        str
        """
        phase = self.get_phase(dt)
        return _PHASE_TO_BIAS[phase]

    def get_size_multiplier(self, dt: Optional[datetime] = None) -> float:
        """
        Return a position-size multiplier for the current cycle phase.

        Multipliers:
          "bull"         → 1.5  (maximum aggression)
          "accumulation" → 0.8  (cautious building)
          "distribution" → 0.6  (reduce longs, small shorts)
          "bear"         → 0.3  (minimal exposure)

        Parameters
        ----------
        dt : datetime, optional

        Returns
        -------
        float
        """
        phase = self.get_phase(dt)
        return _PHASE_TO_MULTIPLIER[phase]

    def next_halving_estimate(self) -> datetime:
        """
        Estimate the date of the next halving.

        Approximation: last known halving + 4 years (1460 days).
        Bitcoin targets ~210,000 blocks between halvings at ~10 min/block.

        Returns
        -------
        datetime  Approximate next halving date.
        """
        last = self.HALVING_DATES[-1]
        return last + timedelta(days=self.CYCLE_DAYS)

    def summary(self, dt: Optional[datetime] = None) -> None:
        """
        Print a human-readable summary of the current halving cycle state.

        Parameters
        ----------
        dt : datetime, optional  Reference date (default: now UTC).
        """
        dt = self._resolve(dt)
        phase      = self.get_phase(dt)
        bias       = self.get_direction_bias(dt)
        multiplier = self.get_size_multiplier(dt)
        days       = self.days_since_halving(dt)
        last       = self._last_halving(dt)
        next_est   = self.next_halving_estimate()

        print("=" * 50)
        print("  Bitcoin Halving Cycle Summary")
        print("=" * 50)
        print(f"  Reference date  : {dt.strftime('%Y-%m-%d')}")
        print(f"  Last halving    : {last.strftime('%Y-%m-%d')}")
        print(f"  Days elapsed    : {days} days")
        print(f"  Cycle phase     : {phase.upper()}")
        print(f"  Direction bias  : {bias}")
        print(f"  Size multiplier : {multiplier}")
        print(f"  Next halving    : ~{next_est.strftime('%Y-%m-%d')} (est.)")
        print("=" * 50)
