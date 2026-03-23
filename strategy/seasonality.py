"""
seasonality.py — Seasonality Calendar for Position Sizing

Returns a bias multiplier (0.5–1.5) for each instrument based on the month of
the year, encoding well-known seasonal tendencies in commodities, FX, equities,
and crypto.

For BTC, the multiplier delegates to HalvingClock (btc_cycle.py) so that the
position size reflects the current 4-year halving cycle phase rather than
calendar month alone.

Usage:
    from strategy.seasonality import SeasonalityCalendar
    from datetime import datetime

    cal = SeasonalityCalendar()
    mult = cal.get_multiplier("XAUUSD", datetime(2026, 3, 1))  # → 0.9
    note = cal.get_note("NAS100", datetime(2026, 1, 15))       # → "January effect"
    cal.is_eia_day(datetime(2026, 3, 18))                      # → True (Wednesday)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from strategy.btc_cycle import HalvingClock


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class SeasonalBias:
    """Represents a single seasonal rule for a symbol / month pair."""

    symbol:     str    # Instrument identifier (e.g. "XAUUSD")
    month:      int    # Calendar month (1 = January … 12 = December)
    multiplier: float  # Position-size multiplier (0.5 – 1.5)
    note:       str    # Human-readable explanation


# ---------------------------------------------------------------------------
# Seasonal bias tables
# ---------------------------------------------------------------------------
# Each entry is (month, multiplier, note).  Months not listed default to 1.0.

_XAUUSD_BIASES: dict[int, tuple[float, str]] = {
    # Seasonal gold bull window — Sep through Feb
    9:  (1.3, "Gold seasonal bull"),
    10: (1.3, "Gold seasonal bull"),
    11: (1.3, "Gold seasonal bull"),
    12: (1.3, "Gold seasonal bull"),
    1:  (1.3, "Gold seasonal bull"),
    2:  (1.3, "Gold seasonal bull"),
    # Spring weakness
    3:  (0.9, "Gold spring weakness"),
    4:  (0.9, "Gold spring weakness"),
    # May – Aug neutral (no entry needed; falls through to default 1.0)
}

_XAGUSD_BIASES: dict[int, tuple[float, str]] = {
    # Silver follows gold in the bull window but with less precision
    9:  (1.3, "Silver seasonal bull"),
    10: (1.3, "Silver seasonal bull"),
    11: (1.3, "Silver seasonal bull"),
    12: (1.3, "Silver seasonal bull"),
    1:  (1.3, "Silver seasonal bull"),
    2:  (1.3, "Silver seasonal bull"),
    # Silver underperforms gold more sharply in spring
    3:  (0.7, "Silver spring weakness (underperforms gold)"),
    4:  (0.7, "Silver spring weakness (underperforms gold)"),
}

_NAS100_BIASES: dict[int, tuple[float, str]] = {
    1:  (1.2, "January effect"),
    8:  (0.7, "Summer doldrums"),
    10: (1.2, "Q4 tech rally"),
    11: (1.2, "Q4 tech rally"),
    12: (1.2, "Q4 tech rally"),
}

_EURUSD_BIASES: dict[int, tuple[float, str]] = {
    1:  (0.9, "USD typically strong Q1"),
    2:  (0.9, "USD typically strong Q1"),
    3:  (0.9, "USD typically strong Q1"),
    7:  (0.7, "Low summer volatility"),
    8:  (0.7, "Low summer volatility"),
    9:  (0.7, "Low summer volatility"),
}

# GBPUSD: BOE/news-driven, no meaningful seasonal edge
_GBPUSD_BIASES: dict[int, tuple[float, str]] = {}

_OIL_BIASES: dict[int, tuple[float, str]] = {
    2:  (1.2, "Refinery maintenance, tight supply"),
    3:  (1.2, "Refinery maintenance, tight supply"),
    4:  (1.2, "Refinery maintenance, tight supply"),
    5:  (1.2, "Driving season"),
    6:  (1.2, "Driving season"),
    7:  (1.2, "Driving season"),
    8:  (1.2, "Driving season"),
    10: (1.1, "Heating season buildup"),
    11: (1.1, "Heating season buildup"),
    12: (0.8, "Mild demand"),
    1:  (0.8, "Mild demand"),
}

# Master registry — maps normalised symbol → bias table
_BIAS_TABLES: dict[str, dict[int, tuple[float, str]]] = {
    "XAUUSD": _XAUUSD_BIASES,
    "XAGUSD": _XAGUSD_BIASES,
    "NAS100": _NAS100_BIASES,
    "EURUSD": _EURUSD_BIASES,
    "GBPUSD": _GBPUSD_BIASES,
    "OIL":    _OIL_BIASES,
    # BTC is handled separately via HalvingClock
}

_DEFAULT_MULTIPLIER = 1.0
_DEFAULT_NOTE       = "No seasonal bias defined"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SeasonalityCalendar:
    """
    Returns seasonality-adjusted position-size multipliers per instrument.

    Supported symbols
    -----------------
    XAUUSD, XAGUSD, NAS100, EURUSD, GBPUSD, OIL, BTC

    For BTC the multiplier comes from HalvingClock.get_size_multiplier() and
    the note reflects the current halving cycle phase.

    For all other instruments a hardcoded monthly table is used.  Months not
    explicitly listed fall back to 1.0 (neutral).
    """

    def __init__(self) -> None:
        self._halving_clock = HalvingClock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_multiplier(self, symbol: str, dt: datetime) -> float:
        """
        Return the seasonality multiplier for `symbol` at date `dt`.

        Parameters
        ----------
        symbol : str      Instrument code (case-insensitive).
        dt     : datetime Reference date.

        Returns
        -------
        float  Multiplier in [0.5, 1.5].  1.0 = neutral / unknown.
        """
        norm = symbol.upper().strip()

        if norm == "BTC":
            return self._halving_clock.get_size_multiplier(dt)

        table = _BIAS_TABLES.get(norm)
        if table is None:
            # Unknown symbol — return neutral
            return _DEFAULT_MULTIPLIER

        entry = table.get(dt.month)
        if entry is None:
            return _DEFAULT_MULTIPLIER

        return entry[0]

    def get_note(self, symbol: str, dt: datetime) -> str:
        """
        Return a human-readable note describing the seasonal bias.

        Parameters
        ----------
        symbol : str
        dt     : datetime

        Returns
        -------
        str
        """
        norm = symbol.upper().strip()

        if norm == "BTC":
            phase = self._halving_clock.get_phase(dt)
            mult  = self._halving_clock.get_size_multiplier(dt)
            days  = self._halving_clock.days_since_halving(dt)
            return (
                f"BTC halving cycle: {phase} phase "
                f"(day {days}, multiplier={mult})"
            )

        table = _BIAS_TABLES.get(norm)
        if table is None:
            return _DEFAULT_NOTE

        entry = table.get(dt.month)
        if entry is None:
            return "Neutral"

        return entry[1]

    def get_bias(self, symbol: str, dt: datetime) -> SeasonalBias:
        """
        Return a full SeasonalBias dataclass for the given symbol and date.

        Parameters
        ----------
        symbol : str
        dt     : datetime

        Returns
        -------
        SeasonalBias
        """
        return SeasonalBias(
            symbol=symbol.upper(),
            month=dt.month,
            multiplier=self.get_multiplier(symbol, dt),
            note=self.get_note(symbol, dt),
        )

    @staticmethod
    def is_eia_day(dt: datetime) -> bool:
        """
        Return True if `dt` falls on a Wednesday (EIA petroleum report day).

        The U.S. Energy Information Administration (EIA) publishes its weekly
        petroleum inventory report every Wednesday at 10:30 ET.  OIL and
        energy-related instruments often see elevated volatility on this day.

        Parameters
        ----------
        dt : datetime

        Returns
        -------
        bool
        """
        # Python weekday(): 0=Monday, 1=Tuesday, 2=Wednesday …
        return dt.weekday() == 2

    def get_btc_cycle_month(self, dt: datetime) -> str:
        """
        Return the Bitcoin halving cycle phase name for the given date.

        Delegates to HalvingClock.get_phase() which uses days-since-halving
        bucketed into 365-day years.

        Returns
        -------
        str  One of: "accumulation", "bull", "distribution", "bear"
        """
        return self._halving_clock.get_phase(dt)

    def all_biases(self, symbol: str) -> list[SeasonalBias]:
        """
        Return a list of SeasonalBias entries for every month (1–12) for
        the given symbol.  Useful for reporting or UI display.

        BTC entries show the halving phase as of the given year's months
        relative to today's halving clock (approximate — month-level only).

        Parameters
        ----------
        symbol : str

        Returns
        -------
        list[SeasonalBias]
        """
        norm = symbol.upper().strip()
        now_year = datetime.utcnow().year
        return [
            self.get_bias(norm, datetime(now_year, m, 1))
            for m in range(1, 13)
        ]
