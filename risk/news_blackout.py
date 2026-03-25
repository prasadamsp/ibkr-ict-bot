"""
News Blackout Guard
────────────────────
Blocks new trade entries during high-impact economic news releases.

Two sources of blackout windows:

1. RECURRING (auto-calculated, zero maintenance):
   - NFP   — first Friday of every month, 13:30 UTC
   - EIA   — every Wednesday, 14:30 UTC

2. SCHEDULED (loaded from config/news_calendar.yaml):
   - FOMC, CPI, PCE, BOE, ECB, GDP, PPI, Retail Sales (2026 dates pre-loaded)

Usage
─────
    from risk.news_blackout import NewsBlackout
    blackout = NewsBlackout()

    # In strategy router, before any signal generation:
    blocked, reason = blackout.is_blocked("EURUSD", current_dt)
    if blocked:
        log.info("NEWS BLACKOUT: %s", reason)
        return None

Instrument groups
─────────────────
  ALL    → blocks all 7 instruments
  USD    → XAUUSD, NAS100, EURUSD, GBPUSD, XAGUSD, OIL
  EURUSD → EURUSD only
  GBPUSD → GBPUSD only

Architecture
────────────
    NewsBlackout
         │
         ├── _build_recurring_windows()   → NFP / EIA auto-dates
         ├── _load_scheduled_windows()    → YAML calendar
         └── is_blocked(symbol, dt)       → O(n) scan of relevant windows
"""

from __future__ import annotations

import calendar
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

_log = logging.getLogger("risk")

# Instruments covered by each group tag in the YAML
_GROUP_MAP = {
    "ALL":    {"XAUUSD", "NAS100", "EURUSD", "GBPUSD", "BTC", "XAGUSD", "OIL"},
    "USD":    {"XAUUSD", "NAS100", "EURUSD", "GBPUSD", "XAGUSD", "OIL"},
    "EURUSD": {"EURUSD"},
    "GBPUSD": {"GBPUSD"},
    "BTC":    {"BTC"},
    "OIL":   {"OIL"},
}

_CALENDAR_PATH = Path(__file__).resolve().parent.parent / "config" / "news_calendar.yaml"


# ---------------------------------------------------------------------------
# Window container
# ---------------------------------------------------------------------------

@dataclass
class BlackoutWindow:
    """A single news blackout window."""
    event_type:  str          # e.g. "NFP", "FOMC", "CPI"
    instruments: set          # set of symbol strings
    start_utc:   datetime     # window open (tz-aware UTC)
    end_utc:     datetime     # window close (tz-aware UTC)

    def covers(self, symbol: str, dt: datetime) -> bool:
        """Return True if *symbol* at *dt* falls inside this window."""
        dt_utc = _to_utc(dt)
        return symbol.upper() in self.instruments and self.start_utc <= dt_utc <= self.end_utc

    def __repr__(self) -> str:
        return (
            f"BlackoutWindow({self.event_type} "
            f"{self.start_utc.strftime('%Y-%m-%d %H:%M')}–"
            f"{self.end_utc.strftime('%H:%M')} UTC "
            f"[{', '.join(sorted(self.instruments))}])"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_utc(dt: datetime) -> datetime:
    """Coerce datetime to UTC-aware."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _make_window(
    event_type: str,
    instruments: set,
    release_utc: datetime,
    before_min: int,
    after_min: int,
) -> BlackoutWindow:
    return BlackoutWindow(
        event_type=event_type,
        instruments=instruments,
        start_utc=release_utc - timedelta(minutes=before_min),
        end_utc=release_utc + timedelta(minutes=after_min),
    )


def _first_friday_of_month(year: int, month: int) -> int:
    """Return the day-of-month of the first Friday in the given year/month."""
    c = calendar.monthcalendar(year, month)
    # calendar.monthcalendar returns weeks; Friday = index 4
    for week in c:
        if week[4] != 0:
            return week[4]
    raise ValueError(f"No Friday found in {year}-{month}")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class NewsBlackout:
    """
    Manages news blackout windows for all 7 instruments.

    Parameters
    ----------
    calendar_path :
        Path to the YAML news calendar file.  Defaults to
        config/news_calendar.yaml relative to the project root.
    lookahead_days :
        How many days of recurring windows to pre-generate on startup.
        Default 90 days (~3 months ahead) — refreshed if you restart.
    """

    def __init__(
        self,
        calendar_path: Optional[Path] = None,
        lookahead_days: int = 90,
    ) -> None:
        self._windows: List[BlackoutWindow] = []
        self._lookahead_days = lookahead_days
        self._calendar_path = calendar_path or _CALENDAR_PATH

        # Load default window widths from YAML
        self._defaults: dict = {}
        self._load_defaults()

        # Build windows
        self._build_recurring_windows()
        self._load_scheduled_windows()

        _log.info(
            "NewsBlackout initialised — %d blackout windows loaded "
            "(%d recurring + %d scheduled)",
            len(self._windows),
            self._recurring_count,
            self._scheduled_count,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_blocked(
        self, symbol: str, dt: datetime
    ) -> Tuple[bool, str]:
        """
        Check whether trading symbol *symbol* is blacked out at *dt*.

        Returns
        -------
        (blocked: bool, reason: str)
            reason is empty string when not blocked.
        """
        sym = symbol.upper().strip()
        for w in self._windows:
            if w.covers(sym, dt):
                return True, (
                    f"{w.event_type} blackout "
                    f"{w.start_utc.strftime('%H:%M')}–"
                    f"{w.end_utc.strftime('%H:%M')} UTC"
                )
        return False, ""

    def upcoming(self, symbol: str, dt: datetime, hours: int = 24) -> List[BlackoutWindow]:
        """
        Return blackout windows affecting *symbol* within the next *hours* hours.
        Useful for dashboard display.
        """
        sym = symbol.upper().strip()
        dt_utc = _to_utc(dt)
        horizon = dt_utc + timedelta(hours=hours)
        return [
            w for w in self._windows
            if sym in w.instruments and dt_utc <= w.start_utc <= horizon
        ]

    # ------------------------------------------------------------------
    # Recurring windows (NFP, EIA)
    # ------------------------------------------------------------------

    def _build_recurring_windows(self) -> None:
        """
        Auto-generate blackout windows for recurring releases:
        - NFP: first Friday of every month, 13:30 UTC
        - EIA: every Wednesday, 14:30 UTC
        """
        now = datetime.now(tz=timezone.utc)
        end = now + timedelta(days=self._lookahead_days)
        count = 0

        nfp_cfg = self._defaults.get("nfp", {"before_min": 15, "after_min": 30})
        eia_cfg = self._defaults.get("eia", {"before_min": 5,  "after_min": 15})

        nfp_instruments = _GROUP_MAP["ALL"]
        eia_instruments = _GROUP_MAP["OIL"]

        # NFP — first Friday of each month
        months_seen = set()
        cursor = now.replace(day=1)
        while cursor <= end:
            ym = (cursor.year, cursor.month)
            if ym not in months_seen:
                months_seen.add(ym)
                day = _first_friday_of_month(cursor.year, cursor.month)
                release = datetime(
                    cursor.year, cursor.month, day,
                    13, 30, tzinfo=timezone.utc
                )
                if release >= now:
                    self._windows.append(_make_window(
                        "NFP", nfp_instruments, release,
                        nfp_cfg["before_min"], nfp_cfg["after_min"],
                    ))
                    count += 1
            cursor += timedelta(days=32)
            cursor = cursor.replace(day=1)

        # EIA — every Wednesday 14:30 UTC
        # Find next Wednesday from now
        days_until_wed = (2 - now.weekday()) % 7  # 2 = Wednesday
        next_wed = now + timedelta(days=days_until_wed)
        next_wed = next_wed.replace(hour=14, minute=30, second=0, microsecond=0)
        eia_date = next_wed
        while eia_date <= end:
            self._windows.append(_make_window(
                "EIA", eia_instruments, eia_date,
                eia_cfg["before_min"], eia_cfg["after_min"],
            ))
            count += 1
            eia_date += timedelta(weeks=1)

        self._recurring_count = count

    # ------------------------------------------------------------------
    # Scheduled windows (YAML)
    # ------------------------------------------------------------------

    def _load_defaults(self) -> None:
        """Load default window widths from the YAML file."""
        if not self._calendar_path.exists():
            _log.warning("NewsBlackout: calendar not found at %s", self._calendar_path)
            return
        with open(self._calendar_path) as f:
            data = yaml.safe_load(f) or {}
        self._defaults = data.get("defaults", {})

    def _load_scheduled_windows(self) -> None:
        """Parse the YAML news calendar and build BlackoutWindow objects."""
        if not self._calendar_path.exists():
            _log.warning("NewsBlackout: calendar not found — scheduled windows skipped")
            self._scheduled_count = 0
            return

        with open(self._calendar_path) as f:
            data = yaml.safe_load(f) or {}

        count = 0
        now = datetime.now(tz=timezone.utc)

        # All event types except 'defaults'
        event_types = [k for k in data if k != "defaults"]

        for event_type in event_types:
            entries = data[event_type]
            if not isinstance(entries, list):
                continue

            cfg = self._defaults.get(event_type, {"before_min": 15, "after_min": 30})
            before = cfg.get("before_min", 15)
            after  = cfg.get("after_min", 30)

            for entry in entries:
                try:
                    date_str  = str(entry["date"])
                    time_str  = str(entry["time"])
                    instr_key = str(entry.get("instruments", "ALL"))

                    release_utc = datetime.strptime(
                        f"{date_str} {time_str}", "%Y-%m-%d %H:%M"
                    ).replace(tzinfo=timezone.utc)

                    # Skip past events (keep a small buffer to catch in-progress)
                    if release_utc + timedelta(minutes=after) < now:
                        continue

                    instruments = _GROUP_MAP.get(instr_key, _GROUP_MAP["ALL"])

                    self._windows.append(_make_window(
                        event_type.upper(), instruments, release_utc, before, after
                    ))
                    count += 1

                except (KeyError, ValueError) as exc:
                    _log.warning(
                        "NewsBlackout: skipping malformed entry in '%s': %s — %s",
                        event_type, entry, exc
                    )

        self._scheduled_count = count

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of all loaded windows."""
        lines = [
            f"NewsBlackout — {len(self._windows)} windows "
            f"({self._recurring_count} recurring, {self._scheduled_count} scheduled)"
        ]
        for w in sorted(self._windows, key=lambda x: x.start_utc):
            lines.append(
                f"  {w.event_type:<12} {w.start_utc.strftime('%Y-%m-%d %H:%M')}–"
                f"{w.end_utc.strftime('%H:%M')} UTC  "
                f"[{', '.join(sorted(w.instruments))}]"
            )
        return "\n".join(lines)
