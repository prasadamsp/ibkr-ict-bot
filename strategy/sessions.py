"""
Session-Based Logic — London and New York trading sessions.

ICT uses session timing to:
1. Identify when smart money is active (London open, NY open).
2. Avoid trading during low-liquidity periods (Asian range, dead zone).
3. Look for "judas swing" — false move at session open to grab liquidity
   before the true directional move.

Session times (UTC):
    Asian:   23:00 – 03:00
    London:  07:00 – 11:00  (overlap with NY: 12:00–16:00)
    New York: 12:00 – 17:00
    Dead zone: 17:00 – 23:00 (avoid trading)

Key sessions for ICT:
    London Open Kill Zone:  07:00 – 09:00 UTC
    NY Open Kill Zone:      12:00 – 14:00 UTC
    London Close:           10:00 – 11:00 UTC
"""

from datetime import time, datetime, timezone
from typing import Optional, Tuple

import pandas as pd
import pytz

UTC = pytz.UTC


# Session windows as (start_hour_utc, start_min_utc, end_hour_utc, end_min_utc)
SESSION_WINDOWS = {
    "asian":           (23, 0, 3, 0),
    "london":          (7, 0, 11, 0),
    "london_kill":     (7, 0, 9, 0),
    "ny":              (12, 0, 17, 0),
    "ny_kill":         (12, 0, 14, 0),
    "london_close":    (10, 0, 11, 0),
    "dead_zone":       (17, 0, 23, 0),
}

# Preferred sessions for taking entries (highest probability)
HIGH_PROBABILITY_SESSIONS = {"london_kill", "ny_kill"}
ACCEPTABLE_SESSIONS = {"london", "ny", "london_close"}
AVOID_SESSIONS = {"asian", "dead_zone"}


def _utc_time(dt: pd.Timestamp) -> time:
    """Extract UTC time from a timezone-aware timestamp."""
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    else:
        dt = dt.tz_convert("UTC")
    return dt.time()


def get_session(dt: pd.Timestamp) -> str:
    """
    Return the session name for a given UTC timestamp.
    If multiple sessions overlap, returns the highest-priority one.
    """
    t = _utc_time(dt)
    h, m = t.hour, t.minute
    total_min = h * 60 + m

    def in_range(start_h, start_m, end_h, end_m):
        s = start_h * 60 + start_m
        e = end_h * 60 + end_m
        if s <= e:
            return s <= total_min < e
        else:  # wraps midnight
            return total_min >= s or total_min < e

    # Check in priority order
    if in_range(*SESSION_WINDOWS["london_kill"]):
        return "london_kill"
    if in_range(*SESSION_WINDOWS["ny_kill"]):
        return "ny_kill"
    if in_range(*SESSION_WINDOWS["london_close"]):
        return "london_close"
    if in_range(*SESSION_WINDOWS["london"]):
        return "london"
    if in_range(*SESSION_WINDOWS["ny"]):
        return "ny"
    if in_range(*SESSION_WINDOWS["asian"]):
        return "asian"
    return "dead_zone"


def is_tradeable_session(dt: pd.Timestamp) -> bool:
    """Return True if this timestamp falls in an acceptable trading session."""
    return get_session(dt) in ACCEPTABLE_SESSIONS | HIGH_PROBABILITY_SESSIONS


def is_kill_zone(dt: pd.Timestamp) -> bool:
    """Return True if this is a high-probability kill zone."""
    return get_session(dt) in HIGH_PROBABILITY_SESSIONS


def session_score(dt: pd.Timestamp) -> float:
    """
    Return 0–1 score for trade quality based on session.
    1.0 = kill zone, 0.5 = acceptable, 0.0 = avoid.
    """
    session = get_session(dt)
    if session in HIGH_PROBABILITY_SESSIONS:
        return 1.0
    elif session in ACCEPTABLE_SESSIONS:
        return 0.5
    else:
        return 0.0


def get_asian_range(df: pd.DataFrame) -> Optional[Tuple[float, float]]:
    """
    Return (asian_high, asian_low) for the most recent Asian session in the data.
    Used as the initial dealing range for the London session.
    """
    # Get the most recent date in data
    last_date = df.index[-1].date()

    # Asian session is typically from previous day 23:00 to current day 03:00
    asian_bars = []
    for ts, row in df.iterrows():
        sess = get_session(ts)
        if sess == "asian" and ts.date() >= last_date - pd.Timedelta(days=1):
            asian_bars.append(row)

    if not asian_bars:
        return None

    asian_df = pd.DataFrame(asian_bars)
    return float(asian_df["high"].max()), float(asian_df["low"].min())


def add_session_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'session' column to a bar DataFrame."""
    df = df.copy()
    df["session"] = df.index.map(get_session)
    return df
