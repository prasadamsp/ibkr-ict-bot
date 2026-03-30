"""
btc_cycle.py — Bitcoin Halving Cycle Phase Awareness

Tracks which phase of the 4-year BTC halving cycle we are currently in
and exposes helper functions that BTCStrategy can use to:
  - determine the current cycle phase (enum)
  - apply a position-size multiplier
  - apply a directional bias filter

Halving schedule (block reward cut in half):
  - Nov 28 2012 — 1st halving (50 → 25 BTC)
  - Jul  9 2016 — 2nd halving (25 → 12.5 BTC)
  - May 11 2020 — 3rd halving (12.5 → 6.25 BTC)
  - Apr 20 2024 — 4th halving (6.25 → 3.125 BTC)  <-- most recent

Phase boundaries (days elapsed since the last halving):
  - MARKUP       0 – 365 days  : supply shock drives strong upward impulse
  - LATE_BULL  366 – 730 days  : momentum continues, euphoria risk grows
                                 (current as of Mar 2026 → ~340 days post-Apr 2024)
  - DISTRIBUTION 731–1095 days : price peaks, smart money distributes
  - ACCUMULATION 1096–1460 days: pre-halving base-building, lower volatility

Integration with BTCStrategy
-----------------------------
Typical call-site pattern::

    from strategy.btc_cycle import get_cycle_phase, get_size_multiplier, get_direction_bias
    from datetime import date

    phase      = get_cycle_phase(date.today())
    multiplier = get_size_multiplier(phase)   # e.g. 1.5
    bias       = get_direction_bias(phase)    # e.g. +1

    final_size = base_size * multiplier
    # Skip short signals when bias == +1 (long-only filter)
    if signal_direction == -1 and bias == 1:
        return  # suppress short

Usage example::

    from strategy.btc_cycle import cycle_summary
    from datetime import date

    print(cycle_summary(date.today()))
    # "BTC Cycle | Phase: LATE_BULL | Day 345/1460 | Size: 1.25x | Bias: LONG"
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import List


# ---------------------------------------------------------------------------
# Halving dates — add future halvings here as they occur (~every 4 years)
# ---------------------------------------------------------------------------

HALVING_DATES: List[date] = [
    date(2012, 11, 28),  # 1st halving: 50 → 25 BTC/block
    date(2016,  7,  9),  # 2nd halving: 25 → 12.5 BTC/block
    date(2020,  5, 11),  # 3rd halving: 12.5 → 6.25 BTC/block
    date(2024,  4, 20),  # 4th halving: 6.25 → 3.125 BTC/block  <-- most recent
    # date(2028, ~Apr):  # 5th halving — add when confirmed
]


# ---------------------------------------------------------------------------
# Phase enum
# ---------------------------------------------------------------------------

class BtcCyclePhase(Enum):
    """Named phases of Bitcoin's 4-year halving cycle.

    Boundary logic (days since last halving):
      MARKUP          0 –  365  Supply shock, historically strongest uptrend.
      LATE_BULL     366 –  730  Continued momentum, elevated risk of reversal.
      DISTRIBUTION  731 – 1095  Cycle top region, smart-money distribution.
      ACCUMULATION 1096 – 1460  Pre-halving base, reduced volatility, no shorts.
    """

    MARKUP       = "MARKUP"
    LATE_BULL    = "LATE_BULL"
    DISTRIBUTION = "DISTRIBUTION"
    ACCUMULATION = "ACCUMULATION"


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def get_cycle_phase(as_of: date) -> BtcCyclePhase:
    """Return the BTC halving cycle phase for a given date.

    Algorithm
    ---------
    1. Find the most recent halving date that is <= `as_of`.
    2. Calculate days_elapsed = (as_of - last_halving).days
    3. Map days_elapsed to a BtcCyclePhase using the boundaries:
         [0,   365] → MARKUP
         [366, 730] → LATE_BULL
         [731, 1095]→ DISTRIBUTION
         [1096+]    → ACCUMULATION

    Parameters
    ----------
    as_of : date
        The reference date for which to determine the cycle phase.

    Returns
    -------
    BtcCyclePhase
        The phase the market is currently in.

    Examples
    --------
    >>> get_cycle_phase(date(2024, 10, 20))   # ~183 days after Apr-2024 halving
    <BtcCyclePhase.MARKUP: 'MARKUP'>
    >>> get_cycle_phase(date(2026, 3, 15))    # ~330 days → LATE_BULL
    <BtcCyclePhase.LATE_BULL: 'LATE_BULL'>
    """
    # TODO: implement phase detection
    #   Steps:
    #     1. Filter HALVING_DATES to those <= as_of, take the max.
    #     2. Compute days_elapsed = (as_of - last_halving).days
    #     3. Use if/elif ladder to map days_elapsed to a BtcCyclePhase member.
    #   Edge cases to handle:
    #     - as_of is before ALL known halvings → treat as ACCUMULATION or raise?
    #     - as_of is exactly on a halving date → counts as day 0 → MARKUP
    raise NotImplementedError("TODO: implement get_cycle_phase")


def get_size_multiplier(phase: BtcCyclePhase) -> float:
    """Return a position-size multiplier for the given cycle phase.

    The multiplier is applied to the base position size calculated by
    the risk manager before sending orders.  Values > 1.0 increase size;
    values < 1.0 reduce it.

    Suggested multipliers (adjust after back-testing):
      MARKUP        → 1.50  (max aggression — strong trend, tight stops)
      LATE_BULL     → 1.25  (still bullish but approaching cycle peak)
      DISTRIBUTION  → 0.75  (reduce exposure, trend losing momentum)
      ACCUMULATION  → 0.50  (base-building, high uncertainty)

    Parameters
    ----------
    phase : BtcCyclePhase
        The current cycle phase enum member.

    Returns
    -------
    float
        Multiplier in the range [0.5, 1.5].

    Raises
    ------
    ValueError
        If `phase` is not a recognised BtcCyclePhase member.
    """
    # TODO: implement via a dict lookup or match/case statement.
    #   _PHASE_MULTIPLIERS = {
    #       BtcCyclePhase.MARKUP:       1.50,
    #       BtcCyclePhase.LATE_BULL:    1.25,
    #       BtcCyclePhase.DISTRIBUTION: 0.75,
    #       BtcCyclePhase.ACCUMULATION: 0.50,
    #   }
    #   return _PHASE_MULTIPLIERS[phase]
    raise NotImplementedError("TODO: implement get_size_multiplier")


def get_direction_bias(phase: BtcCyclePhase) -> int:
    """Return a directional trading bias integer for the given cycle phase.

    The bias is used by BTCStrategy as a filter:
      +1  → LONG only   (suppress all short signals)
       0  → NEUTRAL     (allow both directions, normal logic)
      -1  → SHORT only  (suppress all long signals)  [reserved, not used yet]

    Suggested biases:
      MARKUP        → +1  (long-only, no shorts)
      LATE_BULL     → +1  (long-only, no shorts)
      DISTRIBUTION  →  0  (neutral — both directions OK, reduced size)
      ACCUMULATION  →  0  (neutral — avoid shorts near potential bottom)

    Parameters
    ----------
    phase : BtcCyclePhase
        The current cycle phase enum member.

    Returns
    -------
    int
        One of: +1 (long), 0 (neutral), -1 (short).

    Raises
    ------
    ValueError
        If `phase` is not a recognised BtcCyclePhase member.
    """
    # TODO: implement via a dict lookup or match/case statement.
    #   _PHASE_BIAS = {
    #       BtcCyclePhase.MARKUP:       +1,
    #       BtcCyclePhase.LATE_BULL:    +1,
    #       BtcCyclePhase.DISTRIBUTION:  0,
    #       BtcCyclePhase.ACCUMULATION:  0,
    #   }
    #   return _PHASE_BIAS[phase]
    raise NotImplementedError("TODO: implement get_direction_bias")


def cycle_summary(as_of: date) -> str:
    """Return a human-readable one-line status string for the cycle state.

    Intended for logging, dashboard display, and quick operator checks.

    Format::

        "BTC Cycle | Phase: LATE_BULL | Day 345/1460 | Size: 1.25x | Bias: LONG"

    Parameters
    ----------
    as_of : date
        The reference date for the summary.

    Returns
    -------
    str
        Single-line status string.

    Example
    -------
    >>> print(cycle_summary(date(2026, 3, 15)))
    BTC Cycle | Phase: LATE_BULL | Day 330/1460 | Size: 1.25x | Bias: LONG
    """
    # TODO: implement the summary string.
    #   Steps:
    #     1. Call get_cycle_phase(as_of) → phase
    #     2. Compute days_elapsed (reuse logic from get_cycle_phase or extract helper)
    #     3. Call get_size_multiplier(phase) → multiplier
    #     4. Call get_direction_bias(phase)  → bias_int
    #     5. Convert bias_int to a label: +1 → "LONG", 0 → "NEUTRAL", -1 → "SHORT"
    #     6. Format and return the string.
    raise NotImplementedError("TODO: implement cycle_summary")
