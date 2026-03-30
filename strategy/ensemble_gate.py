"""
ensemble_gate.py — EnsembleGate

Implements a multi-family algorithm voting gate that requires ≥ 3 out of 4
algorithm families to agree on direction before a trading signal is emitted.

Voting Logic
------------
Four algorithm families are represented on every bar:

    1. trend        — ema_pullback, ma_crossover
    2. momentum     — macd_momentum
    3. breakout     — donchian_breakout
    4. mean_reversion — bb_rsi, rsi_extreme, zscore_reversion

For each bar the gate:
  a) Uses the pre-selected "best" algo for the instrument (already chosen by
     the research layer / auto_selector) as the primary voter.
  b) Picks one representative from each of the *other* three families (the
     first available algo in that family list by default, or the one with the
     highest walk-forward Sharpe if rankings are injected at construction).
  c) Runs all four voters on the current bar via their generate() interface.
  d) Counts how many of the four returned a signal and what direction each one
     suggests (long → "bullish", short → "bearish").
  e) Emits a direction only if ≥ 3 of the 4 voters agree on the *same*
     direction; otherwise returns None (gate blocks the signal).

Usage
-----
    families = {
        "trend":           ["ema_pullback", "ma_crossover"],
        "momentum":        ["macd_momentum"],
        "breakout":        ["donchian_breakout"],
        "mean_reversion":  ["bb_rsi", "rsi_extreme", "zscore_reversion"],
    }
    gate = EnsembleGate(symbol="NAS100", families=families)

    direction, votes_for, total_voters = gate.vote(
        symbol="NAS100",
        m15_df=m15_df,
        h1_df=h1_df,
        current_dt=current_dt,
    )
    if direction is not None:
        # signal passed the gate — proceed to risk sizing
        ...
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

from research.algos.base import BaseAlgo, AlgoSignal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry: maps algo name → concrete class
# Populate this dict as each algo module is imported.
# ---------------------------------------------------------------------------
ALGO_REGISTRY: Dict[str, type] = {}
# TODO: Import and register all algo classes here, e.g.:
#   from research.algos.ema_pullback   import EMAPullbackAlgo
#   from research.algos.ma_crossover   import MACrossoverAlgo
#   from research.algos.macd_momentum  import MACDMomentumAlgo
#   from research.algos.donchian       import DonchianBreakoutAlgo
#   from research.algos.bb_rsi         import BBRSIAlgo
#   from research.algos.rsi_extreme    import RSIExtremeAlgo
#   from research.algos.zscore         import ZScoreReversionAlgo
#
#   ALGO_REGISTRY = {
#       "ema_pullback":       EMAPullbackAlgo,
#       "ma_crossover":       MACrossoverAlgo,
#       "macd_momentum":      MACDMomentumAlgo,
#       "donchian_breakout":  DonchianBreakoutAlgo,
#       "bb_rsi":             BBRSIAlgo,
#       "rsi_extreme":        RSIExtremeAlgo,
#       "zscore_reversion":   ZScoreReversionAlgo,
#   }


class EnsembleGate:
    """
    Multi-family voting gate for algorithm signals.

    Attributes
    ----------
    symbol : str
        Instrument this gate is configured for (e.g. "NAS100").
    families : Dict[str, List[str]]
        Mapping of family name → ordered list of algo names belonging to
        that family.  The first entry is treated as the preferred
        representative unless overridden by ``family_rankings``.
    primary_algo_name : str
        Name of the best algo for this instrument as determined by the
        research layer.  Set via ``set_primary`` after construction.
    family_rankings : Dict[str, str]
        Optional override: maps family name → preferred algo name within
        that family (e.g. chosen by highest walk-forward Sharpe).
    _voters : Dict[str, BaseAlgo]
        Lazily instantiated algo objects, keyed by algo name.
    """

    def __init__(
        self,
        symbol: str,
        families: Dict[str, List[str]],
        family_rankings: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Initialise the gate for a given symbol and family configuration.

        Parameters
        ----------
        symbol : str
            Instrument identifier (must match keys used in the rest of the
            trading stack).
        families : Dict[str, List[str]]
            Four-family mapping described in the module docstring.  Expected
            keys: "trend", "momentum", "breakout", "mean_reversion".
        family_rankings : Dict[str, str] | None
            If provided, maps each family name to the algo name that should
            act as that family's representative.  Overrides the default
            first-in-list selection.
        """
        # TODO: Validate that exactly 4 families are present.
        # TODO: Validate that every algo name in families exists in ALGO_REGISTRY.
        # TODO: Store symbol, families, family_rankings on self.
        # TODO: Initialise self.primary_algo_name = None (set later via set_primary).
        # TODO: Initialise self._voters = {} (populate lazily in _get_voter).
        raise NotImplementedError("TODO: implement __init__")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_primary(self, algo_name: str) -> None:
        """
        Declare which algo is the primary voter for this symbol.

        This is typically called by the strategy router after the research
        layer (auto_selector) has resolved the best algo per instrument.

        Parameters
        ----------
        algo_name : str
            Must be a key in ALGO_REGISTRY and must belong to exactly one
            family defined in ``self.families``.
        """
        # TODO: Validate algo_name exists in ALGO_REGISTRY.
        # TODO: Identify which family the primary algo belongs to.
        # TODO: Store self.primary_algo_name = algo_name.
        # TODO: Store self._primary_family = <resolved family name>.
        raise NotImplementedError("TODO: implement set_primary")

    def vote(
        self,
        symbol: str,
        m15_df: pd.DataFrame,
        h1_df: pd.DataFrame,
        current_dt,
    ) -> Tuple[Optional[str], int, int]:
        """
        Run the four-family vote on the current bar and return the verdict.

        Parameters
        ----------
        symbol : str
            Instrument identifier (used for logging; should match self.symbol).
        m15_df : pd.DataFrame
            15-minute OHLCV DataFrame sliced up to and including current_dt.
            Columns: open, high, low, close, volume.
        h1_df : pd.DataFrame
            1-hour OHLCV DataFrame sliced up to and including current_dt.
        current_dt : datetime-like
            Timestamp of the bar being evaluated.  Used to derive the bar
            index (idx) passed to each algo's generate() method.

        Returns
        -------
        direction : str | None
            "bullish", "bearish", or None if the gate blocks the signal.
        votes_for : int
            Number of voters that agreed on the emitted direction.
            0 when direction is None.
        total_voters : int
            Number of voters that returned a non-None signal this bar
            (always between 0 and 4).

        Implementation Notes
        --------------------
        1. Call ``_select_voters()`` to build the list of 4 algo instances
           (one per family).
        2. Derive ``idx`` as the integer position of current_dt in m15_df.
        3. Call ``voter.generate(m15_df, h1_df, idx)`` for each voter.
        4. Collect non-None AlgoSignal results; map signal.direction to
           "bullish" (long) or "bearish" (short).
        5. Count votes per direction.  If the majority direction has count
           ≥ 3, return (direction, count, total_with_signal).
        6. Otherwise return (None, 0, total_with_signal).
        7. Log a DEBUG line with the per-voter breakdown.
        """
        # TODO: Guard — raise if set_primary() has not been called yet.
        # TODO: Call self._select_voters() → voters: List[BaseAlgo]
        # TODO: Determine bar index (idx) from current_dt and m15_df.index.
        # TODO: Loop over voters, call generate(), collect (algo_name, direction).
        # TODO: Tally directions → {"bullish": n, "bearish": n}.
        # TODO: Determine winning direction; check ≥ 3 threshold.
        # TODO: Return (winning_direction_or_None, votes_for, total_voters).
        raise NotImplementedError("TODO: implement vote")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _select_voters(self) -> List[BaseAlgo]:
        """
        Build the list of exactly 4 algo instances — one per family.

        Selection rules:
        - The primary algo's family uses the primary algo instance.
        - For every other family, use the algo specified in
          ``family_rankings`` if present; otherwise use the first algo
          name in the family's list.
        - Algo instances are cached in ``self._voters`` to avoid
          re-instantiation on every bar.

        Returns
        -------
        List[BaseAlgo]
            Exactly 4 BaseAlgo instances in deterministic order
            (sorted by family name for reproducibility).

        Implementation Notes
        --------------------
        - Iterate over sorted(self.families.keys()).
        - For each family decide which algo name to use (primary if family
          matches self._primary_family, else rankings or first-in-list).
        - Call self._get_voter(algo_name) to get/create the instance.
        """
        # TODO: Implement voter selection per family.
        # TODO: Return list of 4 BaseAlgo instances.
        raise NotImplementedError("TODO: implement _select_voters")

    def _get_voter(self, algo_name: str) -> BaseAlgo:
        """
        Return a cached algo instance, creating it on first access.

        Parameters
        ----------
        algo_name : str
            Must be a key in ALGO_REGISTRY.

        Returns
        -------
        BaseAlgo
            Instantiated algo with default hyperparameters.

        Implementation Notes
        --------------------
        - Check self._voters cache first.
        - If not cached, look up ALGO_REGISTRY[algo_name], instantiate
          with no arguments (rely on algo defaults), cache and return.
        - Raise KeyError with a helpful message if algo_name is missing
          from the registry.
        """
        # TODO: Check self._voters cache.
        # TODO: Instantiate from ALGO_REGISTRY[algo_name]().
        # TODO: Cache under algo_name and return.
        raise NotImplementedError("TODO: implement _get_voter")

    @staticmethod
    def _signal_to_direction(signal: AlgoSignal) -> str:
        """
        Normalise an AlgoSignal direction to a gate-level direction string.

        Parameters
        ----------
        signal : AlgoSignal
            Signal returned by an algo's generate() method.

        Returns
        -------
        str
            "bullish" if signal.direction == "long",
            "bearish" if signal.direction == "short".

        Raises
        ------
        ValueError
            If signal.direction is not "long" or "short".
        """
        # TODO: Map "long" → "bullish", "short" → "bearish".
        # TODO: Raise ValueError for any other value.
        raise NotImplementedError("TODO: implement _signal_to_direction")
