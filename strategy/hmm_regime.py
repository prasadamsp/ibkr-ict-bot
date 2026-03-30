"""
hmm_regime.py — Hidden Markov Model Regime Classifier

Drop-in replacement for the rule-based ``RegimeDetector`` / ``RegimeClassifier``
in ``strategy/regime.py``.  Instead of hand-crafted ADX/ATR thresholds, a
GaussianHMM with 4 latent states is trained on 3 years of daily OHLCV data per
instrument.  After fitting, each hidden state is post-hoc labelled as one of
the four ``MarketRegime`` values by inspecting the state's emission means
(e.g. sign of average log-return, relative ATR level, EMA slope direction).

Architecture overview
---------------------
1. ``_extract_features``  — builds a (T × 4) feature matrix from daily bars:
       col 0: log-return              (daily close-to-close)
       col 1: ATR%                    (ATR-14 / close, normalised by instrument)
       col 2: volume ratio            (volume / rolling-20-day mean volume)
       col 3: EMA slope (normalised)  ((EMA-20 today − EMA-20 yesterday) / close)

2. ``fit``   — scales features → trains GaussianHMM(n_components=4, covariance_type="full")
             → calls ``_map_state_to_regime`` to assign semantic labels per state.

3. ``predict`` — extracts features from the tail of a live daily DataFrame,
               runs ``hmm.predict()`` on the full sequence, reads the last
               predicted state, maps it to a ``MarketRegime``.

4. ``save`` / ``load`` — persists the trained HMM and the StandardScaler to
               ``<model_dir>/<symbol>_hmm.joblib`` and
               ``<model_dir>/<symbol>_scaler.joblib``.

State-to-regime mapping
-----------------------
HMM states are *arbitrary integers* (0–3); their semantic meaning must be
inferred from the emission distribution after training.  The mapping strategy:

  • Sort states by their mean log-return emission (ascending).
  • State with highest mean log-return  → TRENDING_BULL
  • State with lowest  mean log-return  → TRENDING_BEAR
  • Remaining two states: whichever has higher mean ATR% → VOLATILE
                          the other                       → RANGING

This heuristic may need per-instrument tuning.  Override ``_map_state_to_regime``
in a subclass if a specific instrument behaves differently (e.g. BTC has
structurally higher ATR% in all states).

Weekly retraining
-----------------
A scheduler (outside this module) should call ``fit()`` each Monday pre-market
using the latest 3-year daily bars, then call ``save(symbol)`` to persist the
updated model.  ``predict()`` always uses the currently loaded model in memory.

Usage
-----
    from strategy.hmm_regime import HMMRegimeClassifier, MarketRegime

    clf = HMMRegimeClassifier()
    clf.fit(daily_df, symbol="EURUSD")
    clf.save("EURUSD")

    # Later / on-bar:
    clf.load("EURUSD")
    regime = clf.predict(daily_df)  # → MarketRegime.TRENDING_BULL, etc.

Dependencies
------------
    pip install hmmlearn scikit-learn joblib numpy pandas
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional imports — wrapped so the module can be imported even if hmmlearn is
# not yet installed (unit tests can monkey-patch as needed).
# ---------------------------------------------------------------------------

try:
    from hmmlearn.hmm import GaussianHMM  # type: ignore
except ImportError:  # pragma: no cover
    GaussianHMM = None  # type: ignore

try:
    from sklearn.preprocessing import StandardScaler  # type: ignore
except ImportError:  # pragma: no cover
    StandardScaler = None  # type: ignore

try:
    import joblib  # type: ignore
except ImportError:  # pragma: no cover
    joblib = None  # type: ignore


# ---------------------------------------------------------------------------
# Public enum — the four market regimes recognised by this classifier
# ---------------------------------------------------------------------------


class MarketRegime(Enum):
    """
    Four mutually exclusive market regimes emitted by ``HMMRegimeClassifier``.

    Consumers should treat ``VOLATILE`` as a "no-new-entries" signal and may
    optionally treat ``RANGING`` as a mean-reversion opportunity.
    """

    TRENDING_BULL = "trending_bull"  # Upward trend with sufficient momentum
    TRENDING_BEAR = "trending_bear"  # Downward trend with sufficient momentum
    RANGING       = "ranging"        # Low directional momentum / consolidation
    VOLATILE      = "volatile"       # Elevated ATR — chaotic, avoid new entries


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class HMMRegimeClassifier:
    """
    GaussianHMM-based market regime classifier.

    Designed as a drop-in replacement for ``strategy.regime.RegimeDetector``.
    One instance per instrument is recommended so that each model can be
    trained and persisted independently.

    Parameters
    ----------
    n_states : int
        Number of hidden states.  Should remain 4 to match ``MarketRegime``.
        Exposed as a parameter to allow experimentation.
    model_dir : str
        Directory where trained model artefacts are saved/loaded.
        Created automatically on first ``save()`` call.
    n_iter : int
        Maximum EM iterations for hmmlearn training (default 200).
        Increase if convergence warnings appear.
    random_state : int
        Seed for reproducibility of HMM initialisation.
    """

    def __init__(
        self,
        n_states: int = 4,
        model_dir: str = "data/adaptive/hmm",
        n_iter: int = 200,
        random_state: int = 42,
    ) -> None:
        self.n_states      = n_states
        self.model_dir     = model_dir
        self.n_iter        = n_iter
        self.random_state  = random_state

        # Set after ``fit()`` — None until trained
        self._hmm: Optional[GaussianHMM]        = None
        self._scaler: Optional[StandardScaler]  = None

        # Maps integer HMM state → MarketRegime; populated by ``fit()``
        self._state_map: Dict[int, MarketRegime] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, daily_df: pd.DataFrame, symbol: str) -> None:
        """
        Train the GaussianHMM on historical daily OHLCV data.

        Call this once at startup and then weekly (Monday pre-market) to
        keep the model current.  After training, ``_state_map`` is populated
        via ``_map_state_to_regime`` so that ``predict()`` can return a
        semantic label immediately.

        Parameters
        ----------
        daily_df : pd.DataFrame
            3 years of daily OHLCV bars.  Must contain columns:
            ``open``, ``high``, ``low``, ``close``, ``volume``.
            Rows must be sorted ascending by date with no gaps.
            Recommended minimum: 500 rows (~2 years) for stable EM convergence.
        symbol : str
            Instrument identifier (e.g. ``"EURUSD"``, ``"NAS100"``).
            Used only for logging / error messages here; ``save()`` uses it
            as the filename stem.

        Raises
        ------
        ValueError
            If ``daily_df`` has fewer than 100 rows or is missing required columns.
        RuntimeError
            If ``hmmlearn`` is not installed.

        Implementation notes
        --------------------
        TODO:
          1. Validate ``daily_df`` columns and minimum length.
          2. Call ``_extract_features(daily_df)`` → shape (T, 4).
          3. Drop leading NaN rows that arise from rolling windows in feature
             extraction (ATR-14 introduces 13 NaN rows at the start).
          4. Fit a ``StandardScaler`` on the feature matrix; store as
             ``self._scaler``.
          5. Create ``GaussianHMM(n_components=self.n_states,
             covariance_type="full", n_iter=self.n_iter,
             random_state=self.random_state)``.
          6. Call ``hmm.fit(scaled_features)`` — hmmlearn expects shape (T, n_features).
          7. Store trained model as ``self._hmm``.
          8. Call ``self._map_state_to_regime(symbol)`` to populate
             ``self._state_map``.
          9. Log training summary: symbol, n_rows used, convergence flag
             (``hmm.monitor_.converged``), log-likelihood
             (``hmm.score(scaled_features)``).
        """
        raise NotImplementedError(
            "TODO: implement HMMRegimeClassifier.fit() — see docstring for step-by-step guide"
        )

    def predict(self, daily_df: pd.DataFrame) -> MarketRegime:
        """
        Predict the current market regime from recent daily bars.

        Runs the full Viterbi decode on the provided DataFrame and returns the
        regime label for the *last* bar.  Typically called once per day
        (end-of-session) or at strategy start-up after loading a saved model.

        Parameters
        ----------
        daily_df : pd.DataFrame
            Recent daily OHLCV bars (same format as ``fit()``).
            At least 30 rows recommended so that rolling features (ATR-14,
            EMA-20) are stable by the final bar.

        Returns
        -------
        MarketRegime
            Regime label for the final bar in ``daily_df``.

        Raises
        ------
        RuntimeError
            If ``fit()`` or ``load()`` has not been called before ``predict()``.

        Implementation notes
        --------------------
        TODO:
          1. Guard: raise RuntimeError if ``self._hmm`` is None.
          2. Call ``_extract_features(daily_df)`` → shape (T, 4).
          3. Drop leading NaN rows (same logic as in ``fit()``).
          4. Scale features using ``self._scaler.transform(features)``.
          5. Run ``self._hmm.predict(scaled_features)`` → array of state ints.
          6. Take the last element: ``predicted_state = states[-1]``.
          7. Return ``self._state_map[predicted_state]``.
          8. Edge case: if ``_state_map`` is empty (model loaded from disk but
             ``_map_state_to_regime`` not re-run), re-derive the mapping from
             the loaded HMM's emission means.
        """
        raise NotImplementedError(
            "TODO: implement HMMRegimeClassifier.predict() — see docstring for step-by-step guide"
        )

    def save(self, symbol: str) -> None:
        """
        Persist the trained HMM and scaler to disk using ``joblib``.

        Saves two files under ``self.model_dir``:
          - ``<symbol>_hmm.joblib``    — the trained GaussianHMM object
          - ``<symbol>_scaler.joblib`` — the fitted StandardScaler object
          - ``<symbol>_statemap.joblib`` — the ``_state_map`` dict

        Parameters
        ----------
        symbol : str
            Instrument identifier used as the filename stem.

        Implementation notes
        --------------------
        TODO:
          1. Guard: raise RuntimeError if ``self._hmm`` is None.
          2. Create ``self.model_dir`` if it does not exist
             (``Path(self.model_dir).mkdir(parents=True, exist_ok=True)``).
          3. Build paths:
               hmm_path    = Path(self.model_dir) / f"{symbol}_hmm.joblib"
               scaler_path = Path(self.model_dir) / f"{symbol}_scaler.joblib"
               map_path    = Path(self.model_dir) / f"{symbol}_statemap.joblib"
          4. Call ``joblib.dump(self._hmm, hmm_path)``
             and  ``joblib.dump(self._scaler, scaler_path)``
             and  ``joblib.dump(self._state_map, map_path)``.
          5. Log saved paths and file sizes for observability.
        """
        raise NotImplementedError(
            "TODO: implement HMMRegimeClassifier.save() — see docstring for step-by-step guide"
        )

    def load(self, symbol: str) -> bool:
        """
        Load a previously saved HMM model from disk.

        Parameters
        ----------
        symbol : str
            Instrument identifier; must match the stem used in ``save()``.

        Returns
        -------
        bool
            ``True`` if all artefacts were found and loaded successfully.
            ``False`` if the model files do not exist yet (first run).

        Implementation notes
        --------------------
        TODO:
          1. Build the three expected file paths (same as in ``save()``).
          2. If any path does not exist, return False immediately.
          3. Load all three artefacts with ``joblib.load()``.
          4. Assign ``self._hmm``, ``self._scaler``, ``self._state_map``.
          5. Return True.
          6. Wrap in try/except: if loading fails for any reason, log a WARNING
             and return False so the caller can fall back to ``fit()``.
        """
        raise NotImplementedError(
            "TODO: implement HMMRegimeClassifier.load() — see docstring for step-by-step guide"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_features(self, daily_df: pd.DataFrame) -> np.ndarray:
        """
        Build the (T × 4) feature matrix used for HMM training and inference.

        Feature definitions
        -------------------
        col 0 — log_return
            np.log(close[t] / close[t-1])
            Captures daily directional momentum.
            First row will be NaN (no previous close).

        col 1 — atr_pct
            ATR(14) / close[t]
            Normalises volatility by price level so that different instruments
            (e.g. BTC at ~60 000 vs EURUSD at ~1.08) are comparable.
            First 13 rows will be NaN while the ATR warm-up period fills.

        col 2 — volume_ratio
            volume[t] / rolling_mean(volume, 20)[t]
            Values > 1 indicate above-average participation (breakout signal);
            < 1 indicates quiet/consolidating sessions.
            First 19 rows will be NaN.

        col 3 — ema_slope_norm
            (EMA(20)[t] - EMA(20)[t-1]) / close[t]
            Normalised EMA gradient; positive = upward momentum, negative = down.
            First row will be NaN.

        Implementation notes
        --------------------
        TODO:
          1. Extract ``close``, ``high``, ``low``, ``volume`` as numpy arrays.
          2. Compute log_return: ``np.log(close[1:] / close[:-1])``, prepend NaN.
          3. Compute ATR(14) — reuse ``strategy.regime._compute_atr`` or
             re-implement inline:
               TR = max(H-L, |H-Cprev|, |L-Cprev|)
               ATR = Wilder average of TR over 14 bars.
             Then divide ATR by close to get ``atr_pct``.
          4. Compute volume_ratio using a pandas rolling mean (window=20, min_periods=1):
               vol_ma = pd.Series(volume).rolling(20, min_periods=1).mean()
               volume_ratio = volume / vol_ma.values
             Use min_periods=1 so NaN only appears when volume itself is NaN.
          5. Compute EMA(20) via ``pd.Series(close).ewm(span=20, adjust=False).mean()``.
             Then slope = (ema[1:] - ema[:-1]) / close[1:], prepend NaN.
          6. Stack into a (T, 4) float64 array:
               features = np.column_stack([log_return, atr_pct, volume_ratio, ema_slope_norm])
          7. Return the full array including NaN rows — callers are responsible
             for dropping leading NaNs before feeding to the HMM.
             (Keeping them here makes testing the shape easier.)

        Parameters
        ----------
        daily_df : pd.DataFrame
            Daily OHLCV bars with columns ``open``, ``high``, ``low``,
            ``close``, ``volume`` and ascending date index.

        Returns
        -------
        np.ndarray
            Shape (len(daily_df), 4).  Leading rows may contain NaN.
        """
        raise NotImplementedError(
            "TODO: implement HMMRegimeClassifier._extract_features() — see docstring for step-by-step guide"
        )

    def _map_state_to_regime(self, symbol: str) -> None:
        """
        Assign a ``MarketRegime`` label to each integer HMM state.

        Must be called *after* ``fit()`` so that ``self._hmm.means_`` is
        available.  Populates ``self._state_map``.

        Mapping heuristic
        -----------------
        ``self._hmm.means_`` has shape (n_states, n_features).
        Feature column indices (as defined in ``_extract_features``):
            0 → log_return
            1 → atr_pct
            2 → volume_ratio
            3 → ema_slope_norm

        Step-by-step algorithm:
          1. Extract mean log-return per state: ``means[:, 0]``.
          2. Sort states by mean log-return ascending.
          3. Assign:
               - highest mean log-return → TRENDING_BULL
               - lowest  mean log-return → TRENDING_BEAR
          4. From the two remaining states, compare mean atr_pct (``means[:, 1]``):
               - higher atr_pct  → VOLATILE
               - lower  atr_pct  → RANGING
          5. Build ``self._state_map = {state_int: MarketRegime, ...}``.
          6. Log the mapping for inspection (symbol + state index + regime + means).

        Important caveats
        -----------------
        - This is a heuristic; for some instruments the "highest return" state
          may actually be VOLATILE (e.g. during a crash the market has large
          negative log-returns AND high ATR).  If post-training evaluation shows
          poor labelling, consider a more robust clustering approach
          (e.g. k-means on emission means, or manual inspection of sample dates).
        - States are not guaranteed to be consistent across retraining runs
          (label switching problem in HMMs).  The heuristic re-derives the
          mapping from emission means every time, so retraining is safe.

        Parameters
        ----------
        symbol : str
            Used only for logging to identify which instrument's mapping is shown.

        Implementation notes
        --------------------
        TODO:
          1. Assert ``self._hmm is not None``.
          2. Read ``means = self._hmm.means_``  → shape (n_states, 4).
          3. Implement the 4-step algorithm above.
          4. Set ``self._state_map``.
          5. Log the result as a human-readable table.
        """
        raise NotImplementedError(
            "TODO: implement HMMRegimeClassifier._map_state_to_regime() — see docstring for step-by-step guide"
        )
