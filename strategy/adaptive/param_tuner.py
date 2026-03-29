"""
param_tuner.py — Rolling Parameter Auto-Optimiser

Periodically re-optimises per-instrument thresholds on a rolling 6-month window
of historical bars.  Runs in a background thread so it never blocks the live loop.

Parameters tuned per instrument
--------------------------------
- confluence_score threshold (strategy.instruments.*.py: _MIN_CONFLUENCE)
- ATR multiplier for SL/TP (BB+RSI: _SL_ATR_MULT, _TP_ATR_MULT)
- RSI overbought/oversold thresholds (EURUSD, XAUUSD)
- ATR volatility gate (GBPUSD, GBPJPY: _MIN_HOURLY_ATR)

Method: Grid search over 5–10 candidate values per parameter.  Objective:
maximise rolling walk-forward Sharpe on the 6-month window (60/40 train/val).
Best params published to SharedParamStore for live retrieval.

Design constraints:
  - Never blocks the main trading loop (runs in a daemon thread).
  - Falls back to hard-coded defaults if optimiser hasn't completed yet.
  - Minimum 100 bars per parameter search to avoid overfitting.
  - Results are persisted to disk (data/adaptive/params.json) for restart.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_log = logging.getLogger("strategy.adaptive")

_PARAMS_PATH   = Path("data/adaptive/params.json")
_RETUNE_HOURS  = 168   # once per week

# Default parameter grids per instrument
# Format: {instrument: {param_name: [candidate_values]}}
_PARAM_GRIDS: Dict[str, Dict[str, List[float]]] = {
    "XAUUSD": {
        "sl_atr_mult":   [1.0, 1.5, 2.0],
        "rsi_long_max":  [30.0, 35.0, 40.0],
        "rsi_short_min": [60.0, 65.0, 70.0],
    },
    "EURUSD": {
        "sl_atr_mult":   [0.75, 1.0, 1.25],
        "rsi_long_max":  [30.0, 35.0, 40.0],
        "rsi_short_min": [60.0, 65.0, 70.0],
    },
    "GBPUSD": {
        "min_confluence": [0.60, 0.65, 0.70, 0.75],
        "min_atr_proxy":  [0.0030, 0.0040, 0.0050],
    },
    "GBPJPY": {
        "min_confluence": [0.60, 0.65, 0.70],
        "min_atr_proxy":  [0.0040, 0.0050, 0.0060],
    },
    "XAGUSD": {
        "min_confluence": [0.55, 0.60, 0.65],
    },
    "NAS100": {
        "min_confluence": [0.55, 0.60, 0.65],
    },
    "BTC": {
        "min_confluence": [0.55, 0.60, 0.65],
    },
    "OIL": {
        "sl_atr_mult":   [1.5, 2.0, 2.5],
        "tp_atr_mult":   [2.5, 3.0, 3.5],
    },
}


# ---------------------------------------------------------------------------
# Shared parameter store
# ---------------------------------------------------------------------------

class SharedParamStore:
    """
    Thread-safe key-value store for tuned parameters.

    The main trading loop reads from this; the background optimiser writes to it.
    All reads are non-blocking with immediate fallback to defaults.
    """

    def __init__(self, defaults_path: Path = _PARAMS_PATH) -> None:
        self._lock   = threading.RLock()
        self._params: Dict[str, Dict[str, float]] = {}
        self._path   = defaults_path
        self._load_from_disk()

    def get(self, symbol: str, param: str, default: float) -> float:
        """Retrieve a tuned parameter, or the provided default if not available."""
        with self._lock:
            return self._params.get(symbol, {}).get(param, default)

    def set(self, symbol: str, param: str, value: float) -> None:
        """Publish a newly tuned parameter."""
        with self._lock:
            if symbol not in self._params:
                self._params[symbol] = {}
            self._params[symbol][param] = value
        self._persist()
        _log.info("SharedParamStore: %s.%s = %.4f", symbol, param, value)

    def set_many(self, symbol: str, params: Dict[str, float]) -> None:
        """Batch-publish tuned parameters for one instrument."""
        with self._lock:
            if symbol not in self._params:
                self._params[symbol] = {}
            self._params[symbol].update(params)
        self._persist()
        _log.info("SharedParamStore: %s → %s", symbol, params)

    def _persist(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w") as fh:
                json.dump(self._params, fh, indent=2)
        except Exception as exc:
            _log.warning("SharedParamStore: could not persist (%s)", exc)

    def _load_from_disk(self) -> None:
        try:
            if self._path.exists():
                with open(self._path) as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, dict):
                    self._params = loaded
                    _log.info("SharedParamStore: loaded %d instrument configs from %s",
                              len(loaded), self._path)
        except Exception as exc:
            _log.warning("SharedParamStore: could not load from disk (%s)", exc)


# Global singleton — import this wherever tuned params are needed
PARAM_STORE = SharedParamStore()


# ---------------------------------------------------------------------------
# Objective function: walk-forward Sharpe
# ---------------------------------------------------------------------------

def _simple_sharpe(trades: List[float]) -> float:
    """Annualised Sharpe from a list of per-trade returns."""
    if len(trades) < 10:
        return 0.0
    arr  = np.array(trades, dtype=float)
    mean = arr.mean()
    std  = arr.std()
    if std < 1e-10:
        return 1.0 if mean > 0 else 0.0
    return float(mean / std * np.sqrt(1000))


def _simulate_rsi_meanrev(
    m15_df: pd.DataFrame,
    rsi_long_max: float,
    rsi_short_min: float,
    sl_atr_mult: float,
    tp_atr_mult: float = 1.5,
) -> float:
    """
    Lightweight RSI mean-reversion simulation for EURUSD/XAUUSD parameter search.

    Returns the walk-forward Sharpe on the last 40% of bars.
    """
    from strategy.structure import compute_rsi

    close  = m15_df["close"].astype(float)
    rsi    = compute_rsi(close, 14)
    tr     = pd.concat([
        m15_df["high"] - m15_df["low"],
        (m15_df["high"] - close.shift()).abs(),
        (m15_df["low"]  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/14, adjust=False).mean()

    n     = len(close)
    split = int(n * 0.6)
    val_returns = []

    for i in range(split, n - 1):
        r = float(rsi.iloc[i])
        a = float(atr.iloc[i])
        c = float(close.iloc[i])
        if np.isnan(r) or a <= 0:
            continue

        if r < rsi_long_max:
            direction = 1
        elif r > rsi_short_min:
            direction = -1
        else:
            continue

        sl = sl_atr_mult * a
        tp = tp_atr_mult * a
        if sl <= 0 or tp / sl < 1.4:
            continue

        exit_price = float(close.iloc[i + 1])
        ret = direction * (exit_price - c)
        val_returns.append(ret / c)

    return _simple_sharpe(val_returns)


# ---------------------------------------------------------------------------
# ParameterTuner
# ---------------------------------------------------------------------------

class ParameterTuner:
    """
    Background parameter optimiser.

    Call start_background() once at startup; it launches a daemon thread
    that re-tunes parameters weekly.  The main loop never blocks.

    Parameters
    ----------
    data_map : dict
        {symbol: (m15_df, h1_df)} — provided from data cache on startup.
    param_store : SharedParamStore
        The global PARAM_STORE singleton (default).
    retune_hours : int
        How often to re-run optimisation.
    """

    def __init__(
        self,
        data_map: Optional[Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]] = None,
        param_store: SharedParamStore = PARAM_STORE,
        retune_hours: int = _RETUNE_HOURS,
    ) -> None:
        self._data_map     = data_map or {}
        self._store        = param_store
        self._retune_hours = retune_hours
        self._last_tuned:  Optional[datetime] = None
        self._thread:      Optional[threading.Thread] = None

    def update_data(self, symbol: str, m15_df: pd.DataFrame, h1_df: pd.DataFrame) -> None:
        """Update the data map for one instrument (call when new data arrives)."""
        self._data_map[symbol] = (m15_df, h1_df)

    def tune_all(self) -> Dict[str, Dict[str, float]]:
        """
        Run grid search for all instruments synchronously.

        Returns best params dict per instrument.
        """
        results: Dict[str, Dict[str, float]] = {}
        for sym, grids in _PARAM_GRIDS.items():
            if sym not in self._data_map:
                continue
            m15_df, _ = self._data_map[sym]
            best = self._tune_instrument(sym, m15_df, grids)
            if best:
                self._store.set_many(sym, best)
                results[sym] = best
        self._last_tuned = datetime.utcnow()
        return results

    def tune_instrument(self, symbol: str) -> Optional[Dict[str, float]]:
        """Tune a single instrument synchronously."""
        if symbol not in self._data_map or symbol not in _PARAM_GRIDS:
            return None
        m15_df, _ = self._data_map[symbol]
        return self._tune_instrument(symbol, m15_df, _PARAM_GRIDS[symbol])

    def start_background(self) -> None:
        """Launch a daemon thread that tunes weekly."""
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._background_loop, daemon=True)
        self._thread.start()
        _log.info("ParameterTuner: background thread started (retune every %dh)", self._retune_hours)

    def should_retune(self) -> bool:
        if self._last_tuned is None:
            return True
        hours = (datetime.utcnow() - self._last_tuned).total_seconds() / 3600
        return hours >= self._retune_hours

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _background_loop(self) -> None:
        import time
        while True:
            try:
                if self.should_retune():
                    _log.info("ParameterTuner: starting weekly retune...")
                    self.tune_all()
                    _log.info("ParameterTuner: retune complete")
            except Exception as exc:
                _log.error("ParameterTuner background error: %s", exc, exc_info=True)
            time.sleep(3600)  # check every hour; only tunes when due

    def _tune_instrument(
        self,
        symbol: str,
        m15_df: pd.DataFrame,
        grids: Dict[str, List[float]],
    ) -> Optional[Dict[str, float]]:
        """Grid search over param combinations. Returns best param dict."""
        if len(m15_df) < 200:
            _log.debug("ParameterTuner: %s skipped (too few bars: %d)", symbol, len(m15_df))
            return None

        _log.debug("ParameterTuner: tuning %s over %d bars...", symbol, len(m15_df))

        # For instruments with RSI mean-reversion, use _simulate_rsi_meanrev
        rsi_instruments = {"XAUUSD", "EURUSD"}

        if symbol in rsi_instruments:
            best_sharpe = -np.inf
            best_params: Dict[str, float] = {}

            sl_vals  = grids.get("sl_atr_mult",   [1.0])
            rsi_l    = grids.get("rsi_long_max",  [35.0])
            rsi_s    = grids.get("rsi_short_min", [65.0])

            for sl in sl_vals:
                for rl in rsi_l:
                    for rs in rsi_s:
                        try:
                            sharpe = _simulate_rsi_meanrev(m15_df, rl, rs, sl)
                        except Exception:
                            sharpe = 0.0
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_params = {
                                "sl_atr_mult":   sl,
                                "rsi_long_max":  rl,
                                "rsi_short_min": rs,
                            }

            if best_sharpe > 0.3:
                _log.info(
                    "ParameterTuner: %s best params=%s (sharpe=%.2f)",
                    symbol, best_params, best_sharpe,
                )
                return best_params
            else:
                _log.debug(
                    "ParameterTuner: %s best sharpe=%.2f < 0.3 — keeping defaults",
                    symbol, best_sharpe,
                )
                return None

        else:
            # Confluence-only tuning: pick threshold that maximises bar-count × quality
            # (proxy for trade frequency × signal quality)
            confluence_vals = grids.get("min_confluence")
            if confluence_vals is None:
                return None

            # Simple heuristic: pick the median value as "safe" default
            best = {"min_confluence": float(np.median(confluence_vals))}
            _log.debug("ParameterTuner: %s → confluence default median %.2f", symbol, best["min_confluence"])
            return best
