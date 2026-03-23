"""
regime.py — Market Regime Detector

Classifies the current market regime per bar into:
  - Trend direction: "up" / "down" / "sideways"
  - Volatility level: "high" / "normal" / "low"

Uses:
  - ADX(14) with Wilder's smoothing (manual numpy implementation, no ta-lib)
  - EMA(50) vs EMA(200) for trend direction
  - ATR(14) as a percentage of price for volatility classification

Wilder smoothing notes
----------------------
Welles Wilder uses two distinct smoothing operations in ADX construction:

1. _wilder_sum  — for True Range, +DM, -DM
   Seed  = simple SUM of first `period` values
   Update: S[i] = S[i-1] - S[i-1]/period + val[i]
   Result is a *smoothed running sum* (not an average).  The period factor
   cancels when computing +DI = smoothed(+DM) / smoothed(TR) * 100.

2. _wilder_avg  — for the final DX → ADX step
   Seed  = simple MEAN of first `period` *valid* DX values
   Update: A[i] = (A[i-1] * (period-1) + dx[i]) / period
   Result is a *smoothed running average* bounded in [0, 100].

Using _wilder_sum for ADX would produce values far above 100; using
_wilder_avg for TR/DM would cause the period factor not to cancel in the
DI ratio, giving wrong DI values.

Usage:
    from strategy.regime import RegimeDetector

    detector = RegimeDetector()
    state = detector.detect(df)
    print(state.trend, state.volatility, state.adx)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class RegimeState:
    """Holds the regime classification for a single bar / snapshot."""

    trend: str          # "up" | "down" | "sideways"
    volatility: str     # "high" | "normal" | "low"
    adx: float          # Average Directional Index value (0–100)
    atr: float          # ATR value in price units
    atr_pct: float      # ATR as a fraction of closing price (e.g. 0.012 = 1.2%)

    def __repr__(self) -> str:
        return (
            f"RegimeState(trend={self.trend!r}, volatility={self.volatility!r}, "
            f"adx={self.adx:.2f}, atr={self.atr:.5f}, atr_pct={self.atr_pct:.4f})"
        )


# ---------------------------------------------------------------------------
# Helpers — pure numpy, no ta-lib
# ---------------------------------------------------------------------------

def _wilder_sum(series: np.ndarray, period: int) -> np.ndarray:
    """
    Wilder's smoothed running SUM.  Used for True Range, +DM, -DM.

    Seed  = sum(series[0 : period])
    Update: result[i] = result[i-1] - result[i-1] / period + series[i]

    The result is a smoothed sum (NOT an average). The `period` factor
    cancels when DI is calculated as smoothed(+DM) / smoothed(TR).

    Parameters
    ----------
    series : np.ndarray  1-D, no NaNs expected (DM/TR arrays start from 0)
    period : int

    Returns
    -------
    np.ndarray  same length; values before index (period-1) are NaN
    """
    result = np.full(len(series), np.nan)
    if len(series) < period:
        return result

    # Seed: sum of first `period` values (not mean)
    result[period - 1] = np.sum(series[:period])

    for i in range(period, len(series)):
        result[i] = result[i - 1] - result[i - 1] / period + series[i]

    return result


def _wilder_avg(series: np.ndarray, period: int) -> np.ndarray:
    """
    Wilder's smoothed running AVERAGE.  Used for DX → ADX.

    Skips leading NaNs (DX has NaN for the first ~2*period-2 bars because
    it is derived from two _wilder_sum series).

    Seed  = mean of first `period` valid values
    Update: result[i] = (result[i-1] * (period-1) + series[i]) / period

    The result stays in [0, 100] when applied to DX values.

    Parameters
    ----------
    series : np.ndarray  1-D float; may contain leading NaNs
    period : int

    Returns
    -------
    np.ndarray  same length; values before seed position are NaN
    """
    result = np.full(len(series), np.nan)

    # Locate first non-NaN value
    valid_idx = np.where(~np.isnan(series))[0]
    if len(valid_idx) < period:
        return result

    first_valid = valid_idx[0]
    seed_end    = first_valid + period  # exclusive upper bound
    if seed_end > len(series):
        return result

    seed_slice = series[first_valid:seed_end]
    if np.any(np.isnan(seed_slice)):
        return result  # still have NaN gaps — cannot seed cleanly

    seed_idx = seed_end - 1
    result[seed_idx] = np.mean(seed_slice)

    for i in range(seed_idx + 1, len(series)):
        val = series[i]
        if np.isnan(val):
            result[i] = np.nan
        else:
            result[i] = (result[i - 1] * (period - 1) + val) / period

    return result


def _ema(series: pd.Series, span: int) -> pd.Series:
    """
    Exponential Moving Average using pandas ewm (adjust=False).

    Parameters
    ----------
    series : pd.Series
    span   : int  — the EMA period (e.g. 50, 200)

    Returns
    -------
    pd.Series aligned to the input index
    """
    return series.ewm(span=span, adjust=False).mean()


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 period: int = 14) -> np.ndarray:
    """
    Compute ATR(period) using Wilder's smoothed running average.

    True Range = max(H-L, |H-Cprev|, |L-Cprev|)

    ATR is the Wilder average of TR (seed = mean of first `period` TR values,
    then updated as a running average).  This keeps ATR in price units.

    Returns
    -------
    np.ndarray  ATR values; first (period-1) values are NaN
    """
    n = len(close)
    tr = np.full(n, np.nan)

    # First TR has no previous close — use H-L only
    tr[0] = high[0] - low[0]

    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    # Use Wilder average (not sum) for ATR so the result stays in price units
    atr = _wilder_avg(tr, period)
    return atr


def _compute_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 period: int = 14) -> np.ndarray:
    """
    Compute ADX(period) using Wilder's original methodology.

    Steps:
      1. Compute +DM, -DM, TR for each bar
      2. Wilder-SUM smooth +DM, -DM, TR  (seed = simple sum)
      3. +DI14 = 100 * SmoothedSum(+DM) / SmoothedSum(TR)
         -DI14 = 100 * SmoothedSum(-DM) / SmoothedSum(TR)
         (The `period` factor in numerator and denominator cancels.)
      4. DX = 100 * |+DI - -DI| / (+DI + -DI)
      5. ADX = Wilder-AVG smooth of DX  (seed = simple mean, stays in [0,100])

    Returns
    -------
    np.ndarray  ADX values in [0, 100]; first ~(2*period - 1) values are NaN
    """
    n = len(close)

    plus_dm  = np.zeros(n)
    minus_dm = np.zeros(n)
    tr       = np.zeros(n)

    # First bar: no previous data — TR = H-L, DMs = 0
    tr[0] = high[0] - low[0]

    for i in range(1, n):
        up_move   = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]

        # +DM: upward move is the dominant move and positive
        plus_dm[i]  = up_move   if (up_move > down_move and up_move > 0)   else 0.0
        # -DM: downward move is the dominant move and positive
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0

        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i]  - close[i - 1])
        tr[i] = max(hl, hc, lc)

    # Step 2: Wilder SUM smoothing (period cancels in DI ratio)
    sm_plus_dm  = _wilder_sum(plus_dm,  period)
    sm_minus_dm = _wilder_sum(minus_dm, period)
    sm_tr       = _wilder_sum(tr,       period)

    # Step 3: Directional Indicators
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di  = np.where(sm_tr != 0, 100.0 * sm_plus_dm  / sm_tr, 0.0)
        minus_di = np.where(sm_tr != 0, 100.0 * sm_minus_dm / sm_tr, 0.0)

    # Step 4: DX — mask out the NaN region from the smoothed sums
    nan_mask = np.isnan(sm_tr)
    di_sum   = plus_di + minus_di
    di_diff  = np.abs(plus_di - minus_di)
    with np.errstate(divide="ignore", invalid="ignore"):
        dx = np.where(
            (~nan_mask) & (di_sum != 0),
            100.0 * di_diff / di_sum,
            np.nan,
        )

    # Step 5: ADX = Wilder average of DX (keeps result in [0, 100])
    adx = _wilder_avg(dx, period)
    return adx


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RegimeDetector:
    """
    Classifies market regime from OHLCV data.

    Parameters
    ----------
    adx_period   : int   ADX / ATR smoothing period (default 14)
    ema_fast     : int   fast EMA period for trend direction (default 50)
    ema_slow     : int   slow EMA period for trend direction (default 200)
    adx_trend    : float ADX threshold above which market is trending (default 25)
    adx_range    : float ADX threshold below which market is ranging (default 20)
    atr_high_pct : float ATR/close above this → "high" volatility (default 0.015 = 1.5%)
    atr_low_pct  : float ATR/close below this → "low"  volatility (default 0.005 = 0.5%)
    """

    def __init__(
        self,
        adx_period:   int   = 14,
        ema_fast:     int   = 50,
        ema_slow:     int   = 200,
        adx_trend:    float = 25.0,
        adx_range:    float = 20.0,
        atr_high_pct: float = 0.015,
        atr_low_pct:  float = 0.005,
    ) -> None:
        self.adx_period   = adx_period
        self.ema_fast     = ema_fast
        self.ema_slow     = ema_slow
        self.adx_trend    = adx_trend
        self.adx_range    = adx_range
        self.atr_high_pct = atr_high_pct
        self.atr_low_pct  = atr_low_pct

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, df: pd.DataFrame, min_bars: int = 50) -> RegimeState:
        """
        Classify the regime as of the last bar in `df`.

        Parameters
        ----------
        df       : pd.DataFrame  OHLCV data with columns open/high/low/close/volume
                                  and a datetime index (ascending).
        min_bars : int            Minimum rows required; raises ValueError otherwise.

        Returns
        -------
        RegimeState
        """
        self._validate(df, min_bars)

        high  = df["high"].to_numpy(dtype=float)
        low   = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)

        # --- ADX (trend strength) ---
        adx_series = _compute_adx(high, low, close, self.adx_period)
        adx_value  = float(adx_series[-1]) if not np.isnan(adx_series[-1]) else 0.0

        # --- ATR (volatility) ---
        atr_series = _compute_atr(high, low, close, self.adx_period)
        atr_value  = float(atr_series[-1]) if not np.isnan(atr_series[-1]) else 0.0
        last_close = close[-1]
        atr_pct    = atr_value / last_close if last_close != 0 else 0.0

        # --- EMA crossover (trend direction) ---
        close_series  = df["close"].astype(float)
        ema_fast_val  = float(_ema(close_series, self.ema_fast).iloc[-1])
        ema_slow_val  = float(_ema(close_series, self.ema_slow).iloc[-1])

        # Classify trend direction based on ADX strength + EMA cross
        if adx_value > self.adx_trend:
            # Market is trending — use EMA cross to determine direction
            trend = "up" if ema_fast_val > ema_slow_val else "down"
        else:
            # ADX too weak to declare a clear trend
            trend = "sideways"

        # Classify volatility using ATR as percentage of price
        if atr_pct > self.atr_high_pct:
            volatility = "high"
        elif atr_pct < self.atr_low_pct:
            volatility = "low"
        else:
            volatility = "normal"

        return RegimeState(
            trend=trend,
            volatility=volatility,
            adx=round(adx_value, 4),
            atr=round(atr_value, 6),
            atr_pct=round(atr_pct, 6),
        )

    def is_trending(self, df: pd.DataFrame) -> bool:
        """
        Return True when ADX(14) > adx_trend threshold (default 25).

        This does NOT imply direction — just that a trend exists.
        """
        self._validate(df, min_bars=self.adx_period * 3)
        high  = df["high"].to_numpy(dtype=float)
        low   = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)
        adx_series = _compute_adx(high, low, close, self.adx_period)
        last_adx   = adx_series[-1]
        return bool(not np.isnan(last_adx) and last_adx > self.adx_trend)

    def is_ranging(self, df: pd.DataFrame) -> bool:
        """
        Return True when ADX(14) < adx_range threshold (default 20).

        Indicates a sideways / consolidating market.
        """
        self._validate(df, min_bars=self.adx_period * 3)
        high  = df["high"].to_numpy(dtype=float)
        low   = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)
        adx_series = _compute_adx(high, low, close, self.adx_period)
        last_adx   = adx_series[-1]
        return bool(not np.isnan(last_adx) and last_adx < self.adx_range)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(df: pd.DataFrame, min_bars: int) -> None:
        """Raise ValueError if the DataFrame doesn't meet minimum requirements."""
        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(df.columns.str.lower())
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")
        if len(df) < min_bars:
            raise ValueError(
                f"DataFrame has {len(df)} rows; need at least {min_bars}."
            )
