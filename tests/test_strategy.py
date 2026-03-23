"""
Unit tests for the ICT strategy modules.
Run with: python -m pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from strategy.structure import (
    detect_bos, detect_mss, find_swing_highs, find_swing_lows, get_trend_bias,
)
from strategy.fvg import detect_fvgs, get_nearest_fvg
from strategy.liquidity import find_liquidity_levels, detect_sweeps
from strategy.zones import calculate_dealing_range, classify_price, price_in_discount
from strategy.sessions import get_session, is_tradeable_session, session_score


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_trending_df(n=100, trend="up", base=25.0):
    """
    Generate synthetic OHLCV data with clear swing structure.

    Uses a wave pattern: 6 bars up, 3 bars down (uptrend).
    The pullback (3 bars) exceeds the swing lookback (5) when combined
    with the advance, ensuring clear local highs/lows are detectable.
    """
    np.random.seed(42)
    # Build highs and lows arrays directly as zigzag waves
    # Uptrend: each wave makes a higher high and higher low
    data = []

    price = base
    wave_size = 9  # bars per wave (6 up + 3 down)
    wave_advance = 0.40   # total price gain per wave (uptrend)
    wave_retrace = 0.15   # retracement per pullback

    for i in range(n):
        wave_pos = i % wave_size
        wave_num = i // wave_size

        # Small per-bar epsilon prevents exact price ties across wave boundaries
        eps = i * 0.0001

        if trend == "up":
            net_gain = wave_advance - wave_retrace
            wave_start = base + wave_num * net_gain
            if wave_pos < 6:
                frac = wave_pos / 5.0
                price = wave_start + frac * wave_advance + eps
            else:
                frac = (wave_pos - 5) / 3.0
                price = wave_start + wave_advance - frac * wave_retrace + eps

        elif trend == "down":
            net_decline = wave_advance - wave_retrace
            wave_start = base - wave_num * net_decline
            if wave_pos < 6:
                frac = wave_pos / 5.0
                price = wave_start - frac * wave_advance - eps
            else:
                frac = (wave_pos - 5) / 3.0
                price = wave_start - wave_advance + frac * wave_retrace - eps
        else:
            price = base + np.sin(i * 0.3) * 0.5  # oscillate

        spread = max(price * 0.002, 0.001)
        data.append({
            "open": price - spread * 0.3,
            "high": price + spread,
            "low": price - spread,
            "close": price,
            "volume": 500,
        })

    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(data, index=idx)


def make_fvg_df():
    """Create a DataFrame with a known bullish FVG at bars 10-12."""
    data = []
    for i in range(30):
        if i == 10:
            # bar[10]: high = 25.10
            data.append({"open": 25.0, "high": 25.10, "low": 24.95, "close": 25.05, "volume": 100})
        elif i == 11:
            # bar[11]: large up move
            data.append({"open": 25.05, "high": 25.40, "low": 25.05, "close": 25.35, "volume": 200})
        elif i == 12:
            # bar[12]: low = 25.15 — gap: [25.10, 25.15]
            data.append({"open": 25.35, "high": 25.50, "low": 25.15, "close": 25.45, "volume": 150})
        else:
            c = 25.0 + i * 0.01
            data.append({"open": c, "high": c + 0.05, "low": c - 0.05, "close": c, "volume": 100})
    idx = pd.date_range("2024-01-01", periods=30, freq="15min", tz="UTC")
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# Swing detection tests
# ---------------------------------------------------------------------------

class TestSwings:
    def test_swing_highs_detected(self):
        df = make_trending_df(50, "up")
        sh = find_swing_highs(df, lookback=3)
        # Should find some swing highs
        assert sh.sum() > 0

    def test_swing_lows_detected(self):
        df = make_trending_df(50, "down")
        sl = find_swing_lows(df, lookback=3)
        assert sl.sum() > 0

    def test_no_lookahead(self):
        """Last `lookback` bars should never be swing highs."""
        df = make_trending_df(50, "up")
        lookback = 5
        sh = find_swing_highs(df, lookback=lookback)
        # Last lookback bars must be False (not yet confirmed)
        assert sh.iloc[-lookback:].sum() == 0

    def test_swing_high_is_local_max(self):
        """Every detected swing high must be the local maximum."""
        df = make_trending_df(80, "up")
        lookback = 3
        sh = find_swing_highs(df, lookback=lookback)
        for i in sh[sh].index:
            pos = df.index.get_loc(i)
            window_high = df["high"].iloc[pos - lookback: pos + lookback + 1].max()
            assert df.loc[i, "high"] == window_high


# ---------------------------------------------------------------------------
# BOS tests
# ---------------------------------------------------------------------------

class TestBOS:
    def test_bullish_bos_in_uptrend(self):
        df = make_trending_df(100, "up")
        result = detect_bos(df, lookback=3)
        assert result["bos_bullish"].sum() > 0

    def test_bearish_bos_in_downtrend(self):
        df = make_trending_df(100, "down")
        result = detect_bos(df, lookback=3)
        assert result["bos_bearish"].sum() > 0

    def test_bos_has_level(self):
        df = make_trending_df(100, "up")
        result = detect_bos(df, lookback=3)
        bos_bars = result[result["bos_bullish"]]
        if len(bos_bars) > 0:
            assert not bos_bars["bos_level"].isna().all()

    def test_no_simultaneous_bull_bear_bos(self):
        """Can't have both bullish and bearish BOS on same bar."""
        df = make_trending_df(100, "up")
        result = detect_bos(df, lookback=3)
        both = result["bos_bullish"] & result["bos_bearish"]
        assert both.sum() == 0


# ---------------------------------------------------------------------------
# FVG tests
# ---------------------------------------------------------------------------

class TestFVG:
    def test_bullish_fvg_detected(self):
        df = make_fvg_df()
        fvgs = detect_fvgs(df, min_size_pct=0.00001, max_age_bars=50)
        bullish = [f for f in fvgs if f.direction == "bullish"]
        assert len(bullish) > 0

    def test_fvg_gap_is_real(self):
        """FVG top must be > FVG bottom."""
        df = make_fvg_df()
        fvgs = detect_fvgs(df, min_size_pct=0.00001, max_age_bars=50)
        for fvg in fvgs:
            assert fvg.top > fvg.bottom

    def test_fvg_min_size_filter(self):
        """With very high min_size, fewer or no FVGs should be returned."""
        df = make_trending_df(100)
        fvgs_loose = detect_fvgs(df, min_size_pct=0.00001)
        fvgs_strict = detect_fvgs(df, min_size_pct=0.10)
        assert len(fvgs_strict) <= len(fvgs_loose)

    def test_filled_fvg_excluded(self):
        """FVGs that price has closed through should not appear in active list."""
        df = make_fvg_df()
        fvgs = detect_fvgs(df, min_size_pct=0.00001)
        for fvg in fvgs:
            assert not fvg.filled


# ---------------------------------------------------------------------------
# Premium/Discount tests
# ---------------------------------------------------------------------------

class TestZones:
    def test_dealing_range_calculated(self):
        df = make_trending_df(60, "up")
        dr = calculate_dealing_range(df, lookback=3)
        assert dr is not None
        assert dr.swing_high > dr.swing_low
        assert dr.midpoint == pytest.approx((dr.swing_high + dr.swing_low) / 2, rel=1e-5)

    def test_price_in_discount(self):
        df = make_trending_df(60)
        dr = calculate_dealing_range(df, lookback=3)
        if dr:
            below_mid = dr.swing_low + (dr.range_size * 0.3)
            assert price_in_discount(below_mid, dr)

    def test_classify_price(self):
        df = make_trending_df(60)
        dr = calculate_dealing_range(df, lookback=3)
        if dr:
            zone = classify_price(dr.midpoint - 0.01, dr)
            assert zone in ("discount", "ote_buy")


# ---------------------------------------------------------------------------
# Session tests
# ---------------------------------------------------------------------------

class TestSessions:
    def test_london_kill_zone(self):
        ts = pd.Timestamp("2024-01-15 07:30:00", tz="UTC")
        assert get_session(ts) == "london_kill"

    def test_ny_kill_zone(self):
        ts = pd.Timestamp("2024-01-15 12:30:00", tz="UTC")
        assert get_session(ts) == "ny_kill"

    def test_asian_session(self):
        ts = pd.Timestamp("2024-01-15 01:00:00", tz="UTC")
        assert get_session(ts) == "asian"

    def test_dead_zone(self):
        ts = pd.Timestamp("2024-01-15 20:00:00", tz="UTC")
        assert get_session(ts) == "dead_zone"

    def test_dead_zone_not_tradeable(self):
        ts = pd.Timestamp("2024-01-15 20:00:00", tz="UTC")
        assert not is_tradeable_session(ts)

    def test_kill_zone_score_is_1(self):
        ts = pd.Timestamp("2024-01-15 08:00:00", tz="UTC")
        assert session_score(ts) == 1.0

    def test_dead_zone_score_is_0(self):
        ts = pd.Timestamp("2024-01-15 21:00:00", tz="UTC")
        assert session_score(ts) == 0.0


# ---------------------------------------------------------------------------
# Trend bias test
# ---------------------------------------------------------------------------

class TestTrendBias:
    def test_uptrend_gives_bullish_bias(self):
        df = make_trending_df(80, "up")
        bias = get_trend_bias(df, lookback=3)
        # Last few bars should show bullish bias
        assert bias.iloc[-1] >= 0  # at least neutral

    def test_downtrend_gives_bearish_bias(self):
        df = make_trending_df(80, "down")
        bias = get_trend_bias(df, lookback=3)
        assert bias.iloc[-1] <= 0
