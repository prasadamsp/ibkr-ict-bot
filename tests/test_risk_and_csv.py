"""
Tests for RiskManager position sizing and CSV data loader.
Run with: python -m pytest tests/ -v
"""

import io
import textwrap
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from config.settings import CONFIG, RiskConfig, StrategyConfig
from data.csv_loader import load_csv, load_pair
from risk.risk_manager import RiskManager
from strategy.strategy import TradeSignal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_signal(
    symbol="XAGUSD",
    direction="bullish",
    entry=25.0,
    sl=24.5,
    tp=26.0,
) -> TradeSignal:
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    return TradeSignal(
        symbol=symbol,
        direction=direction,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        rr_ratio=round(reward / risk, 2),
        confluence_score=0.75,
    )


def make_csv_content(n_rows=10, freq="15min", base_price=25.0) -> str:
    """Return a CSV string with standard OHLCV columns."""
    idx = pd.date_range("2024-01-01", periods=n_rows, freq=freq, tz="UTC")
    rows = []
    for i, ts in enumerate(idx):
        price = base_price + i * 0.01
        rows.append(
            f"{ts},{price:.4f},{price+0.05:.4f},{price-0.05:.4f},{price+0.02:.4f},1000"
        )
    header = "datetime,open,high,low,close,volume"
    return header + "\n" + "\n".join(rows) + "\n"


# ---------------------------------------------------------------------------
# RiskManager tests
# ---------------------------------------------------------------------------

class TestRiskManager:
    def setup_method(self):
        self.rm = RiskManager()
        self.rm.update_account(50_000)

    def test_position_size_nonzero(self):
        sig = make_signal(entry=25.0, sl=24.5, tp=26.0)
        approved, qty, reason = self.rm.approve_signal(sig)
        assert approved
        assert qty > 0

    def test_position_size_capped_at_max(self):
        """Tiny stop → huge raw qty → must be capped at per-symbol max_position_size."""
        # Very tight stop: 0.001 → raw qty would be enormous
        sig = make_signal(entry=25.0, sl=24.999, tp=26.0)
        approved, qty, reason = self.rm.approve_signal(sig)
        if approved:
            sym_cfg = CONFIG.symbols.get(sig.symbol)
            per_symbol_max = getattr(sym_cfg, "max_position_size", float("inf"))
            assert qty <= per_symbol_max

    def test_rejected_if_rr_below_minimum(self):
        # RR = 0.5 (reward < risk) — should be rejected
        sig = make_signal(entry=25.0, sl=24.0, tp=25.5)
        approved, qty, reason = self.rm.approve_signal(sig)
        assert not approved
        assert "RR ratio" in reason

    def test_rejected_if_kill_switch_active(self):
        self.rm._kill_switch_active = True
        sig = make_signal()
        approved, qty, reason = self.rm.approve_signal(sig)
        assert not approved
        assert "Kill switch" in reason

    def test_daily_trade_limit_enforced(self):
        self.rm._daily_trade_count = CONFIG.risk.max_daily_trades
        sig = make_signal()
        approved, qty, reason = self.rm.approve_signal(sig)
        assert not approved
        assert "Daily trade limit" in reason

    def test_consecutive_loss_kill_switch(self):
        # Simulate enough consecutive losses to trigger kill switch
        self.rm._consecutive_losses = CONFIG.risk.kill_switch_consecutive_losses
        self.rm._check_kill_switch()
        assert self.rm.kill_switch_active

    def test_kill_switch_reset(self):
        self.rm._kill_switch_active = True
        self.rm.reset_kill_switch("test")
        assert not self.rm.kill_switch_active
        assert self.rm.consecutive_losses == 0

    def test_register_and_close_trade(self):
        from risk.risk_manager import OpenTrade

        sig = make_signal()
        _, qty, _ = self.rm.approve_signal(sig)
        trade = OpenTrade(
            signal=sig,
            quantity=qty,
            actual_entry=sig.entry_price,
            order_id="test-001",
            open_time=datetime.now(timezone.utc),
        )
        self.rm.register_open_trade(trade)
        assert "test-001" in self.rm.open_trades

        pnl = self.rm.register_closed_trade("test-001", 26.0, "tp")
        assert pnl is not None
        assert "test-001" not in self.rm.open_trades

    def test_winning_trade_resets_consecutive_losses(self):
        from risk.risk_manager import OpenTrade

        self.rm._consecutive_losses = 3
        sig = make_signal()
        trade = OpenTrade(
            signal=sig,
            quantity=1.0,
            actual_entry=sig.entry_price,
            order_id="win-001",
            open_time=datetime.now(timezone.utc),
        )
        self.rm.register_open_trade(trade)
        # Close at profit (exit > entry for bullish)
        self.rm.register_closed_trade("win-001", 26.5, "tp")
        assert self.rm.consecutive_losses == 0

    def test_status_report_fields(self):
        status = self.rm.status_report()
        expected_keys = {
            "equity", "start_of_day_equity", "daily_pnl", "daily_pnl_pct",
            "open_trades", "daily_trade_count", "consecutive_losses", "kill_switch",
        }
        assert expected_keys.issubset(set(status.keys()))


# ---------------------------------------------------------------------------
# CSV Loader tests
# ---------------------------------------------------------------------------

class TestCSVLoader:
    def _write_tmp(self, content: str, tmp_path, name="test.csv") -> str:
        p = tmp_path / name
        p.write_text(content)
        return str(p)

    def test_loads_standard_csv(self, tmp_path):
        content = make_csv_content(n_rows=20)
        path = self._write_tmp(content, tmp_path)
        df = load_csv(path, "M15")
        assert len(df) == 20
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_datetime_index_is_utc(self, tmp_path):
        content = make_csv_content(n_rows=5)
        path = self._write_tmp(content, tmp_path)
        df = load_csv(path)
        assert df.index.tz is not None
        assert str(df.index.tz) == "UTC"

    def test_sorted_ascending(self, tmp_path):
        # Write rows in reverse order
        idx = pd.date_range("2024-01-01", periods=10, freq="15min", tz="UTC")
        rows = [f"{ts},25.0,25.1,24.9,25.0,100" for ts in reversed(idx)]
        content = "datetime,open,high,low,close,volume\n" + "\n".join(rows)
        path = self._write_tmp(content, tmp_path)
        df = load_csv(path)
        assert df.index.is_monotonic_increasing

    def test_missing_volume_column_fills_zero(self, tmp_path):
        rows = ["datetime,open,high,low,close"]
        for i in range(5):
            rows.append(f"2024-01-0{i+1} 00:00:00+00:00,25.0,25.1,24.9,25.0")
        path = self._write_tmp("\n".join(rows), tmp_path)
        df = load_csv(path)
        assert "volume" in df.columns
        assert (df["volume"] == 0.0).all()

    def test_missing_required_column_raises(self, tmp_path):
        # No 'low' column
        content = "datetime,open,high,close,volume\n2024-01-01,25,25.1,25.0,100"
        path = self._write_tmp(content, tmp_path)
        with pytest.raises(ValueError, match="missing required columns"):
            load_csv(path)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_csv("/nonexistent/path/data.csv")

    def test_unix_timestamp_column(self, tmp_path):
        # timestamp column (unix seconds)
        base_ts = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp())
        rows = ["timestamp,open,high,low,close,volume"]
        for i in range(5):
            rows.append(f"{base_ts + i * 900},25.0,25.1,24.9,25.0,100")
        path = self._write_tmp("\n".join(rows), tmp_path)
        df = load_csv(path)
        assert len(df) == 5
        assert df.index[0] == pd.Timestamp("2024-01-01", tz="UTC")

    def test_load_pair_overlap_validation(self, tmp_path):
        # M15 and H1 with no overlapping range → should raise
        m15_content = "datetime,open,high,low,close,volume\n2024-01-01 00:00:00+00:00,25,25.1,24.9,25,100"
        h1_content = "datetime,open,high,low,close,volume\n2025-01-01 00:00:00+00:00,25,25.1,24.9,25,100"
        m15_path = self._write_tmp(m15_content, tmp_path, "m15.csv")
        h1_path = self._write_tmp(h1_content, tmp_path, "h1.csv")
        with pytest.raises(ValueError, match="no overlapping time range"):
            load_pair(m15_path, h1_path)

    def test_load_pair_returns_both_dfs(self, tmp_path):
        m15_content = make_csv_content(n_rows=50, freq="15min")
        h1_content = make_csv_content(n_rows=20, freq="1h")
        m15_path = self._write_tmp(m15_content, tmp_path, "m15.csv")
        h1_path = self._write_tmp(h1_content, tmp_path, "h1.csv")
        m15, h1 = load_pair(m15_path, h1_path)
        assert len(m15) == 50
        assert len(h1) == 20
