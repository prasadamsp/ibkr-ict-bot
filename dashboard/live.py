"""
Rich terminal dashboard for the ICT Multi-Instrument IBKR Bot.

Usage (standalone demo):
    python -m dashboard.live

Usage (from main.py):
    from dashboard.live import LiveDashboard
    dashboard = LiveDashboard()
    dashboard.update("XAUUSD", regime="TRENDING", seasonal_multiplier=1.3, daily_pnl=240)
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# ---------------------------------------------------------------------------
# Per-instrument state
# ---------------------------------------------------------------------------

@dataclass
class InstrumentState:
    symbol: str
    equity: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    open_positions: str = "-"       # e.g. "1 LONG", "2 SHORT", "-"
    regime: str = "UNKNOWN"         # TRENDING | RANGING | HIGH_VOL | LOW_VOL
    seasonal_multiplier: float = 1.0
    win_rate_20: float = 0.0        # 0–100 (%)
    last_signal_time: Optional[str] = None
    status: str = "IDLE"            # IDLE | ACTIVE | HALTED | ERROR


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

REGIME_COLORS: Dict[str, str] = {
    "TRENDING": "green",
    "RANGING": "yellow",
    "HIGH_VOL": "red",
    "LOW_VOL": "dim",
    "UNKNOWN": "dim",
}

DEFAULT_SYMBOLS = ["XAUUSD", "NAS100", "EURUSD", "GBPUSD", "BTC", "XAGUSD", "OIL"]


class LiveDashboard:
    """Thread-safe Rich terminal dashboard."""

    def __init__(self, refresh_interval: float = 5.0) -> None:
        self.refresh_interval = refresh_interval
        self.console = Console()
        self._lock = threading.Lock()

        # Portfolio-level state
        self._equity: float = 0.0
        self._daily_pnl: float = 0.0
        self._open_count: int = 0

        # Per-symbol state, pre-populated with defaults
        self._instruments: Dict[str, InstrumentState] = {
            sym: InstrumentState(symbol=sym) for sym in DEFAULT_SYMBOLS
        }

        # Footer info (can be updated via update_footer)
        self._footer_notes: str = ""

    # ------------------------------------------------------------------
    # Public update API
    # ------------------------------------------------------------------

    def update(self, symbol: str, **kwargs: Any) -> None:
        """Update any field(s) for a symbol. Thread-safe."""
        with self._lock:
            if symbol not in self._instruments:
                self._instruments[symbol] = InstrumentState(symbol=symbol)
            state = self._instruments[symbol]
            for key, value in kwargs.items():
                if hasattr(state, key):
                    setattr(state, key, value)

    def set_portfolio(self, equity: float, daily_pnl: float, open_count: int) -> None:
        """Update portfolio-level metrics. Thread-safe."""
        with self._lock:
            self._equity = equity
            self._daily_pnl = daily_pnl
            self._open_count = open_count

    def update_footer(self, notes: str) -> None:
        """Replace the footer notes line (e.g. next event, macro phase)."""
        with self._lock:
            self._footer_notes = notes

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> Layout:
        """Build and return the full Rich Layout. Thread-safe snapshot."""
        with self._lock:
            equity = self._equity
            daily_pnl = self._daily_pnl
            open_count = self._open_count
            instruments = {k: v for k, v in self._instruments.items()}
            footer_notes = self._footer_notes

        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        layout["header"].update(self._build_header(equity, daily_pnl, open_count))
        layout["body"].update(self._build_table(instruments))
        layout["footer"].update(self._build_footer(footer_notes))

        return layout

    def _build_header(self, equity: float, daily_pnl: float, open_count: int) -> Panel:
        pnl_color = "green" if daily_pnl >= 0 else "red"
        pnl_sign = "+" if daily_pnl >= 0 else ""

        header_text = Text(justify="center")
        header_text.append("  ICT MULTI-INSTRUMENT BOT  ", style="bold white")
        header_text.append("|  ", style="dim")
        header_text.append(f"Equity: ${equity:,.2f}", style="bold cyan")
        header_text.append("  |  ", style="dim")
        header_text.append(f"Daily P&L: {pnl_sign}${daily_pnl:,.2f}", style=f"bold {pnl_color}")
        header_text.append("  |  ", style="dim")
        header_text.append(f"Open: {open_count}", style="bold white")
        header_text.append("  |  ", style="dim")
        now_utc = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        header_text.append(now_utc, style="dim")

        return Panel(header_text, box=box.HEAVY_HEAD, style="bold")

    def _build_table(self, instruments: Dict[str, InstrumentState]) -> Table:
        table = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold white",
            expand=True,
            padding=(0, 1),
        )

        table.add_column("Symbol", style="bold", width=8, justify="right")
        table.add_column("Regime", width=10, justify="center")
        table.add_column("Seasonal Bias", width=13, justify="center")
        table.add_column("Open Positions", width=14, justify="center")
        table.add_column("Win Rate (20)", width=13, justify="center")
        table.add_column("P&L", width=10, justify="right")
        table.add_column("Status", width=8, justify="center")

        for sym in DEFAULT_SYMBOLS:
            state = instruments.get(sym, InstrumentState(symbol=sym))
            table.add_row(
                sym,
                self._regime_cell(state.regime),
                self._seasonal_cell(state.seasonal_multiplier),
                self._position_cell(state.open_positions),
                self._winrate_cell(state.win_rate_20),
                self._pnl_cell(state.daily_pnl),
                self._status_cell(state.status),
            )

        # Add any extra symbols not in the default list
        for sym, state in instruments.items():
            if sym not in DEFAULT_SYMBOLS:
                table.add_row(
                    sym,
                    self._regime_cell(state.regime),
                    self._seasonal_cell(state.seasonal_multiplier),
                    self._position_cell(state.open_positions),
                    self._winrate_cell(state.win_rate_20),
                    self._pnl_cell(state.daily_pnl),
                    self._status_cell(state.status),
                )

        return table

    def _build_footer(self, notes: str) -> Panel:
        if notes:
            text = Text(notes, justify="center", style="dim")
        else:
            text = Text(
                "  Next EIA: Wednesday 14:30 UTC  |  BTC: BULL PHASE  |  London: ACTIVE  ",
                justify="center",
                style="dim",
            )
        return Panel(text, box=box.HEAVY_HEAD, style="dim")

    # ------------------------------------------------------------------
    # Cell helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _regime_cell(regime: str) -> Text:
        color = REGIME_COLORS.get(regime.upper(), "dim")
        return Text(regime, style=color, justify="center")

    @staticmethod
    def _seasonal_cell(multiplier: float) -> Text:
        if multiplier > 1.0:
            label = f"{multiplier:.1f}x \u2191"
            style = "green"
        elif multiplier < 1.0:
            label = f"{multiplier:.1f}x \u2193"
            style = "red"
        else:
            label = f"{multiplier:.1f}x"
            style = "white"
        return Text(label, style=style, justify="center")

    @staticmethod
    def _position_cell(open_positions: str) -> Text:
        if open_positions in ("", "-", None):
            return Text("-", style="dim", justify="center")
        pos_upper = open_positions.upper()
        if "LONG" in pos_upper:
            style = "green"
        elif "SHORT" in pos_upper:
            style = "red"
        else:
            style = "white"
        return Text(open_positions, style=style, justify="center")

    @staticmethod
    def _winrate_cell(win_rate: float) -> Text:
        if win_rate <= 0:
            return Text("-", style="dim", justify="center")
        if win_rate >= 60:
            style = "green"
        elif win_rate >= 50:
            style = "yellow"
        else:
            style = "red"
        return Text(f"{win_rate:.0f}%", style=style, justify="center")

    @staticmethod
    def _pnl_cell(pnl: float) -> Text:
        if pnl > 0:
            return Text(f"+${pnl:,.0f}", style="green", justify="right")
        elif pnl < 0:
            return Text(f"-${abs(pnl):,.0f}", style="red", justify="right")
        else:
            return Text("$0", style="dim", justify="right")

    @staticmethod
    def _status_cell(status: str) -> Text:
        status_styles = {
            "ACTIVE": "bold green",
            "IDLE": "dim",
            "HALTED": "bold red",
            "ERROR": "bold red",
        }
        style = status_styles.get(status.upper(), "dim")
        return Text(status, style=style, justify="center")

    # ------------------------------------------------------------------
    # Blocking run loop
    # ------------------------------------------------------------------

    def run_forever(self) -> None:
        """Block and refresh the dashboard at refresh_interval. Ctrl-C to exit."""
        with Live(
            self.render(),
            console=self.console,
            refresh_per_second=1,
            screen=True,
        ) as live:
            try:
                while True:
                    time.sleep(self.refresh_interval)
                    live.update(self.render())
            except KeyboardInterrupt:
                pass


# ---------------------------------------------------------------------------
# Standalone demo — python -m dashboard.live
# ---------------------------------------------------------------------------

def _run_demo() -> None:
    """Run a live demo with fake data for all 7 instruments."""
    import math
    import random

    dashboard = LiveDashboard(refresh_interval=2.0)

    # Set initial portfolio state
    dashboard.set_portfolio(equity=52_430.00, daily_pnl=680.00, open_count=2)
    dashboard.update_footer(
        "  Next EIA: Wednesday 14:30 UTC  |  BTC: BULL PHASE  |  [bold]Press Ctrl-C to exit[/bold]  "
    )

    # Seed per-symbol data
    seed_data = {
        "XAUUSD": dict(regime="TRENDING", seasonal_multiplier=1.3,
                       open_positions="1 LONG", win_rate_20=62.0,
                       daily_pnl=240.0, status="ACTIVE"),
        "NAS100": dict(regime="RANGING", seasonal_multiplier=1.0,
                       open_positions="-", win_rate_20=58.0,
                       daily_pnl=-80.0, status="IDLE"),
        "EURUSD": dict(regime="TRENDING", seasonal_multiplier=0.9,
                       open_positions="-", win_rate_20=55.0,
                       daily_pnl=0.0, status="IDLE"),
        "GBPUSD": dict(regime="HIGH_VOL", seasonal_multiplier=1.0,
                       open_positions="-", win_rate_20=60.0,
                       daily_pnl=0.0, status="IDLE"),
        "BTC":    dict(regime="TRENDING", seasonal_multiplier=1.5,
                       open_positions="1 LONG", win_rate_20=65.0,
                       daily_pnl=520.0, status="ACTIVE"),
        "XAGUSD": dict(regime="RANGING", seasonal_multiplier=1.1,
                       open_positions="-", win_rate_20=50.0,
                       daily_pnl=0.0, status="IDLE"),
        "OIL":    dict(regime="TRENDING", seasonal_multiplier=1.2,
                       open_positions="-", win_rate_20=57.0,
                       daily_pnl=0.0, status="IDLE"),
    }
    for sym, kwargs in seed_data.items():
        dashboard.update(sym, **kwargs)

    # Animate in a background thread — small random P&L drift every cycle
    def _drift() -> None:
        tick = 0
        while True:
            time.sleep(2.0)
            tick += 1
            for sym in DEFAULT_SYMBOLS:
                drift = random.uniform(-15, 20)
                # Fetch current pnl (read from state dict directly)
                with dashboard._lock:
                    current_pnl = dashboard._instruments[sym].daily_pnl
                dashboard.update(sym, daily_pnl=current_pnl + drift)
            # Update portfolio equity with a sine wave for demo effect
            new_equity = 52_430.00 + 200 * math.sin(tick * 0.3)
            new_pnl = sum(
                dashboard._instruments[s].daily_pnl for s in DEFAULT_SYMBOLS
            )
            dashboard.set_portfolio(equity=new_equity, daily_pnl=new_pnl, open_count=2)

    t = threading.Thread(target=_drift, daemon=True)
    t.start()

    dashboard.run_forever()


if __name__ == "__main__":
    _run_demo()
