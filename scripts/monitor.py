"""
Live Trading Monitor
=====================
Terminal dashboard showing real-time P&L, open trades, and log tail.
Works locally or against a remote server via SSH log forwarding.

Local usage:
  python scripts/monitor.py

Remote usage (SSH tunnel or shared log dir):
  python scripts/monitor.py --logs /path/to/remote/logs

Refreshes every REFRESH_SECS seconds. Press Ctrl+C to exit.
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── ensure project root in path ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_csv_tail(path: str, n: int = 20) -> list[dict]:
    """Read last N rows from a CSV as list of dicts."""
    p = Path(path)
    if not p.exists():
        return []
    try:
        import csv
        with open(p, newline="") as f:
            rows = list(csv.DictReader(f))
        return rows[-n:]
    except Exception:
        return []


def _read_log_tail(path: str, n: int = 15) -> list[str]:
    """Read last N lines from a log file."""
    p = Path(path)
    if not p.exists():
        return []
    try:
        with open(p) as f:
            lines = f.readlines()
        return [l.rstrip() for l in lines[-n:]]
    except Exception:
        return []


def _colorize(val: float, fmt: str = "+,.2f") -> str:
    """Return green/red coloured string based on positive/negative value."""
    GREEN = "\033[92m"
    RED   = "\033[91m"
    RESET = "\033[0m"
    s = format(val, fmt)
    if val > 0:
        return f"{GREEN}{s}{RESET}"
    elif val < 0:
        return f"{RED}{s}{RESET}"
    return s


# ---------------------------------------------------------------------------
# Dashboard sections
# ---------------------------------------------------------------------------

def section_summary(trades: list[dict]) -> str:
    """P&L summary from completed trades log."""
    today = datetime.now(timezone.utc).date().isoformat()
    today_trades = [t for t in trades if t.get("timestamp", "").startswith(today)]

    total_pnl = sum(float(t.get("pnl", 0)) for t in today_trades)
    wins  = [t for t in today_trades if float(t.get("pnl", 0)) > 0]
    losses = [t for t in today_trades if float(t.get("pnl", 0)) <= 0]

    all_pnl = sum(float(t.get("pnl", 0)) for t in trades)

    lines = [
        f"  {'Today P&L':<22} {_colorize(total_pnl, '+$,.2f')}",
        f"  {'Today trades':<22} {len(today_trades)}  "
        f"(W:{len(wins)}  L:{len(losses)}  "
        f"WR:{len(wins)/max(len(today_trades),1)*100:.0f}%)",
        f"  {'All-time P&L':<22} {_colorize(all_pnl, '+$,.2f')}",
        f"  {'Total trades':<22} {len(trades)}",
    ]
    return "\n".join(lines)


def section_recent_trades(trades: list[dict], n: int = 5) -> str:
    """Table of last N closed trades."""
    if not trades:
        return "  No trades yet."

    recent = trades[-n:]
    header = f"  {'Time':<20} {'Sym':<8} {'Dir':<8} {'Entry':>8} {'Exit':>8} {'P&L':>10} {'Reason':<10}"
    sep    = "  " + "─" * 76
    rows = [header, sep]

    for t in reversed(recent):
        ts = t.get("timestamp", "")[:16]
        pnl = float(t.get("pnl", 0))
        rows.append(
            f"  {ts:<20} {t.get('symbol',''):<8} {t.get('direction',''):<8} "
            f"{float(t.get('entry_price',0)):>8.4f} {float(t.get('exit_price',0)):>8.4f} "
            f"{_colorize(pnl, '+$,.2f'):>18} {t.get('exit_reason',''):<10}"
        )
    return "\n".join(rows)


def section_log_tail(log_lines: list[str]) -> str:
    """Last N system log lines, colour-coded by level."""
    YELLOW = "\033[93m"; RED = "\033[91m"; RESET = "\033[0m"
    out = []
    for line in log_lines:
        if "CRITICAL" in line or "KILL SWITCH" in line:
            out.append(f"{RED}{line}{RESET}")
        elif "WARNING" in line or "rejected" in line:
            out.append(f"{YELLOW}{line}{RESET}")
        else:
            out.append(f"  {line}")
    return "\n".join(out)


def section_signals(signals: list[dict], n: int = 5) -> str:
    """Last N signal decisions."""
    if not signals:
        return "  No signals yet."

    recent = signals[-n:]
    rows = []
    for s in reversed(recent):
        ts = s.get("timestamp", "")[:16]
        action = s.get("action", "")
        BOLD = "\033[1m" if action == "placed" else ""
        RESET = "\033[0m"
        rows.append(
            f"  {BOLD}{ts:<20} {s.get('symbol',''):<8} {s.get('direction',''):<8} "
            f"score={float(s.get('confluence_score',0)):.2f}  "
            f"RR={float(s.get('rr_ratio',0)):.1f}  "
            f"→ {action}  {s.get('reject_reason','')}{RESET}"
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Main render loop
# ---------------------------------------------------------------------------

def render(logs_dir: str) -> None:
    """Render the full dashboard to stdout."""
    trades_path  = os.path.join(logs_dir, "trades.csv")
    signals_path = os.path.join(logs_dir, "signals.csv")
    log_path     = os.path.join(logs_dir, "system.log")

    trades  = _read_csv_tail(trades_path, 200)
    signals = _read_csv_tail(signals_path, 20)
    log_lines = _read_log_tail(log_path, 12)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    BOLD = "\033[1m"; CYAN = "\033[96m"; RESET = "\033[0m"

    print("\033[H\033[J", end="")  # clear screen
    print(f"{CYAN}{'═'*80}{RESET}")
    print(f"{BOLD}  ICT TRADING BOT MONITOR                          {now}{RESET}")
    print(f"{CYAN}{'═'*80}{RESET}\n")

    print(f"{BOLD}  P&L SUMMARY{RESET}")
    print(section_summary(trades))

    print(f"\n{BOLD}  RECENT CLOSED TRADES{RESET}")
    print(section_recent_trades(trades, 6))

    print(f"\n{BOLD}  RECENT SIGNALS{RESET}")
    print(section_signals(signals, 5))

    print(f"\n{BOLD}  SYSTEM LOG (last 12 lines){RESET}")
    print(section_log_tail(log_lines))

    print(f"\n{CYAN}{'─'*80}{RESET}")
    print(f"  Logs dir: {Path(logs_dir).resolve()}   [Ctrl+C to exit]")


def main():
    parser = argparse.ArgumentParser(description="ICT Trading Bot live monitor")
    parser.add_argument(
        "--logs",
        default="logs",
        help="Path to logs directory (default: ./logs)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=15,
        help="Refresh interval in seconds (default: 15)",
    )
    args = parser.parse_args()

    print("Starting monitor... (press Ctrl+C to exit)")
    try:
        while True:
            render(args.logs)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
