"""
run_research.py — Autonomous Research Daemon

Runs on the VM as a systemd service. Wakes up every Sunday at 22:00 UTC,
runs the full algo grid search across all 8 instruments, and:
  1. Writes best_algos.json (live bot reads this)
  2. Writes full ranked grid CSV for inspection
  3. Writes a human-readable weekly report
  4. Optionally sends a Telegram notification (if TELEGRAM_TOKEN set)

Also supports one-shot mode (--once) for manual runs.

Usage on VM:
  # One-shot
  cd /opt/trading/IBKR && .venv/bin/python research/run_research.py --once

  # Service mode (loop, runs weekly)
  cd /opt/trading/IBKR && .venv/bin/python research/run_research.py

  # Specific symbols
  cd /opt/trading/IBKR && .venv/bin/python research/run_research.py --once --symbols XAUUSD BTC
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.auto_selector import run_research, ALL_SYMBOLS

_log = logging.getLogger("research")

_RUN_DAY    = 6      # Sunday (0=Mon, 6=Sun)
_RUN_HOUR   = 22     # 22:00 UTC
_SLEEP_SECS = 1800   # check every 30 minutes


def _send_telegram(message: str) -> None:
    """Send Telegram notification if credentials are configured."""
    token   = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    try:
        import urllib.request, urllib.parse
        url  = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": message}).encode()
        urllib.request.urlopen(url, data=data, timeout=10)
        _log.info("Telegram notification sent")
    except Exception as e:
        _log.warning("Telegram notification failed: %s", e)


def _should_run(last_run: datetime | None) -> bool:
    """Return True if it's Sunday 22:xx UTC and we haven't run today."""
    now = datetime.now(timezone.utc)
    if now.weekday() != _RUN_DAY:
        return False
    if now.hour != _RUN_HOUR:
        return False
    if last_run and last_run.date() == now.date():
        return False
    return True


def run_once(symbols: list[str]) -> dict:
    """Execute one research cycle."""
    _log.info("Research daemon: starting one-shot run at %s UTC",
              datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))
    try:
        results = run_research(symbols)
        # Build notification text
        lines = ["🔬 Weekly Research Complete\n"]
        for sym, info in sorted(results.items()):
            lines.append(
                f"  {sym}: {info['algo']} | "
                f"val={info['val_sharpe']:.2f} test={info['test_sharpe']:.2f} | "
                f"months={info['active_months'][:4]}"
            )
        lines.append(f"\nUpdated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        msg = "\n".join(lines)
        _log.info("\n%s", msg)
        _send_telegram(msg)
        return results
    except Exception as exc:
        _log.error("Research run failed: %s", exc, exc_info=True)
        _send_telegram(f"⚠️ Research run FAILED: {exc}")
        return {}


def run_daemon(symbols: list[str]) -> None:
    """Loop forever, running research on the configured schedule."""
    _log.info("Research daemon started — will run every Sunday %d:00 UTC", _RUN_HOUR)
    _log.info("Symbols: %s", ", ".join(symbols))

    last_run: datetime | None = None

    while True:
        try:
            if _should_run(last_run):
                last_run = datetime.now(timezone.utc)
                run_once(symbols)
            else:
                now = datetime.now(timezone.utc)
                _log.debug("Research daemon sleeping — next run: Sunday %d:00 UTC (now: %s %s)",
                           _RUN_HOUR, now.strftime("%A"), now.strftime("%H:%M"))
        except Exception as exc:
            _log.error("Daemon loop error: %s", exc, exc_info=True)

        time.sleep(_SLEEP_SECS)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/research_daemon.log", mode="a"),
        ],
    )

    parser = argparse.ArgumentParser(description="Autonomous algo research daemon")
    parser.add_argument("--once",    action="store_true", help="Run once and exit")
    parser.add_argument("--symbols", nargs="+", default=None, help="Subset of symbols")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    symbols = [s.upper() for s in args.symbols] if args.symbols else list(ALL_SYMBOLS)

    if args.once:
        results = run_once(symbols)
        sys.exit(0 if results else 1)
    else:
        run_daemon(symbols)


if __name__ == "__main__":
    main()
