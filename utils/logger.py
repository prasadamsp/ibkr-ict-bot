"""
Structured logging for the trading system.
- System log: rotating file + console
- Trade log: CSV with structured fields
- Signal log: CSV for all signals (taken and skipped)
"""

import csv
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Any, Optional


def setup_logger(name: str, log_file: str, level: str = "INFO") -> logging.Logger:
    """Create a logger with rotating file handler and console output."""
    Path(os.path.dirname(log_file)).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:
        return logger  # already configured

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rotating file: 10MB max, keep 5 files
    fh = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


class TradeLogger:
    """
    Append-only CSV logger for trade records.
    Each row = one completed trade (entry + exit).
    """

    TRADE_FIELDS = [
        "timestamp", "symbol", "direction", "entry_price", "exit_price",
        "stop_loss", "take_profit", "quantity", "pnl", "pnl_pct",
        "hold_bars", "exit_reason", "confluence_score",
        "bos", "fvg", "liquidity_sweep", "session", "pd_zone",
        "order_id", "notes",
    ]

    SIGNAL_FIELDS = [
        "timestamp", "symbol", "direction", "entry_price", "stop_loss",
        "take_profit", "rr_ratio", "confluence_score", "action",
        "reject_reason",
    ]

    def __init__(self, trade_file: str, signal_file: str):
        Path(os.path.dirname(trade_file)).mkdir(parents=True, exist_ok=True)
        self.trade_file = trade_file
        self.signal_file = signal_file
        self._init_csv(trade_file, self.TRADE_FIELDS)
        self._init_csv(signal_file, self.SIGNAL_FIELDS)

    def _init_csv(self, path: str, fields: list):
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()

    def log_trade(self, record: Dict[str, Any]):
        record["timestamp"] = record.get("timestamp", datetime.utcnow().isoformat())
        self._append(self.trade_file, self.TRADE_FIELDS, record)

    def log_signal(self, record: Dict[str, Any]):
        record["timestamp"] = record.get("timestamp", datetime.utcnow().isoformat())
        self._append(self.signal_file, self.SIGNAL_FIELDS, record)

    def _append(self, path: str, fields: list, record: Dict):
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writerow(record)


# Module-level logger instances (import these)
system_log = setup_logger("system", "logs/system.log")
data_log = setup_logger("data", "logs/system.log")
strategy_log = setup_logger("strategy", "logs/system.log")
execution_log = setup_logger("execution", "logs/system.log")
risk_log = setup_logger("risk", "logs/system.log")
