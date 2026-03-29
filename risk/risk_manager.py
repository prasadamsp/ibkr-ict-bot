"""
Risk Manager — professional-grade position sizing and risk controls.

Rules enforced (all hard limits, not suggestions):
───────────────────────────────────────────────────
1. Risk per trade: max 1% of account equity (default 0.5%).
2. Position size = risk_amount / (entry - stop_loss) in price units × lot_size.
3. Daily loss limit: if realized + unrealized P&L < -3% of start-of-day equity → halt.
4. Max concurrent trades: default 3.
5. Min RR ratio: 2:1.
6. Kill switch: triggered by drawdown > 5% OR 5 consecutive losses.
7. Slippage check: reject order if fill price > max_slippage_pct from expected.
"""

from datetime import date, datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Give-back rule constants
# When monthly P&L exceeds 3× the expected monthly return, position size
# is halved for the rest of the month — protecting gains already banked.
# ---------------------------------------------------------------------------
_GIVEBACK_TRIGGER     = 3.0   # monthly P&L > 3× expected → activate
_GIVEBACK_MULTIPLIER  = 0.5   # cut all new positions to 50% size
_EXPECTED_MONTHLY_PCT = 0.02  # 2% of equity = a solid month baseline

from config.settings import CONFIG, RiskConfig, SymbolConfig
from strategy.strategy import TradeSignal
from utils.logger import risk_log


# ---------------------------------------------------------------------------
# Trade record for open positions
# ---------------------------------------------------------------------------

class OpenTrade:
    def __init__(
        self,
        signal: TradeSignal,
        quantity: float,
        actual_entry: float,
        order_id: str,
        open_time: datetime,
    ):
        self.signal = signal
        self.symbol = signal.symbol
        self.direction = signal.direction
        self.quantity = quantity
        self.entry_price = actual_entry
        self.stop_loss = signal.stop_loss
        self.take_profit = signal.take_profit
        self.order_id = order_id
        self.open_time = open_time
        self.risk_amount = 0.0  # set after creation

    def unrealized_pnl(self, current_price: float, pip_value: float) -> float:
        """Calculate unrealized P&L at current price."""
        if self.direction == "bullish":
            return (current_price - self.entry_price) * self.quantity * pip_value
        else:
            return (self.entry_price - current_price) * self.quantity * pip_value


# ---------------------------------------------------------------------------
# Risk Manager
# ---------------------------------------------------------------------------

class RiskManager:
    """
    Manages all risk controls for the trading system.

    Usage:
        rm = RiskManager()
        rm.update_account(equity=50000)
        approved, qty, reason = rm.approve_signal(signal)
    """

    def __init__(self, cfg: RiskConfig = None):
        self.cfg = cfg or CONFIG.risk
        self._equity: float = 0.0
        self._start_of_day_equity: float = 0.0
        self._open_trades: Dict[str, OpenTrade] = {}   # order_id → trade
        self._daily_realized_pnl: float = 0.0
        self._daily_trade_count: int = 0
        self._consecutive_losses: int = 0
        self._kill_switch_active: bool = False
        self._last_reset_date: Optional[date] = None

        # Give-back rule state
        self._monthly_realized_pnl: float = 0.0
        self._month_start_equity: float = 0.0
        self._giveback_active: bool = False
        self._last_reset_month: Optional[int] = None   # (year*12 + month)

    # ------------------------------------------------------------------
    # Account state
    # ------------------------------------------------------------------

    def update_account(self, equity: float):
        """Call this periodically with current account equity."""
        self._equity = equity

        today = datetime.now(timezone.utc).date()
        if self._last_reset_date != today:
            self._reset_daily(equity)

        # Check kill switch
        self._check_kill_switch()

    def _reset_daily(self, equity: float):
        today = datetime.now(timezone.utc).date()
        risk_log.info(
            f"Daily reset: date={today} equity={equity:.2f} "
            f"prev_realized_pnl={self._daily_realized_pnl:.2f}"
        )
        self._start_of_day_equity = equity
        self._daily_realized_pnl = 0.0
        self._daily_trade_count = 0
        self._last_reset_date = today

        # Monthly give-back reset — fires on the first daily reset of each month
        month_key = today.year * 12 + today.month
        if self._last_reset_month != month_key:
            prev_monthly = self._monthly_realized_pnl
            self._monthly_realized_pnl = 0.0
            self._month_start_equity = equity
            self._giveback_active = False
            self._last_reset_month = month_key
            risk_log.info(
                f"Monthly reset: month={today.year}-{today.month:02d} "
                f"prev_monthly_pnl={prev_monthly:.2f} giveback=OFF"
            )

    def _check_kill_switch(self):
        """Activate kill switch if any hard limit is breached."""
        if self._kill_switch_active:
            return

        # Drawdown kill switch
        if self._start_of_day_equity > 0:
            daily_loss_pct = self._daily_realized_pnl / self._start_of_day_equity
            if daily_loss_pct <= -self.cfg.max_daily_loss:
                risk_log.critical(
                    f"KILL SWITCH: Daily loss {daily_loss_pct*100:.2f}% "
                    f"exceeds limit {self.cfg.max_daily_loss*100:.2f}%"
                )
                self._kill_switch_active = True
                return

        # Consecutive losses kill switch
        if self._consecutive_losses >= self.cfg.kill_switch_consecutive_losses:
            risk_log.critical(
                f"KILL SWITCH: {self._consecutive_losses} consecutive losses."
            )
            self._kill_switch_active = True
            return

        # Equity drawdown kill switch
        if self._start_of_day_equity > 0:
            total_drawdown = (self._equity - self._start_of_day_equity) / self._start_of_day_equity
            if total_drawdown <= -self.cfg.kill_switch_drawdown:
                risk_log.critical(
                    f"KILL SWITCH: Equity drawdown {total_drawdown*100:.2f}% "
                    f"exceeds {self.cfg.kill_switch_drawdown*100:.2f}%"
                )
                self._kill_switch_active = True

    def reset_kill_switch(self, authorized_by: str = "manual"):
        """Manually reset kill switch. Requires explicit authorization."""
        risk_log.warning(f"Kill switch reset by: {authorized_by}")
        self._kill_switch_active = False
        self._consecutive_losses = 0

    # ------------------------------------------------------------------
    # Signal approval + position sizing
    # ------------------------------------------------------------------

    def approve_signal(
        self, signal: TradeSignal
    ) -> Tuple[bool, float, str]:
        """
        Check if a signal passes all risk rules.

        Returns: (approved: bool, quantity: float, reason: str)
        """
        # Kill switch
        if self._kill_switch_active:
            return False, 0.0, "Kill switch active"

        # Account equity available
        if self._equity <= 0:
            return False, 0.0, "No account equity data"

        # Daily trade count
        if self._daily_trade_count >= self.cfg.max_daily_trades:
            return False, 0.0, f"Daily trade limit {self.cfg.max_daily_trades} reached"

        # Max concurrent trades
        open_count = len(self._open_trades)
        if open_count >= self.cfg.max_concurrent_trades:
            return False, 0.0, (
                f"Max concurrent trades {self.cfg.max_concurrent_trades} reached "
                f"({open_count} open)"
            )

        # RR ratio check
        if signal.rr_ratio < self.cfg.min_rr_ratio:
            return False, 0.0, (
                f"RR ratio {signal.rr_ratio:.2f} below minimum {self.cfg.min_rr_ratio}"
            )

        # Daily loss check
        if self._start_of_day_equity > 0:
            daily_loss_pct = self._daily_realized_pnl / self._start_of_day_equity
            if daily_loss_pct <= -self.cfg.max_daily_loss:
                return False, 0.0, (
                    f"Daily loss limit reached: {daily_loss_pct*100:.2f}%"
                )

        # Position sizing
        quantity, size_reason = self._calculate_position_size(signal)
        if quantity <= 0:
            return False, 0.0, f"Position size zero: {size_reason}"

        risk_log.info(
            f"Signal APPROVED: {signal.symbol} {signal.direction} "
            f"qty={quantity:.2f} risk={self._get_risk_amount(signal):.2f} "
            f"entry={signal.entry_price:.4f} sl={signal.stop_loss:.4f}"
        )
        return True, quantity, "approved"

    def _get_risk_amount(self, signal: TradeSignal) -> float:
        """Risk amount in account currency."""
        risk_pct = min(self.cfg.risk_per_trade, self.cfg.max_risk_per_trade)
        return self._equity * risk_pct

    def _calculate_position_size(
        self, signal: TradeSignal
    ) -> Tuple[float, str]:
        """
        Position size = risk_amount / (stop_distance_in_price * pip_value_per_lot)

        For XAGUSD:
            pip_value = $50 per 1 price unit move per lot
            If SL distance = 0.50 and risk = $500:
            qty = 500 / (0.50 * 50) = 20 oz... let's use lot notation.
        """
        sym_cfg = CONFIG.symbols.get(signal.symbol)
        if not sym_cfg:
            return 0.0, f"Unknown symbol {signal.symbol}"

        risk_amount = self._get_risk_amount(signal)
        stop_distance = abs(signal.entry_price - signal.stop_loss)

        if stop_distance <= 0:
            return 0.0, "Zero stop distance"

        # pip_value is per lot per 1 unit price move
        # For XAGUSD: 1 lot = 5000 oz, $1 price move = $5000 P&L per lot
        # But IBKR treats it differently. Use a simplified:
        # P&L per lot = stop_distance * contract_size
        pnl_per_lot = stop_distance * sym_cfg.contract_size

        if pnl_per_lot <= 0:
            return 0.0, "Zero P&L per lot"

        raw_qty = risk_amount / pnl_per_lot

        # Cap: use per-symbol max if defined, else global (RiskConfig has none now)
        per_symbol_max = getattr(sym_cfg, "max_position_size", float("inf"))
        qty = min(raw_qty, per_symbol_max)

        # Give-back rule: halve size when monthly gains are exceptional
        if self._giveback_active:
            qty *= _GIVEBACK_MULTIPLIER
            risk_log.debug(f"Give-back active: size reduced to {_GIVEBACK_MULTIPLIER*100:.0f}%")

        # Round to lot step
        lot_step = getattr(sym_cfg, "qty_step", 1.0) or 1.0
        qty = round(qty / lot_step) * lot_step
        qty = round(qty, 4)

        risk_log.debug(
            f"Position size: symbol={signal.symbol} "
            f"risk=${risk_amount:.2f} stop_dist={stop_distance:.4f} "
            f"pnl_per_lot=${pnl_per_lot:.2f} qty={qty:.2f}"
        )

        return qty, "ok"

    # ------------------------------------------------------------------
    # Slippage check
    # ------------------------------------------------------------------

    def check_fill_slippage(
        self, signal: TradeSignal, actual_fill: float
    ) -> Tuple[bool, float]:
        """
        Check if actual fill price is within acceptable slippage.
        Returns (acceptable: bool, slippage_pct: float).
        """
        expected = signal.entry_price
        slippage = abs(actual_fill - expected) / expected

        if slippage > self.cfg.max_slippage_pct:
            risk_log.warning(
                f"Slippage {slippage*100:.3f}% exceeds max "
                f"{self.cfg.max_slippage_pct*100:.3f}% for {signal.symbol}"
            )
            return False, slippage

        return True, slippage

    # ------------------------------------------------------------------
    # Trade lifecycle
    # ------------------------------------------------------------------

    def register_open_trade(self, trade: OpenTrade):
        """Record that a trade has been opened."""
        self._open_trades[trade.order_id] = trade
        self._daily_trade_count += 1
        risk_log.info(
            f"Trade opened: {trade.symbol} {trade.direction} "
            f"qty={trade.quantity:.2f} entry={trade.entry_price:.4f} "
            f"sl={trade.stop_loss:.4f} tp={trade.take_profit:.4f}"
        )

    def register_closed_trade(
        self, order_id: str, exit_price: float, exit_reason: str
    ) -> Optional[float]:
        """
        Record that a trade has been closed. Returns realized P&L or None.
        """
        if order_id not in self._open_trades:
            risk_log.warning(f"Tried to close unknown trade: {order_id}")
            return None

        trade = self._open_trades.pop(order_id)
        sym_cfg = CONFIG.symbols.get(trade.symbol)
        contract_size = sym_cfg.contract_size if sym_cfg else 1.0

        if trade.direction == "bullish":
            pnl = (exit_price - trade.entry_price) * trade.quantity * contract_size
        else:
            pnl = (trade.entry_price - exit_price) * trade.quantity * contract_size

        self._daily_realized_pnl += pnl
        self._monthly_realized_pnl += pnl

        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        # Give-back rule: protect exceptional winning months
        if not self._giveback_active and self._month_start_equity > 0:
            expected = self._month_start_equity * _EXPECTED_MONTHLY_PCT
            if expected > 0 and self._monthly_realized_pnl >= _GIVEBACK_TRIGGER * expected:
                self._giveback_active = True
                risk_log.warning(
                    f"GIVE-BACK RULE ACTIVATED: monthly_pnl=${self._monthly_realized_pnl:.2f} "
                    f"exceeded {_GIVEBACK_TRIGGER}× expected (${expected:.2f}). "
                    f"Position sizes cut to {_GIVEBACK_MULTIPLIER*100:.0f}% for rest of month."
                )

        risk_log.info(
            f"Trade closed: {trade.symbol} {trade.direction} "
            f"exit={exit_price:.4f} pnl=${pnl:.2f} reason={exit_reason} "
            f"consecutive_losses={self._consecutive_losses}"
        )

        # Check kill switch after closing
        self._check_kill_switch()

        return pnl

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def open_trades(self) -> Dict[str, OpenTrade]:
        return self._open_trades

    @property
    def kill_switch_active(self) -> bool:
        return self._kill_switch_active

    @property
    def daily_pnl(self) -> float:
        return self._daily_realized_pnl

    @property
    def consecutive_losses(self) -> int:
        return self._consecutive_losses

    def status_report(self) -> dict:
        return {
            "equity": self._equity,
            "start_of_day_equity": self._start_of_day_equity,
            "daily_pnl": self._daily_realized_pnl,
            "daily_pnl_pct": (
                self._daily_realized_pnl / self._start_of_day_equity * 100
                if self._start_of_day_equity > 0 else 0
            ),
            "monthly_pnl": self._monthly_realized_pnl,
            "giveback_active": self._giveback_active,
            "open_trades": len(self._open_trades),
            "daily_trade_count": self._daily_trade_count,
            "consecutive_losses": self._consecutive_losses,
            "kill_switch": self._kill_switch_active,
        }
