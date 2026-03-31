"""
Execution Engine — places and manages orders via IBKR API.

Supports:
- Limit bracket orders (entry + SL + TP as an OCA group)
- Market bracket orders (fallback)
- Order timeout: cancel unfilled limit entries after N M15 bars
- Slippage detection on fills
- Emergency kill-switch close-all

Order flow:
    signal → risk approval → qualify contract → place bracket → monitor → close
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, Optional

from ib_insync import IB, LimitOrder, MarketOrder, StopOrder, Trade

from config.settings import CONFIG
from data.data_handler import DataHandler
from risk.risk_manager import OpenTrade, RiskManager
from strategy.strategy import TradeSignal
from utils.logger import TradeLogger, execution_log


class ExecutionEngine:
    """
    Manages the full lifecycle of orders: placement, monitoring, and closure.

    Usage:
        engine = ExecutionEngine(ib, data_handler, risk_manager, trade_logger)
        await engine.execute(signal)
        # In each M15 callback:
        await engine.cancel_stale_orders(current_bar_count)
    """

    def __init__(
        self,
        ib: IB,
        data_handler: DataHandler,
        risk_manager: RiskManager,
        trade_logger: TradeLogger,
    ):
        self.ib = ib
        self.data = data_handler
        self.risk = risk_manager
        self.logger = trade_logger

        # order_id → OpenTrade
        self._active_trades: Dict[str, OpenTrade] = {}

        # order_id → bar count when order was placed (for timeout)
        self._order_placed_at_bar: Dict[str, int] = {}

        # order_id → (tp_order, sl_order) for cancellation on timeout
        self._bracket_children: Dict[str, tuple] = {}

        # Running bar count (incremented on each M15 bar close)
        self._bar_count: int = 0

        # Wire IBKR event callbacks
        self.ib.orderStatusEvent += self._on_order_status
        self.ib.execDetailsEvent += self._on_exec_details

    # ------------------------------------------------------------------
    # Public: count bars (call from on_m15_bar callback)
    # ------------------------------------------------------------------

    def tick_bar(self) -> int:
        """Increment internal bar counter. Call once per M15 bar close."""
        self._bar_count += 1
        return self._bar_count

    # ------------------------------------------------------------------
    # Signal execution
    # ------------------------------------------------------------------

    async def execute(self, signal: TradeSignal) -> Optional[str]:
        """
        Execute a trade signal.

        Steps:
          1. Risk approval + position sizing
          2. Qualify contract
          3. Place bracket order (parent + TP + SL)
          4. Register with risk manager
          5. Log signal

        Returns order_id on success, None on failure.
        """
        # 1. Risk approval
        approved, quantity, reason = self.risk.approve_signal(signal)
        if not approved:
            execution_log.warning(f"Signal rejected by risk: {reason}")
            self._log_signal(signal, action="rejected", reject_reason=reason)
            return None

        # Enforce minimum qty from symbol config
        sym_cfg = CONFIG.symbols.get(signal.symbol)
        if sym_cfg and quantity < sym_cfg.min_qty:
            execution_log.warning(
                f"Calculated qty {quantity:.2f} below minimum "
                f"{sym_cfg.min_qty} for {signal.symbol}. Boosting to minimum."
            )
            quantity = sym_cfg.min_qty

        # Round to qty_step
        if sym_cfg and sym_cfg.qty_step > 0:
            quantity = round(quantity / sym_cfg.qty_step) * sym_cfg.qty_step

        # Check if this symbol uses IBKR for execution.
        # CRYPTO/PAXOS orders are rejected in paper accounts (Error 321 Read-Only).
        # For these, log the signal as a paper trade and return without placing.
        sym_cfg_exec = CONFIG.symbols.get(signal.symbol)
        if sym_cfg_exec and sym_cfg_exec.sec_type == "CRYPTO":
            execution_log.info(
                f"[{signal.symbol}] CRYPTO paper-trade (PAXOS not supported in paper): "
                f"{signal.direction.upper()} entry={signal.entry_price:.2f} "
                f"sl={signal.stop_loss:.2f} tp={signal.take_profit:.2f} "
                f"qty={quantity:.4f} RR={signal.rr_ratio:.1f}"
            )
            self._log_signal(signal, action="paper_logged", reject_reason="CRYPTO_PAPER_ONLY")
            return None

        if not self.data.is_connected():
            execution_log.error("IBKR not connected — cannot place order.")
            return None

        # 2. Qualify contract
        try:
            contract = await self.data._get_contract(signal.symbol)
        except Exception as e:
            execution_log.error(f"Contract qualification failed: {e}")
            return None

        # 3. Place bracket order
        action = "BUY" if signal.direction == "bullish" else "SELL"
        close_action = "SELL" if action == "BUY" else "BUY"

        try:
            if signal.entry_type == "limit":
                # ib.bracketOrder() creates correctly linked parent + TP + SL
                bracket = self.ib.bracketOrder(
                    action=action,
                    quantity=quantity,
                    limitPrice=round(signal.entry_price, 5),
                    takeProfitPrice=round(signal.take_profit, 5),
                    stopLossPrice=round(signal.stop_loss, 5),
                )
            else:
                # Market entry + attached SL/TP
                parent_order = MarketOrder(action, quantity)
                parent_order.transmit = False

                tp_order = LimitOrder(
                    close_action,
                    quantity,
                    round(signal.take_profit, 5),
                )
                tp_order.transmit = False

                sl_order = StopOrder(
                    close_action,
                    quantity,
                    round(signal.stop_loss, 5),
                )
                sl_order.transmit = True  # transmit all together

                bracket = (parent_order, tp_order, sl_order)

            # Place all three at once — parent first, then children
            trades = []
            for order in bracket:
                t = self.ib.placeOrder(contract, order)
                trades.append(t)

            parent_trade = trades[0]
            await asyncio.sleep(0.5)  # brief pause for acknowledgement

            order_id = str(parent_trade.order.orderId)
            if order_id == "0":
                execution_log.error(
                    "Order placement failed — IBKR returned orderId=0. "
                    "Check TWS connection and account permissions."
                )
                return None

            execution_log.info(
                f"Bracket order placed: {signal.symbol} {action} "
                f"qty={quantity:.4f}  entry={signal.entry_price:.5f}  "
                f"sl={signal.stop_loss:.5f}  tp={signal.take_profit:.5f}  "
                f"orderId={order_id}  type={signal.entry_type}"
            )

            # Store child orders for potential timeout cancellation
            if len(trades) == 3:
                self._bracket_children[order_id] = (trades[1], trades[2])

        except Exception as e:
            execution_log.error(f"Order placement error: {e}", exc_info=True)
            return None

        # 4. Register with risk manager
        open_trade = OpenTrade(
            signal=signal,
            quantity=quantity,
            actual_entry=signal.entry_price,  # updated on fill callback
            order_id=order_id,
            open_time=datetime.now(timezone.utc),
        )
        self.risk.register_open_trade(open_trade)
        self._active_trades[order_id] = open_trade
        self._order_placed_at_bar[order_id] = self._bar_count

        # 5. Log signal
        self._log_signal(signal, action="placed", reject_reason="")

        return order_id

    # ------------------------------------------------------------------
    # Order timeout
    # ------------------------------------------------------------------

    async def cancel_stale_orders(self) -> int:
        """
        Cancel unfilled limit orders that have been open longer than
        strategy_cfg.order_timeout_bars M15 bars.

        Call this at the top of each M15 bar callback.
        Returns number of orders cancelled.
        """
        timeout = CONFIG.strategy.order_timeout_bars
        cancelled = 0

        for order_id in list(self._order_placed_at_bar.keys()):
            placed_at = self._order_placed_at_bar[order_id]
            age_bars = self._bar_count - placed_at

            if age_bars < timeout:
                continue

            trade = self._active_trades.get(order_id)
            if trade is None:
                # Already closed / filled — clean up tracker
                self._order_placed_at_bar.pop(order_id, None)
                continue

            # Check if the order is still pending (not yet filled)
            open_orders = {str(o.orderId) for o in self.ib.openOrders()}
            if order_id not in open_orders:
                # Not in open orders — either filled, cancelled, or lost after reconnect.
                # Also check ib.trades() to distinguish filled vs unknown.
                all_trade_ids = {str(t.order.orderId) for t in self.ib.trades()}
                if order_id not in all_trade_ids:
                    # Completely unknown to IBKR (e.g. dropped after reconnect) — purge
                    execution_log.warning(
                        f"Order {order_id} ({trade.symbol} {trade.direction}) not found "
                        f"in IBKR trades after reconnect — purging from active trades."
                    )
                    self.risk.register_closed_trade(order_id, trade.entry_price, "reconnect_purge")
                    self._active_trades.pop(order_id, None)
                self._order_placed_at_bar.pop(order_id, None)
                continue

            # Cancel the order
            execution_log.info(
                f"Cancelling stale order {order_id} ({age_bars} bars old, "
                f"timeout={timeout}): {trade.symbol} {trade.direction}"
            )
            try:
                # Find and cancel the Trade object
                for ib_trade in self.ib.trades():
                    if str(ib_trade.order.orderId) == order_id:
                        self.ib.cancelOrder(ib_trade.order)
                        break

                # Also cancel children
                children = self._bracket_children.get(order_id, ())
                for child_trade in children:
                    try:
                        self.ib.cancelOrder(child_trade.order)
                    except Exception:
                        pass

            except Exception as e:
                execution_log.error(f"Failed to cancel stale order {order_id}: {e}")

            self.risk.register_closed_trade(order_id, trade.entry_price, "timeout")
            self._active_trades.pop(order_id, None)
            self._order_placed_at_bar.pop(order_id, None)
            self._bracket_children.pop(order_id, None)
            cancelled += 1

        return cancelled

    # ------------------------------------------------------------------
    # IBKR order/fill callbacks
    # ------------------------------------------------------------------

    def _on_order_status(self, trade: Trade):
        """Called by ib_insync when any order status changes."""
        order_id = str(trade.order.orderId)
        status = trade.orderStatus.status

        execution_log.debug(f"Order status: id={order_id}  status={status}")

        if status == "Filled":
            fill_price = trade.orderStatus.avgFillPrice
            open_trade = self._active_trades.get(order_id)

            if open_trade:
                ok, slippage = self.risk.check_fill_slippage(open_trade.signal, fill_price)
                open_trade.actual_entry = fill_price
                self._order_placed_at_bar.pop(order_id, None)  # no longer pending

                execution_log.info(
                    f"FILLED: {open_trade.symbol} {fill_price:.5f}  "
                    f"(expected={open_trade.signal.entry_price:.5f}  "
                    f"slippage={slippage*100:.3f}%)"
                )

                if not ok:
                    execution_log.warning(
                        f"Slippage {slippage*100:.3f}% exceeds max "
                        f"{CONFIG.risk.max_slippage_pct*100:.3f}% — trade remains open."
                    )

        elif status in ("Cancelled", "Inactive", "ApiCancelled", "ApiError"):
            open_trade = self._active_trades.pop(order_id, None)
            if open_trade:
                self.risk.register_closed_trade(order_id, open_trade.entry_price, status)
                self._order_placed_at_bar.pop(order_id, None)
                self._bracket_children.pop(order_id, None)
                execution_log.info(f"Order {order_id} {status} — removed from active.")

    def _on_exec_details(self, trade: Trade, fill):
        """Called on confirmed execution detail (actual fill receipt)."""
        order_id = str(trade.order.orderId)
        execution_log.debug(
            f"Exec detail: id={order_id}  "
            f"price={fill.execution.price:.5f}  "
            f"qty={fill.execution.shares}"
        )

        # Update PnL tracking when SL or TP child order fills
        # The child orders have parentId set; match by checking if this
        # order_id is a child of a tracked trade
        for parent_id, children in self._bracket_children.items():
            child_ids = {str(c.order.orderId) for c in children}
            if order_id in child_ids:
                parent_trade = self._active_trades.get(parent_id)
                if parent_trade:
                    exit_price = fill.execution.price
                    # Determine exit reason from order type
                    order_type = trade.order.orderType
                    exit_reason = "tp" if order_type in ("LMT", "LMT + MKT") else "sl"

                    pnl = self.risk.register_closed_trade(parent_id, exit_price, exit_reason)
                    self.logger.log_trade({
                        "symbol":           parent_trade.symbol,
                        "direction":        parent_trade.direction,
                        "entry_price":      parent_trade.actual_entry,
                        "exit_price":       exit_price,
                        "stop_loss":        parent_trade.stop_loss,
                        "take_profit":      parent_trade.take_profit,
                        "quantity":         parent_trade.quantity,
                        "pnl":              pnl or 0.0,
                        "exit_reason":      exit_reason,
                        "confluence_score": parent_trade.signal.confluence_score,
                        "order_id":         parent_id,
                        "bos":              parent_trade.signal.score_bos,
                        "fvg":              parent_trade.signal.score_fvg,
                        "liquidity_sweep":  parent_trade.signal.score_liquidity,
                        "session":          parent_trade.signal.score_session,
                        "pd_zone":          parent_trade.signal.score_pd_zone,
                    })
                    self._active_trades.pop(parent_id, None)
                    self._order_placed_at_bar.pop(parent_id, None)
                    self._bracket_children.pop(parent_id, None)

                    execution_log.info(
                        f"Trade {parent_id} closed via {exit_reason}: "
                        f"exit={exit_price:.5f}  pnl=${pnl:.2f}"
                    )
                break

    # ------------------------------------------------------------------
    # Manual / emergency close
    # ------------------------------------------------------------------

    async def close_trade(self, order_id: str, reason: str = "manual") -> bool:
        """Market-close a specific open trade."""
        trade = self._active_trades.get(order_id)
        if trade is None:
            execution_log.warning(f"close_trade: {order_id} not found in active trades")
            return False

        try:
            contract = await self.data._get_contract(trade.symbol)
            close_action = "SELL" if trade.direction == "bullish" else "BUY"
            close_order = MarketOrder(close_action, trade.quantity)
            self.ib.placeOrder(contract, close_order)
            await asyncio.sleep(2)

            current_price = self.data.latest_price(trade.symbol) or trade.entry_price
            pnl = self.risk.register_closed_trade(order_id, current_price, reason)

            self.logger.log_trade({
                "symbol":           trade.symbol,
                "direction":        trade.direction,
                "entry_price":      trade.actual_entry,
                "exit_price":       current_price,
                "stop_loss":        trade.stop_loss,
                "take_profit":      trade.take_profit,
                "quantity":         trade.quantity,
                "pnl":              pnl or 0.0,
                "exit_reason":      reason,
                "confluence_score": trade.signal.confluence_score,
                "order_id":         order_id,
            })

            self._active_trades.pop(order_id, None)
            self._order_placed_at_bar.pop(order_id, None)
            self._bracket_children.pop(order_id, None)

            execution_log.info(
                f"Trade {order_id} closed ({reason}): pnl=${pnl or 0:.2f}"
            )
            return True

        except Exception as e:
            execution_log.error(f"Failed to close trade {order_id}: {e}", exc_info=True)
            return False

    async def close_all_trades(self, reason: str = "kill_switch") -> None:
        """Emergency: market-close all open trades immediately."""
        execution_log.warning(f"Closing ALL trades: reason={reason}")
        for order_id in list(self._active_trades.keys()):
            await self.close_trade(order_id, reason)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def active_trades(self) -> Dict[str, OpenTrade]:
        return self._active_trades

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _log_signal(self, signal: TradeSignal, action: str, reject_reason: str) -> None:
        self.logger.log_signal({
            "symbol":           signal.symbol,
            "direction":        signal.direction,
            "entry_price":      signal.entry_price,
            "stop_loss":        signal.stop_loss,
            "take_profit":      signal.take_profit,
            "rr_ratio":         signal.rr_ratio,
            "confluence_score": signal.confluence_score,
            "action":           action,
            "reject_reason":    reject_reason,
        })
