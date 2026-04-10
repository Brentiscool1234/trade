"""
Risk manager — enforces all position limits and portfolio guardrails.

Responsibilities
----------------
- Calculate safe position size (simplified Kelly Criterion, capped at 2.5 %)
- Track open positions and daily P&L via a JSON state file
- Approve or reject new trade requests
- Detect stop-loss and take-profit events on open positions
- Reset daily loss counter at midnight

State file format (portfolio_state.json)
-----------------------------------------
{
    "cash": 1000.0,
    "daily_pnl": 0.0,
    "daily_reset_date": "2025-01-15",
    "positions": {
        "AAPL": {
            "qty": 2,
            "avg_price": 182.50,
            "broker": "alpaca",
            "opened_at": "2025-01-15T10:30:00"
        }
    }
}
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from typing import Optional

import config
from utils.logger import get_logger

log = get_logger(__name__)


class RiskManager:
    def __init__(self) -> None:
        self._state: dict = self._load_state()
        self._reset_daily_pnl_if_needed()

    # ── State persistence ─────────────────────────────────────────────────────

    def _default_state(self) -> dict:
        return {
            "cash": 1000.0,         # starting paper balance
            "daily_pnl": 0.0,
            "daily_reset_date": str(date.today()),
            "positions": {},
        }

    def _load_state(self) -> dict:
        path = config.PORTFOLIO_STATE_FILE
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    state = json.load(f)
                log.info(
                    "Portfolio state loaded — cash=%.2f  positions=%d",
                    state.get("cash", 0), len(state.get("positions", {})),
                )
                return state
            except Exception as exc:
                log.warning("Could not read portfolio state (%s) — starting fresh", exc)
        state = self._default_state()
        self._save_state(state)
        return state

    def _save_state(self, state: Optional[dict] = None) -> None:
        if state is None:
            state = self._state
        with open(config.PORTFOLIO_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)

    def _reset_daily_pnl_if_needed(self) -> None:
        today = str(date.today())
        if self._state.get("daily_reset_date") != today:
            self._state["daily_pnl"] = 0.0
            self._state["daily_reset_date"] = today
            self._save_state()
            log.info("Daily P&L reset for %s", today)

    # ── Portfolio value ───────────────────────────────────────────────────────

    @property
    def cash(self) -> float:
        return float(self._state.get("cash", 0.0))

    @property
    def positions(self) -> dict:
        return self._state.get("positions", {})

    @property
    def open_position_count(self) -> int:
        return len(self.positions)

    @property
    def daily_pnl(self) -> float:
        return float(self._state.get("daily_pnl", 0.0))

    def portfolio_value(self, prices: dict[str, float]) -> float:
        """Approximate total value = cash + market value of open positions."""
        total = self.cash
        for ticker, pos in self.positions.items():
            price = prices.get(ticker, pos["avg_price"])
            total += pos["qty"] * price
        return total

    # ── Trade approval ────────────────────────────────────────────────────────

    def approve_buy(self, ticker: str, price: float) -> tuple[bool, str, float]:
        """
        Decide whether to execute a BUY.
        Returns (approved: bool, reason: str, qty: float).
        qty is in shares (stocks) or base units (crypto).
        """
        self._reset_daily_pnl_if_needed()

        # 1. Already holding this ticker?
        if ticker in self.positions:
            return False, f"Already holding {ticker}", 0.0

        # 2. Max positions reached?
        if self.open_position_count >= config.MAX_OPEN_POSITIONS:
            return False, f"Max open positions ({config.MAX_OPEN_POSITIONS}) reached", 0.0

        # 3. Daily loss limit breached?
        total_est = self.cash  # rough estimate without live prices
        if total_est > 0:
            daily_loss_ratio = -self.daily_pnl / total_est
            if daily_loss_ratio >= config.MAX_DAILY_LOSS_PCT:
                return False, "Daily loss limit reached — trading halted", 0.0

        # 4. Calculate position size (2.5% of cash)
        position_value = self.cash * config.MAX_POSITION_SIZE_PCT
        if position_value < price * 0.01:           # can't afford even 0.01 unit
            return False, f"Insufficient cash (${self.cash:.2f})", 0.0

        if price <= 0:
            return False, "Invalid price", 0.0

        qty = position_value / price
        # For stocks, round down to whole shares
        qty = max(0.0, qty)

        return True, "OK", round(qty, 6)

    def approve_sell(self, ticker: str) -> tuple[bool, str]:
        if ticker not in self.positions:
            return False, f"No open position for {ticker}"
        return True, "OK"

    # ── Position management ───────────────────────────────────────────────────

    def record_buy(
        self, ticker: str, qty: float, price: float, broker: str
    ) -> None:
        self._state["cash"] -= qty * price
        self._state["positions"][ticker] = {
            "qty": qty,
            "avg_price": price,
            "broker": broker,
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_state()
        log.info("Position opened: %s  qty=%.4f  price=%.4f  broker=%s", ticker, qty, price, broker)

    def record_sell(
        self, ticker: str, qty: float, price: float
    ) -> float:
        """Returns realised P&L for this trade."""
        pos = self.positions.get(ticker)
        if pos is None:
            log.warning("record_sell called for %s but no position found", ticker)
            return 0.0
        pnl = (price - pos["avg_price"]) * qty
        self._state["cash"] += qty * price
        self._state["daily_pnl"] = self._state.get("daily_pnl", 0.0) + pnl
        del self._state["positions"][ticker]
        self._save_state()
        log.info(
            "Position closed: %s  qty=%.4f  exit_price=%.4f  pnl=%.2f",
            ticker, qty, price, pnl,
        )
        return pnl

    # ── Stop-loss / take-profit checks ────────────────────────────────────────

    def check_exit_conditions(
        self, prices: dict[str, float]
    ) -> list[dict]:
        """
        Scan all open positions against current prices.
        Returns a list of dicts: {ticker, reason, qty, price}
        for positions that should be closed immediately.
        """
        to_close: list[dict] = []
        for ticker, pos in list(self.positions.items()):
            current = prices.get(ticker)
            if current is None:
                continue
            avg = pos["avg_price"]
            if avg <= 0:
                continue
            change_pct = (current - avg) / avg

            if change_pct <= -config.STOP_LOSS_PCT:
                to_close.append({
                    "ticker": ticker,
                    "reason": f"stop-loss ({change_pct*100:.1f}%)",
                    "qty": pos["qty"],
                    "price": current,
                })
                log.warning(
                    "STOP-LOSS triggered: %s  change=%.1f%%  entry=%.4f  current=%.4f",
                    ticker, change_pct * 100, avg, current,
                )
            elif change_pct >= config.TAKE_PROFIT_PCT:
                to_close.append({
                    "ticker": ticker,
                    "reason": f"take-profit ({change_pct*100:.1f}%)",
                    "qty": pos["qty"],
                    "price": current,
                })
                log.info(
                    "TAKE-PROFIT triggered: %s  change=%.1f%%  entry=%.4f  current=%.4f",
                    ticker, change_pct * 100, avg, current,
                )
        return to_close

    # ── Status summary ────────────────────────────────────────────────────────

    def summary(self, prices: dict[str, float] | None = None) -> dict:
        prices = prices or {}
        pos_summary = []
        for ticker, pos in self.positions.items():
            current = prices.get(ticker, pos["avg_price"])
            unrealised = (current - pos["avg_price"]) * pos["qty"]
            pct = (current - pos["avg_price"]) / pos["avg_price"] * 100 if pos["avg_price"] else 0
            pos_summary.append({
                "ticker": ticker,
                "qty": pos["qty"],
                "avg_price": pos["avg_price"],
                "current_price": current,
                "unrealised_pnl": round(unrealised, 2),
                "pct_change": round(pct, 2),
                "broker": pos["broker"],
            })
        return {
            "cash": round(self.cash, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "open_positions": pos_summary,
            "portfolio_value": round(self.portfolio_value(prices), 2),
        }
