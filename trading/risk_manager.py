"""
Risk manager — enforces all position limits and portfolio guardrails.

Responsibilities
----------------
- ATR-based position sizing (volatility-adjusted, capped at MAX_POSITION_SIZE_PCT)
- Stop-loss / take-profit monitoring
- Trailing stop-loss (activates after a position gains enough)
- Circuit breaker (pauses new entries after N consecutive losses)
- Daily loss limit
- Minimum hold time gate (prevents signal-driven churn)
- Persistent state via JSON (portfolio_state.json)

State file schema
-----------------
{
  "cash": 1000.0,
  "daily_pnl": 0.0,
  "daily_reset_date": "2025-01-15",
  "consecutive_losses": 0,
  "circuit_breaker_until": null,        // ISO timestamp or null
  "positions": {
    "AAPL": {
      "qty": 0.138,
      "avg_price": 181.50,
      "broker": "alpaca",
      "opened_at": "2025-01-15T10:30:00+00:00",
      "trailing_high": 181.50,
      "trailing_stop_active": false
    }
  }
}
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta, timezone
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
            "cash": 1000.0,
            "daily_pnl": 0.0,
            "daily_reset_date": str(date.today()),
            "consecutive_losses": 0,
            "circuit_breaker_until": None,
            "positions": {},
        }

    def _load_state(self) -> dict:
        path = config.PORTFOLIO_STATE_FILE
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    state = json.load(f)
                # Backfill keys added in later versions
                state.setdefault("consecutive_losses", 0)
                state.setdefault("circuit_breaker_until", None)
                for pos in state.get("positions", {}).values():
                    pos.setdefault("trailing_high", pos.get("avg_price", 0))
                    pos.setdefault("trailing_stop_active", False)
                log.info(
                    "Portfolio state loaded — cash=$%.2f  positions=%d  "
                    "consecutive_losses=%d",
                    state.get("cash", 0),
                    len(state.get("positions", {})),
                    state.get("consecutive_losses", 0),
                )
                return state
            except Exception as exc:
                log.warning("Could not read state file (%s) — starting fresh", exc)
        state = self._default_state()
        self._save_state(state)
        return state

    def _save_state(self, state: Optional[dict] = None) -> None:
        if state is None:
            state = self._state
        with open(config.PORTFOLIO_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def _reset_daily_pnl_if_needed(self) -> None:
        today = str(date.today())
        if self._state.get("daily_reset_date") != today:
            self._state["daily_pnl"] = 0.0
            self._state["daily_reset_date"] = today
            self._save_state()
            log.info("Daily P&L reset for %s", today)

    # ── Properties ────────────────────────────────────────────────────────────

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

    @property
    def consecutive_losses(self) -> int:
        return int(self._state.get("consecutive_losses", 0))

    def portfolio_value(self, prices: dict[str, float]) -> float:
        total = self.cash
        for ticker, pos in self.positions.items():
            price = prices.get(ticker, pos["avg_price"])
            total += pos["qty"] * price
        return total

    # ── Circuit breaker ───────────────────────────────────────────────────────

    def _circuit_breaker_active(self) -> tuple[bool, str]:
        """Returns (is_active, reason_string)."""
        cb_until = self._state.get("circuit_breaker_until")
        if not cb_until:
            return False, ""
        until_dt = datetime.fromisoformat(cb_until)
        now = datetime.now(timezone.utc)
        if now < until_dt:
            remaining_h = int((until_dt - now).total_seconds() // 3600)
            remaining_m = int((until_dt - now).total_seconds() % 3600 // 60)
            return True, (
                f"Circuit breaker active — {remaining_h}h {remaining_m}m remaining "
                f"({self.consecutive_losses} consecutive losses)"
            )
        # Expired — clear it
        self._state["circuit_breaker_until"] = None
        self._state["consecutive_losses"] = 0
        self._save_state()
        return False, ""

    def _update_circuit_breaker(self, pnl: float) -> None:
        if pnl < 0:
            losses = self._state.get("consecutive_losses", 0) + 1
            self._state["consecutive_losses"] = losses
            log.debug("Consecutive losses: %d", losses)
            if losses >= config.CIRCUIT_BREAKER_LOSSES:
                until = datetime.now(timezone.utc) + timedelta(
                    hours=config.CIRCUIT_BREAKER_PAUSE_HOURS
                )
                self._state["circuit_breaker_until"] = until.isoformat()
                log.warning(
                    "CIRCUIT BREAKER TRIGGERED: %d consecutive losses. "
                    "All new entries paused until %s",
                    losses,
                    until.strftime("%Y-%m-%d %H:%M UTC"),
                )
        else:
            self._state["consecutive_losses"] = 0
            self._state["circuit_breaker_until"] = None

    # ── Position sizing ───────────────────────────────────────────────────────

    def _calculate_qty(self, price: float, atr: float) -> float:
        """
        ATR-based position sizing (if enabled), always capped by MAX_POSITION_SIZE_PCT.

        ATR formula:
          risk_dollars = portfolio_cash * ATR_RISK_PER_TRADE
          stop_distance = atr * ATR_STOP_MULTIPLIER
          qty = risk_dollars / stop_distance

        High volatility → smaller qty.
        Result is further capped so position value ≤ cash * MAX_POSITION_SIZE_PCT.
        """
        max_position_value = self.cash * config.MAX_POSITION_SIZE_PCT
        max_qty_by_pct = max_position_value / price if price > 0 else 0.0

        if config.USE_ATR_SIZING and atr > 0 and price > 0:
            risk_dollars = self.cash * config.ATR_RISK_PER_TRADE
            stop_distance = atr * config.ATR_STOP_MULTIPLIER
            atr_qty = risk_dollars / stop_distance
            qty = min(atr_qty, max_qty_by_pct)
            log.debug(
                "ATR sizing: risk=$%.2f  stop=%.4f  atr_qty=%.6f  "
                "pct_qty=%.6f  final=%.6f",
                risk_dollars, stop_distance, atr_qty, max_qty_by_pct, qty,
            )
        else:
            qty = max_qty_by_pct

        return max(0.0, round(qty, 6))

    # ── Trade approval ────────────────────────────────────────────────────────

    def approve_buy(
        self, ticker: str, price: float, atr: float = 0.0
    ) -> tuple[bool, str, float]:
        """
        Decide whether to open a new position.
        Returns (approved, reason, qty).
        qty is in base units (shares or crypto coins).
        """
        self._reset_daily_pnl_if_needed()

        # 1. Circuit breaker
        active, cb_reason = self._circuit_breaker_active()
        if active:
            return False, cb_reason, 0.0

        # 2. Already holding this ticker
        if ticker in self.positions:
            return False, f"Already holding {ticker}", 0.0

        # 3. Max open positions
        if self.open_position_count >= config.MAX_OPEN_POSITIONS:
            return False, f"Max open positions ({config.MAX_OPEN_POSITIONS}) reached", 0.0

        # 4. Daily loss limit (rough estimate without live prices)
        if self.cash > 0:
            daily_loss_ratio = -self.daily_pnl / self.cash
            if daily_loss_ratio >= config.MAX_DAILY_LOSS_PCT:
                return False, (
                    f"Daily loss limit reached "
                    f"(${self.daily_pnl:.2f}, -{daily_loss_ratio*100:.1f}%)"
                ), 0.0

        # 5. Price validity
        if price <= 0:
            return False, "Invalid price (≤ 0)", 0.0

        # 6. Calculate qty
        qty = self._calculate_qty(price, atr)
        if qty * price < 0.01:
            return False, f"Insufficient cash (${self.cash:.2f}) for minimum order", 0.0

        return True, "OK", qty

    def approve_sell(self, ticker: str, from_signal: bool = False) -> tuple[bool, str]:
        """
        Approve a sell.
        *from_signal* = True applies the minimum hold time check.
        Stop-loss and take-profit exits bypass the hold time check.
        """
        if ticker not in self.positions:
            return False, f"No open position for {ticker}"

        if from_signal and config.MIN_HOLD_MINUTES > 0:
            pos = self.positions[ticker]
            opened_at_str = pos.get("opened_at")
            if opened_at_str:
                try:
                    opened_at = datetime.fromisoformat(opened_at_str)
                    min_hold = timedelta(minutes=config.MIN_HOLD_MINUTES)
                    elapsed = datetime.now(timezone.utc) - opened_at
                    if elapsed < min_hold:
                        remaining_m = int((min_hold - elapsed).total_seconds() // 60)
                        return False, (
                            f"Min hold time not met for {ticker} "
                            f"({remaining_m} min remaining)"
                        )
                except Exception:
                    pass  # can't parse timestamp — allow sell

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
            "trailing_high": price,
            "trailing_stop_active": False,
        }
        self._save_state()
        log.info(
            "Position opened: %s  qty=%.6f  price=%.4f  value=$%.2f  broker=%s",
            ticker, qty, price, qty * price, broker,
        )

    def record_sell(self, ticker: str, qty: float, price: float) -> float:
        """Close position, update P&L and circuit breaker. Returns realised P&L."""
        pos = self.positions.get(ticker)
        if pos is None:
            log.warning("record_sell: no position found for %s", ticker)
            return 0.0

        pnl = (price - pos["avg_price"]) * qty
        self._state["cash"] += qty * price
        self._state["daily_pnl"] = self._state.get("daily_pnl", 0.0) + pnl
        del self._state["positions"][ticker]

        self._update_circuit_breaker(pnl)
        self._save_state()

        pct = (price - pos["avg_price"]) / pos["avg_price"] * 100 if pos["avg_price"] else 0
        log.info(
            "Position closed: %s  qty=%.6f  entry=%.4f  exit=%.4f  "
            "pnl=$%.2f (%+.2f%%)  consecutive_losses=%d",
            ticker, qty, pos["avg_price"], price, pnl, pct,
            self.consecutive_losses,
        )
        return pnl

    # ── Stop-loss / take-profit / trailing stop ───────────────────────────────

    def check_exit_conditions(self, prices: dict[str, float]) -> list[dict]:
        """
        Scan open positions against current prices.
        Returns list of {ticker, reason, qty, price} for positions to close.

        Checks (in order):
          1. Trailing stop (if active and price dropped from peak)
          2. Fixed stop-loss
          3. Fixed take-profit
        """
        to_close: list[dict] = []
        state_dirty = False  # track whether trailing_high changed

        for ticker, pos in list(self.positions.items()):
            current = prices.get(ticker)
            if current is None or current <= 0:
                continue

            avg = pos["avg_price"]
            if avg <= 0:
                continue

            change_pct = (current - avg) / avg
            trailing_high = pos.get("trailing_high", avg)
            trailing_active = pos.get("trailing_stop_active", False)

            # ── Update trailing high ──────────────────────────────────────────
            if current > trailing_high:
                pos["trailing_high"] = current
                trailing_high = current
                state_dirty = True

            # ── Activate trailing stop ────────────────────────────────────────
            if (
                config.USE_TRAILING_STOP
                and not trailing_active
                and change_pct >= config.TRAILING_STOP_ACTIVATION_PCT
            ):
                pos["trailing_stop_active"] = True
                trailing_active = True
                state_dirty = True
                log.info(
                    "Trailing stop activated for %s — peak=%.4f  gain=%.1f%%",
                    ticker, trailing_high, change_pct * 100,
                )

            # ── Trailing stop trigger ─────────────────────────────────────────
            if config.USE_TRAILING_STOP and trailing_active:
                drop_from_peak = (trailing_high - current) / trailing_high
                if drop_from_peak >= config.TRAILING_STOP_TRAIL_PCT:
                    to_close.append({
                        "ticker": ticker,
                        "reason": (
                            f"trailing-stop (peak={trailing_high:.4f}, "
                            f"drop={drop_from_peak*100:.1f}%)"
                        ),
                        "qty": pos["qty"],
                        "price": current,
                    })
                    log.warning(
                        "TRAILING STOP triggered: %s  peak=%.4f  current=%.4f  "
                        "drop=%.1f%%  overall=%.1f%%",
                        ticker, trailing_high, current,
                        drop_from_peak * 100, change_pct * 100,
                    )
                    continue  # don't also check fixed SL/TP for this ticker

            # ── Fixed stop-loss ───────────────────────────────────────────────
            if change_pct <= -config.STOP_LOSS_PCT:
                to_close.append({
                    "ticker": ticker,
                    "reason": f"stop-loss ({change_pct*100:.1f}%)",
                    "qty": pos["qty"],
                    "price": current,
                })
                log.warning(
                    "STOP-LOSS triggered: %s  entry=%.4f  current=%.4f  Δ%.1f%%",
                    ticker, avg, current, change_pct * 100,
                )
                continue

            # ── Fixed take-profit ─────────────────────────────────────────────
            if change_pct >= config.TAKE_PROFIT_PCT:
                to_close.append({
                    "ticker": ticker,
                    "reason": f"take-profit ({change_pct*100:.1f}%)",
                    "qty": pos["qty"],
                    "price": current,
                })
                log.info(
                    "TAKE-PROFIT triggered: %s  entry=%.4f  current=%.4f  Δ%.1f%%",
                    ticker, avg, current, change_pct * 100,
                )

        if state_dirty:
            self._save_state()

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
                "trailing_stop_active": pos.get("trailing_stop_active", False),
                "trailing_high": pos.get("trailing_high", pos["avg_price"]),
            })
        cb_active, cb_reason = self._circuit_breaker_active()
        return {
            "cash": round(self.cash, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "open_positions": pos_summary,
            "portfolio_value": round(self.portfolio_value(prices), 2),
            "consecutive_losses": self.consecutive_losses,
            "circuit_breaker_active": cb_active,
            "circuit_breaker_reason": cb_reason,
        }
