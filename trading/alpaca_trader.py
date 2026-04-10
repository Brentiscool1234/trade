"""
Alpaca trader — executes paper trades for US equities via the Alpaca API.

Uses alpaca-py (the official modern SDK).
Paper trading endpoint: https://paper-api.alpaca.markets

Alpaca supports fractional shares, so qty can be a float.
"""

from __future__ import annotations

from typing import Optional

import config
from trading.risk_manager import RiskManager
from utils.logger import get_logger

log = get_logger(__name__)

_BROKER_NAME = "alpaca"


class AlpacaTrader:
    def __init__(self, risk_manager: RiskManager) -> None:
        self._rm = risk_manager
        self._client: Optional[object] = None
        self._available = False
        self._setup()

    def _setup(self) -> None:
        if not all([config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY]):
            log.warning(
                "ALPACA_API_KEY / ALPACA_SECRET_KEY not set — "
                "Alpaca trader running in DRY_RUN mode"
            )
            return
        try:
            from alpaca.trading.client import TradingClient
            self._client = TradingClient(
                api_key=config.ALPACA_API_KEY,
                secret_key=config.ALPACA_SECRET_KEY,
                paper=True,         # always paper trading
            )
            account = self._client.get_account()
            log.info(
                "Alpaca connected — paper account cash=$%.2f",
                float(account.cash),
            )
            self._available = True
        except ImportError:
            log.warning("alpaca-py not installed — run: pip install alpaca-py")
        except Exception as exc:
            log.error("Alpaca init failed: %s", exc)

    # ── Current price ─────────────────────────────────────────────────────────

    def get_price(self, ticker: str) -> Optional[float]:
        """Fetch latest trade price from Alpaca market data."""
        if not self._available:
            return self._yf_price(ticker)
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestTradeRequest
            data_client = StockHistoricalDataClient(
                api_key=config.ALPACA_API_KEY,
                secret_key=config.ALPACA_SECRET_KEY,
            )
            req = StockLatestTradeRequest(symbol_or_symbols=ticker)
            resp = data_client.get_stock_latest_trade(req)
            return float(resp[ticker].price)
        except Exception as exc:
            log.debug("Alpaca price fetch failed for %s: %s — falling back to yfinance", ticker, exc)
            return self._yf_price(ticker)

    def _yf_price(self, ticker: str) -> Optional[float]:
        try:
            import yfinance as yf
            data = yf.download(ticker, period="1d", interval="1m", progress=False, auto_adjust=True)
            if not data.empty:
                return float(data["Close"].iloc[-1])
        except Exception:
            pass
        return None

    # ── Trade execution ───────────────────────────────────────────────────────

    def buy(self, ticker: str, price: Optional[float] = None) -> bool:
        """
        Submit a fractional-share market BUY order.
        Returns True if the order was submitted (or logged in DRY_RUN).
        """
        price = price or self.get_price(ticker)
        if not price:
            log.warning("Cannot buy %s — price unavailable", ticker)
            return False

        approved, reason, qty = self._rm.approve_buy(ticker, price)
        if not approved:
            log.info("BUY rejected [%s]: %s", ticker, reason)
            return False

        if config.DRY_RUN:
            log.info("DRY_RUN BUY: %s  qty=%.4f  price=%.2f", ticker, qty, price)
            self._rm.record_buy(ticker, qty, price, _BROKER_NAME)
            return True

        if not self._available:
            log.warning(
                "Alpaca not connected — simulating BUY %s qty=%.4f price=%.2f",
                ticker, qty, price,
            )
            self._rm.record_buy(ticker, qty, price, _BROKER_NAME)
            return True

        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            order_req = MarketOrderRequest(
                symbol=ticker,
                notional=round(qty * price, 2),   # dollar-value order (fractional)
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            order = self._client.submit_order(order_req)
            self._rm.record_buy(ticker, qty, price, _BROKER_NAME)
            log.info(
                "BUY ORDER submitted: %s  qty=%.4f  ~$%.2f  order_id=%s",
                ticker, qty, qty * price, order.id,
            )
            return True
        except Exception as exc:
            log.error("Alpaca BUY failed for %s: %s", ticker, exc)
            return False

    def sell(self, ticker: str, price: Optional[float] = None, reason: str = "signal") -> bool:
        """Close the position for *ticker*."""
        approved, msg = self._rm.approve_sell(ticker)
        if not approved:
            log.info("SELL rejected [%s]: %s", ticker, msg)
            return False

        pos = self._rm.positions.get(ticker, {})
        qty = pos.get("qty", 0)
        price = price or self.get_price(ticker) or pos.get("avg_price", 0)

        if config.DRY_RUN:
            pnl = self._rm.record_sell(ticker, qty, price)
            log.info(
                "DRY_RUN SELL: %s  qty=%.4f  price=%.2f  pnl=%.2f  reason=%s",
                ticker, qty, price, pnl, reason,
            )
            return True

        if not self._available:
            pnl = self._rm.record_sell(ticker, qty, price)
            log.warning(
                "Alpaca not connected — simulating SELL %s pnl=%.2f  reason=%s",
                ticker, pnl, reason,
            )
            return True

        try:
            self._client.close_position(ticker)
            pnl = self._rm.record_sell(ticker, qty, price)
            log.info(
                "SELL ORDER submitted: %s  qty=%.4f  ~$%.2f  pnl=%.2f  reason=%s",
                ticker, qty, qty * price, pnl, reason,
            )
            return True
        except Exception as exc:
            log.error("Alpaca SELL failed for %s: %s", ticker, exc)
            return False

    # ── Portfolio sync ────────────────────────────────────────────────────────

    def sync_positions(self) -> None:
        """Optionally sync local state with actual Alpaca positions."""
        if not self._available:
            return
        try:
            positions = self._client.get_all_positions()
            log.debug("Alpaca reports %d open position(s)", len(positions))
        except Exception as exc:
            log.debug("Position sync failed: %s", exc)
