"""
Binance trader — executes paper/testnet trades for crypto.

Uses python-binance with BINANCE_TESTNET=true pointing to
https://testnet.binance.vision (free paper trading).

Crypto quantities are stored in base asset (e.g. BTC, ETH).
Orders use USDT as the quote asset.
"""

from __future__ import annotations

from typing import Optional

import config
from trading.risk_manager import RiskManager
from utils.logger import get_logger

log = get_logger(__name__)

_BROKER_NAME = "binance"

# Minimum order sizes (USDT notional) per pair — below these Binance rejects orders
_MIN_NOTIONAL = 10.0    # most pairs require ≥ $10 USDT

# Precision map: how many decimal places for qty of each coin
_QTY_PRECISION: dict[str, int] = {
    "BTC": 5, "ETH": 4, "SOL": 2, "ADA": 0, "DOGE": 0,
    "AVAX": 2, "LINK": 2, "DOT": 2, "MATIC": 0, "UNI": 2,
}


def _binance_symbol(ticker: str) -> str:
    return f"{ticker}{config.BINANCE_QUOTE_ASSET}"


def _round_qty(ticker: str, qty: float) -> float:
    precision = _QTY_PRECISION.get(ticker, 4)
    factor = 10 ** precision
    return float(int(qty * factor)) / factor


class BinanceTrader:
    def __init__(self, risk_manager: RiskManager) -> None:
        self._rm = risk_manager
        self._client: Optional[object] = None
        self._available = False
        self._setup()

    def _setup(self) -> None:
        if not all([config.BINANCE_API_KEY, config.BINANCE_SECRET_KEY]):
            log.warning(
                "BINANCE_API_KEY / BINANCE_SECRET_KEY not set — "
                "Binance trader running in DRY_RUN mode"
            )
            return
        try:
            from binance.client import Client
            self._client = Client(
                api_key=config.BINANCE_API_KEY,
                api_secret=config.BINANCE_SECRET_KEY,
                testnet=config.BINANCE_TESTNET,
            )
            # Quick connectivity check
            self._client.ping()
            acct = self._client.get_account()
            usdt_balance = next(
                (float(b["free"]) for b in acct["balances"] if b["asset"] == "USDT"),
                0.0,
            )
            log.info(
                "Binance %s connected — USDT balance=%.2f",
                "testnet" if config.BINANCE_TESTNET else "live",
                usdt_balance,
            )
            self._available = True
        except ImportError:
            log.warning("python-binance not installed — run: pip install python-binance")
        except Exception as exc:
            log.error("Binance init failed: %s", exc)

    # ── Current price ─────────────────────────────────────────────────────────

    def get_price(self, ticker: str) -> Optional[float]:
        symbol = _binance_symbol(ticker)
        if self._available:
            try:
                ticker_data = self._client.get_symbol_ticker(symbol=symbol)
                return float(ticker_data["price"])
            except Exception as exc:
                log.debug("Binance price failed for %s: %s", symbol, exc)
        # Fallback: yfinance
        return self._yf_price(ticker)

    def _yf_price(self, ticker: str) -> Optional[float]:
        try:
            import yfinance as yf
            symbol = f"{ticker}-USD"
            data = yf.download(symbol, period="1d", interval="1m", progress=False, auto_adjust=True)
            if not data.empty:
                return float(data["Close"].iloc[-1])
        except Exception:
            pass
        return None

    # ── Trade execution ───────────────────────────────────────────────────────

    def buy(self, ticker: str, price: Optional[float] = None, atr: float = 0.0) -> bool:
        price = price or self.get_price(ticker)
        if not price:
            log.warning("Cannot buy %s — price unavailable", ticker)
            return False

        approved, reason, qty_raw = self._rm.approve_buy(ticker, price, atr=atr)
        if not approved:
            log.info("BUY rejected [%s]: %s", ticker, reason)
            return False

        qty = _round_qty(ticker, qty_raw)
        notional = qty * price

        if notional < _MIN_NOTIONAL:
            log.info(
                "BUY skipped [%s]: notional $%.2f below minimum $%.2f",
                ticker, notional, _MIN_NOTIONAL,
            )
            return False

        if config.DRY_RUN:
            log.info("DRY_RUN BUY: %s  qty=%.6f  price=%.2f", ticker, qty, price)
            self._rm.record_buy(ticker, qty, price, _BROKER_NAME)
            return True

        if not self._available:
            log.warning(
                "Binance not connected — simulating BUY %s qty=%.6f price=%.2f",
                ticker, qty, price,
            )
            self._rm.record_buy(ticker, qty, price, _BROKER_NAME)
            return True

        try:
            symbol = _binance_symbol(ticker)
            order = self._client.create_order(
                symbol=symbol,
                side="BUY",
                type="MARKET",
                quantity=qty,
            )
            self._rm.record_buy(ticker, qty, price, _BROKER_NAME)
            log.info(
                "BUY ORDER submitted: %s  qty=%.6f  ~$%.2f  order_id=%s",
                ticker, qty, notional, order["orderId"],
            )
            return True
        except Exception as exc:
            log.error("Binance BUY failed for %s: %s", ticker, exc)
            return False

    def sell(self, ticker: str, price: Optional[float] = None, reason: str = "signal") -> bool:
        from_signal = reason == "signal"
        approved, msg = self._rm.approve_sell(ticker, from_signal=from_signal)
        if not approved:
            log.info("SELL rejected [%s]: %s", ticker, msg)
            return False

        pos = self._rm.positions.get(ticker, {})
        qty = _round_qty(ticker, pos.get("qty", 0))
        price = price or self.get_price(ticker) or pos.get("avg_price", 0)

        if config.DRY_RUN:
            pnl = self._rm.record_sell(ticker, qty, price)
            log.info(
                "DRY_RUN SELL: %s  qty=%.6f  price=%.2f  pnl=%.2f  reason=%s",
                ticker, qty, price, pnl, reason,
            )
            return True

        if not self._available:
            pnl = self._rm.record_sell(ticker, qty, price)
            log.warning(
                "Binance not connected — simulating SELL %s pnl=%.2f  reason=%s",
                ticker, pnl, reason,
            )
            return True

        try:
            symbol = _binance_symbol(ticker)
            order = self._client.create_order(
                symbol=symbol,
                side="SELL",
                type="MARKET",
                quantity=qty,
            )
            pnl = self._rm.record_sell(ticker, qty, price)
            log.info(
                "SELL ORDER submitted: %s  qty=%.6f  ~$%.2f  pnl=%.2f  reason=%s",
                ticker, qty, qty * price, pnl, reason,
            )
            return True
        except Exception as exc:
            log.error("Binance SELL failed for %s: %s", ticker, exc)
            return False
