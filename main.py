"""
Trading Bot — main entry point.

Run:
    python main.py

The bot runs two independent scan loops:
  - Stock loop  : every STOCK_SCAN_INTERVAL_MINUTES during NYSE market hours
  - Crypto loop : every CRYPTO_SCAN_INTERVAL_MINUTES (24/7)

Additionally, every hour it:
  - Checks all open positions for stop-loss / take-profit
  - Sends a portfolio summary to Discord

Stop with Ctrl-C.
"""

import sys
import time
from datetime import datetime, timezone

import pytz
import schedule

import config
import notifications.discord as discord
from data.news_collector import NewsCollector
from data.reddit_collector import RedditCollector
from data.twitter_collector import TwitterCollector
from signals.generator import generate_signals, Action
from trading.alpaca_trader import AlpacaTrader
from trading.binance_trader import BinanceTrader
from trading.risk_manager import RiskManager
from utils.logger import get_logger

log = get_logger(__name__)

# NYSE timezone for market hours detection
_NYSE_TZ = pytz.timezone("America/New_York")

# Crypto tickers (don't need market-hours check)
_CRYPTO_TICKERS = set(config.WATCHED_CRYPTO)


# ── Market hours ──────────────────────────────────────────────────────────────

def _is_market_open() -> bool:
    """Return True if NYSE is currently open (weekdays 09:30–16:00 ET)."""
    now_et = datetime.now(_NYSE_TZ)
    if now_et.weekday() >= 5:               # Saturday=5, Sunday=6
        return False
    open_h, open_m = 9, 30
    close_h, close_m = 16, 0
    t = now_et.time()
    from datetime import time as dtime
    return dtime(open_h, open_m) <= t < dtime(close_h, close_m)


# ── Bot orchestration ─────────────────────────────────────────────────────────

class TradingBot:
    def __init__(self) -> None:
        log.info("Initialising trading bot...")
        self.risk_manager = RiskManager()
        self.alpaca = AlpacaTrader(self.risk_manager)
        self.binance = BinanceTrader(self.risk_manager)

        self.news = NewsCollector()
        self.reddit = RedditCollector()
        self.twitter = TwitterCollector()

        log.info("All components initialised")
        discord.startup_message(dry_run=config.DRY_RUN)

    # ── Data collection ───────────────────────────────────────────────────────

    def _collect_stock_mentions(self) -> list[dict]:
        mentions: list[dict] = []
        mentions.extend(self.news.fetch_stock_mentions())
        mentions.extend(self.reddit.fetch_stock_mentions())
        mentions.extend(self.twitter.fetch_stock_mentions())
        log.info("Collected %d stock mentions from all sources", len(mentions))
        return mentions

    def _collect_crypto_mentions(self) -> list[dict]:
        mentions: list[dict] = []
        mentions.extend(self.news.fetch_crypto_mentions())
        mentions.extend(self.reddit.fetch_crypto_mentions())
        mentions.extend(self.twitter.fetch_crypto_mentions())
        log.info("Collected %d crypto mentions from all sources", len(mentions))
        return mentions

    # ── Trade execution ───────────────────────────────────────────────────────

    def _execute_signal(self, signal, is_crypto: bool) -> None:
        ticker = signal.ticker
        trader = self.binance if is_crypto else self.alpaca
        price = signal.current_price or None

        if signal.action == Action.BUY:
            log.info(
                "Executing BUY signal: %s  score=%.3f  sent=%.3f  tech=%.3f  "
                "trend=%s  atr=%.4f  notes=%s",
                ticker, signal.composite_score,
                signal.sentiment_score, signal.technical_score,
                signal.trend_direction, signal.atr, signal.notes or "—",
            )
            success = trader.buy(ticker, price, atr=signal.atr)
            if success:
                pos = self.risk_manager.positions.get(ticker, {})
                actual_price = pos.get("avg_price", price or 0)
                actual_qty = pos.get("qty", 0)
                discord.trade_opened(
                    ticker=ticker,
                    qty=actual_qty,
                    price=actual_price,
                    composite_score=signal.composite_score,
                    sources=signal.sources,
                    broker="binance" if is_crypto else "alpaca",
                )

        elif signal.action == Action.SELL:
            if ticker not in self.risk_manager.positions:
                return  # nothing to sell
            log.info(
                "Executing SELL signal: %s  score=%.3f",
                ticker, signal.composite_score,
            )
            pos = self.risk_manager.positions[ticker]
            entry = pos.get("avg_price", 0)
            qty = pos.get("qty", 0)
            success = trader.sell(ticker, price, reason="signal")
            if success:
                exit_price = price or entry
                pnl = (exit_price - entry) * qty
                discord.trade_closed(
                    ticker=ticker,
                    qty=qty,
                    entry_price=entry,
                    exit_price=exit_price,
                    pnl=pnl,
                    reason="sell-signal",
                )

    # ── Position monitoring (stop-loss / take-profit) ─────────────────────────

    def _check_positions(self) -> None:
        """Collect current prices for all held positions and enforce SL/TP."""
        if not self.risk_manager.positions:
            return

        prices: dict[str, float] = {}
        for ticker in list(self.risk_manager.positions.keys()):
            is_crypto = ticker in _CRYPTO_TICKERS
            trader = self.binance if is_crypto else self.alpaca
            p = trader.get_price(ticker)
            if p:
                prices[ticker] = p

        exits = self.risk_manager.check_exit_conditions(prices)
        for exit_order in exits:
            ticker = exit_order["ticker"]
            reason = exit_order["reason"]
            price = exit_order["price"]
            qty = exit_order["qty"]
            pos = self.risk_manager.positions.get(ticker, {})
            entry = pos.get("avg_price", price)

            is_crypto = ticker in _CRYPTO_TICKERS
            trader = self.binance if is_crypto else self.alpaca
            trader.sell(ticker, price, reason=reason)

            pnl = (price - entry) * qty
            if "stop-loss" in reason:
                discord.stop_loss_alert(ticker, entry, price, pnl)
            else:
                discord.trade_closed(ticker, qty, entry, price, pnl, reason)

    # ── Scan loops ────────────────────────────────────────────────────────────

    def run_stock_scan(self) -> None:
        if not _is_market_open():
            log.info("Stock scan skipped — NYSE is closed")
            return
        log.info("=== STOCK SCAN START ===")
        try:
            mentions = self._collect_stock_mentions()
            if not mentions:
                log.info("No stock mentions collected — skipping analysis")
                return
            signals = generate_signals(mentions)
            stock_signals = [s for s in signals if s.ticker not in _CRYPTO_TICKERS]
            log.info("Stock signals: %d actionable", len(stock_signals))
            for signal in stock_signals:
                self._execute_signal(signal, is_crypto=False)
        except Exception as exc:
            log.error("Stock scan error: %s", exc, exc_info=True)
            discord.error_alert(f"Stock scan error: {exc}")
        log.info("=== STOCK SCAN END ===")

    def run_crypto_scan(self) -> None:
        log.info("=== CRYPTO SCAN START ===")
        try:
            mentions = self._collect_crypto_mentions()
            if not mentions:
                log.info("No crypto mentions collected — skipping analysis")
                return
            signals = generate_signals(mentions)
            crypto_signals = [s for s in signals if s.ticker in _CRYPTO_TICKERS]
            log.info("Crypto signals: %d actionable", len(crypto_signals))
            for signal in crypto_signals:
                self._execute_signal(signal, is_crypto=True)
        except Exception as exc:
            log.error("Crypto scan error: %s", exc, exc_info=True)
            discord.error_alert(f"Crypto scan error: {exc}")
        log.info("=== CRYPTO SCAN END ===")

    def run_hourly_checks(self) -> None:
        log.info("Running hourly checks...")
        try:
            self._check_positions()
            summary = self.risk_manager.summary()
            discord.portfolio_summary(summary)
            log.info(
                "Portfolio — cash=$%.2f  daily_pnl=$%.2f  positions=%d  "
                "consec_losses=%d  circuit_breaker=%s",
                summary["cash"], summary["daily_pnl"],
                len(summary["open_positions"]),
                summary["consecutive_losses"],
                summary["circuit_breaker_active"],
            )
        except Exception as exc:
            log.error("Hourly check error: %s", exc, exc_info=True)

    # ── Scheduler setup ───────────────────────────────────────────────────────

    def start(self) -> None:
        log.info(
            "Bot starting — stocks every %d min (market hours) | "
            "crypto every %d min (24/7)",
            config.STOCK_SCAN_INTERVAL_MINUTES,
            config.CRYPTO_SCAN_INTERVAL_MINUTES,
        )

        schedule.every(config.STOCK_SCAN_INTERVAL_MINUTES).minutes.do(self.run_stock_scan)
        schedule.every(config.CRYPTO_SCAN_INTERVAL_MINUTES).minutes.do(self.run_crypto_scan)
        schedule.every(1).hour.do(self.run_hourly_checks)

        # Run immediately on startup so you don't wait 15 minutes for first signal
        log.info("Running initial scans on startup...")
        self.run_stock_scan()
        self.run_crypto_scan()
        self.run_hourly_checks()

        log.info("Entering scheduler loop — press Ctrl-C to stop")
        while True:
            try:
                schedule.run_pending()
                time.sleep(30)
            except KeyboardInterrupt:
                log.info("Bot stopped by user")
                discord.error_alert("Bot stopped manually (Ctrl-C)")
                sys.exit(0)
            except Exception as exc:
                log.error("Scheduler error: %s", exc, exc_info=True)
                discord.error_alert(f"Scheduler error: {exc}")
                time.sleep(60)  # back off before retrying


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    bot = TradingBot()
    bot.start()
