"""
Central configuration — loads .env and exposes all bot settings as constants.
All trading parameters can be adjusted here without touching logic files.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Broker credentials ──────────────────────────────────────────────────────
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")
BINANCE_TESTNET = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

# ── Data-source credentials ──────────────────────────────────────────────────
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "TradingBot/1.0")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")

# ── Notifications ────────────────────────────────────────────────────────────
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# ── Feature flags ────────────────────────────────────────────────────────────
USE_FINBERT = os.getenv("USE_FINBERT", "false").lower() == "true"
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"

# ── Risk management ──────────────────────────────────────────────────────────
# Maximum fraction of portfolio allocated to a single trade
MAX_POSITION_SIZE_PCT = 0.025       # 2.5 %
# Stop-loss: close position if it loses this much
STOP_LOSS_PCT = 0.05                # 5 %
# Take-profit: close position if it gains this much
TAKE_PROFIT_PCT = 0.15              # 15 %
# Halt all trading if daily P&L drops below this fraction
MAX_DAILY_LOSS_PCT = 0.10           # 10 %
# Maximum number of simultaneously open positions (stocks + crypto combined)
MAX_OPEN_POSITIONS = 5

# ── Signal thresholds ────────────────────────────────────────────────────────
BUY_SIGNAL_THRESHOLD = 0.35         # composite score must exceed this to buy
SELL_SIGNAL_THRESHOLD = -0.25       # composite score must drop below this to sell

# Weights for combining sentiment and technical signals (must sum to 1.0)
SENTIMENT_WEIGHT = 0.55
TECHNICAL_WEIGHT = 0.45

# Minimum number of social/news mentions before a ticker is considered
MIN_MENTION_COUNT = 3

# ── Scheduling ───────────────────────────────────────────────────────────────
STOCK_SCAN_INTERVAL_MINUTES = 15    # run during NYSE market hours only
CRYPTO_SCAN_INTERVAL_MINUTES = 15   # runs 24/7

# ── Subreddits to monitor ────────────────────────────────────────────────────
STOCK_SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "investing",
    "SecurityAnalysis",
    "StockMarket",
]
CRYPTO_SUBREDDITS = [
    "CryptoCurrency",
    "Bitcoin",
    "ethereum",
    "CryptoMarkets",
    "altcoin",
]
REDDIT_POST_LIMIT = 50              # posts to fetch per subreddit per scan

# ── Twitter search queries ───────────────────────────────────────────────────
STOCK_TWITTER_QUERIES = [
    "stocks OR $SPY OR $QQQ lang:en -is:retweet",
    "stock market OR trading lang:en -is:retweet",
]
CRYPTO_TWITTER_QUERIES = [
    "$BTC OR $ETH OR bitcoin OR ethereum lang:en -is:retweet",
    "crypto OR cryptocurrency lang:en -is:retweet",
]
TWITTER_MAX_RESULTS = 50            # tweets per query per scan (10–100)

# ── Watched tickers ──────────────────────────────────────────────────────────
# The bot will also discover tickers dynamically from social/news mentions.
WATCHED_STOCKS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD",
    "SPY", "QQQ", "NFLX", "PLTR", "COIN", "SOFI", "RIVN",
]
WATCHED_CRYPTO = [
    "BTC", "ETH", "SOL", "ADA", "DOGE", "AVAX", "LINK", "DOT", "MATIC", "UNI",
]

# Binance symbol format for crypto (appended with USDT)
BINANCE_QUOTE_ASSET = "USDT"

# ── Data / state file paths ──────────────────────────────────────────────────
PORTFOLIO_STATE_FILE = "portfolio_state.json"
TRADE_LOG_FILE = "trades.log"
