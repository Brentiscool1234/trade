"""
Twitter / X collector — searches recent tweets using Tweepy v2.
Requires at minimum the Basic API tier ($100/mo) for search access.
If TWITTER_BEARER_TOKEN is absent the collector is gracefully disabled.

Weight is uniform (1.0) since we have no reliable engagement signal
from the free/basic search endpoint.
"""

import re
from typing import Optional

import config
from utils.logger import get_logger

log = get_logger(__name__)

_DOLLAR_TICKER = re.compile(r"\$([A-Z]{1,5})\b")
_CRYPTO_NAMES = {
    "bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL",
    "cardano": "ADA", "dogecoin": "DOGE", "avalanche": "AVAX",
    "chainlink": "LINK", "polkadot": "DOT", "polygon": "MATIC",
    "uniswap": "UNI",
}

_ALL_WATCHED = set(config.WATCHED_STOCKS + config.WATCHED_CRYPTO)
_FALSE_POSITIVES = {
    "A", "I", "IT", "IS", "BE", "DO", "GO", "SO", "OR", "AT", "TO",
    "BY", "AN", "AS", "IF", "IN", "OF", "ON", "UP", "US", "WE",
    "CEO", "CFO", "CTO", "IPO", "ETF", "SEC", "FDA", "GDP", "CPI",
    "ALL", "NEW", "NOW", "ONE", "TWO", "FOR",
}


def _extract_tickers(text: str) -> list[str]:
    found: set[str] = set()
    for m in _DOLLAR_TICKER.finditer(text):
        t = m.group(1)
        if t in _ALL_WATCHED and t not in _FALSE_POSITIVES:
            found.add(t)
    for word in re.findall(r"\b([A-Z]{2,5})\b", text):
        if word in _ALL_WATCHED and word not in _FALSE_POSITIVES:
            found.add(word)
    lower = text.lower()
    for name, symbol in _CRYPTO_NAMES.items():
        if name in lower:
            found.add(symbol)
    return list(found)


class TwitterCollector:
    def __init__(self) -> None:
        self._client: Optional[object] = None
        self._available = False
        self._setup()

    def _setup(self) -> None:
        if not config.TWITTER_BEARER_TOKEN:
            log.warning(
                "TWITTER_BEARER_TOKEN not set — Twitter collector disabled "
                "(bot still runs on news + Reddit)"
            )
            return
        try:
            import tweepy
            self._client = tweepy.Client(
                bearer_token=config.TWITTER_BEARER_TOKEN,
                wait_on_rate_limit=True,
            )
            self._available = True
            log.info("Twitter/X collector ready")
        except ImportError:
            log.warning("tweepy not installed — run: pip install tweepy")
        except Exception as exc:
            log.error("Twitter init failed: %s", exc)

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_stock_mentions(self) -> list[dict]:
        return self._run_queries(config.STOCK_TWITTER_QUERIES, label="stocks")

    def fetch_crypto_mentions(self) -> list[dict]:
        return self._run_queries(config.CRYPTO_TWITTER_QUERIES, label="crypto")

    # ── Internals ─────────────────────────────────────────────────────────────

    def _run_queries(self, queries: list[str], label: str) -> list[dict]:
        if not self._available:
            return []
        mentions: list[dict] = []
        for query in queries:
            try:
                response = self._client.search_recent_tweets(
                    query=query,
                    max_results=config.TWITTER_MAX_RESULTS,
                    tweet_fields=["created_at", "text"],
                )
                if not response or not response.data:
                    continue
                for tweet in response.data:
                    tickers = _extract_tickers(tweet.text)
                    if not tickers:
                        continue
                    for ticker in tickers:
                        mentions.append({
                            "ticker": ticker,
                            "text": tweet.text,
                            "source": "twitter",
                            "published_at": str(tweet.created_at),
                            "weight": 1.0,
                        })
            except Exception as exc:
                log.warning("Twitter fetch error [%s]: %s", label, exc)
        log.debug("Twitter [%s]: fetched %d mentions", label, len(mentions))
        return mentions
