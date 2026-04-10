"""
News collector — fetches financial headlines from NewsAPI.
Returns a list of Mention dicts: {ticker, text, source, published_at, weight}

Weight reflects source credibility:
  - Major financial outlets → 1.5
  - General news           → 1.0
"""

import re
import time
from datetime import datetime, timezone
from typing import Optional

import config
from utils.logger import get_logger

log = get_logger(__name__)

# Credibility multiplier per news domain
_PREMIUM_DOMAINS = {
    "reuters.com", "bloomberg.com", "wsj.com", "ft.com",
    "cnbc.com", "marketwatch.com", "barrons.com", "seekingalpha.com",
    "coindesk.com", "cointelegraph.com", "theblock.co",
}

# Pre-compile ticker patterns
_DOLLAR_TICKER = re.compile(r"\$([A-Z]{1,5})\b")
_CRYPTO_NAMES = {
    "bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL",
    "cardano": "ADA", "dogecoin": "DOGE", "avalanche": "AVAX",
    "chainlink": "LINK", "polkadot": "DOT", "polygon": "MATIC",
    "uniswap": "UNI",
}

_ALL_WATCHED = set(config.WATCHED_STOCKS + config.WATCHED_CRYPTO)


def _extract_tickers(text: str) -> list[str]:
    """Return unique tickers mentioned in *text*."""
    found: set[str] = set()

    # $TICKER pattern
    for m in _DOLLAR_TICKER.finditer(text):
        t = m.group(1)
        if t in _ALL_WATCHED:
            found.add(t)

    # Bare uppercase words that match watched stocks
    for word in re.findall(r"\b([A-Z]{2,5})\b", text):
        if word in _ALL_WATCHED:
            found.add(word)

    # Crypto name synonyms
    lower = text.lower()
    for name, symbol in _CRYPTO_NAMES.items():
        if name in lower:
            found.add(symbol)

    return list(found)


def _domain_weight(url: str) -> float:
    for domain in _PREMIUM_DOMAINS:
        if domain in url:
            return 1.5
    return 1.0


class NewsCollector:
    """Thin wrapper around the NewsAPI /everything endpoint."""

    def __init__(self) -> None:
        self._client: Optional[object] = None
        self._available = False
        self._setup()

    def _setup(self) -> None:
        if not config.NEWS_API_KEY:
            log.warning("NEWS_API_KEY not set — news collector disabled")
            return
        try:
            from newsapi import NewsApiClient
            self._client = NewsApiClient(api_key=config.NEWS_API_KEY)
            self._available = True
            log.info("NewsAPI collector ready")
        except ImportError:
            log.warning("newsapi-python not installed — run: pip install newsapi-python")
        except Exception as exc:
            log.error("NewsAPI init failed: %s", exc)

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_stock_mentions(self) -> list[dict]:
        return self._fetch(
            q="stock market OR earnings OR NYSE OR NASDAQ",
            label="stocks",
        )

    def fetch_crypto_mentions(self) -> list[dict]:
        return self._fetch(
            q="bitcoin OR ethereum OR cryptocurrency OR crypto",
            label="crypto",
        )

    # ── Internals ─────────────────────────────────────────────────────────────

    def _fetch(self, q: str, label: str) -> list[dict]:
        if not self._available:
            return []
        try:
            resp = self._client.get_everything(
                q=q,
                language="en",
                sort_by="publishedAt",
                page_size=50,
            )
            articles = resp.get("articles", [])
            mentions: list[dict] = []
            for art in articles:
                title = art.get("title") or ""
                description = art.get("description") or ""
                text = f"{title}. {description}"
                tickers = _extract_tickers(text)
                if not tickers:
                    continue
                url = art.get("url") or ""
                weight = _domain_weight(url)
                pub = art.get("publishedAt") or ""
                for ticker in tickers:
                    mentions.append({
                        "ticker": ticker,
                        "text": text,
                        "source": "news",
                        "published_at": pub,
                        "weight": weight,
                    })
            log.debug("NewsAPI [%s]: fetched %d mentions", label, len(mentions))
            return mentions
        except Exception as exc:
            log.error("NewsAPI fetch error [%s]: %s", label, exc)
            return []
