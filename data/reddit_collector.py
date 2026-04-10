"""
Reddit collector — scrapes hot/new posts from financial subreddits using PRAW.
Upvote score is used to weight mentions (popular posts matter more).
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

# Common English words that look like tickers — filtered out to reduce noise
_FALSE_POSITIVES = {
    "A", "I", "IT", "IS", "BE", "DO", "GO", "SO", "OR", "AT", "TO",
    "BY", "AN", "AS", "IF", "IN", "OF", "ON", "UP", "US", "WE",
    "CEO", "CFO", "CTO", "IPO", "ETF", "SEC", "FDA", "GDP", "CPI",
    "ALL", "NEW", "NOW", "ONE", "TWO", "FOR", "HAS", "ARE", "WAS",
    "THE", "AND", "BUT", "NOT", "YOU", "HIM", "HER", "ITS", "OUR",
    "OUT", "CAN", "DID", "GET", "GOT", "HAD", "HER", "HIM", "HIS",
    "HOW", "LET", "MAY", "OWN", "PUT", "RAN", "RUN", "SAW", "SAY",
    "SHE", "TOO", "USE", "WHO", "WHY", "YES", "YET",
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


def _score_to_weight(score: int) -> float:
    """Convert upvote score to a 0.5–2.0 credibility weight."""
    if score <= 0:
        return 0.5
    if score >= 10_000:
        return 2.0
    # log scale between 0.5 and 2.0
    import math
    return 0.5 + 1.5 * (math.log10(score + 1) / math.log10(10_001))


class RedditCollector:
    def __init__(self) -> None:
        self._reddit: Optional[object] = None
        self._available = False
        self._setup()

    def _setup(self) -> None:
        if not all([config.REDDIT_CLIENT_ID, config.REDDIT_CLIENT_SECRET]):
            log.warning("Reddit credentials not set — Reddit collector disabled")
            return
        try:
            import praw
            self._reddit = praw.Reddit(
                client_id=config.REDDIT_CLIENT_ID,
                client_secret=config.REDDIT_CLIENT_SECRET,
                user_agent=config.REDDIT_USER_AGENT,
            )
            # Verify connectivity with a lightweight call
            self._reddit.subreddit("stocks").id  # noqa: B018
            self._available = True
            log.info("Reddit collector ready")
        except ImportError:
            log.warning("praw not installed — run: pip install praw")
        except Exception as exc:
            log.error("Reddit init failed: %s", exc)

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_stock_mentions(self) -> list[dict]:
        return self._fetch_subreddits(config.STOCK_SUBREDDITS)

    def fetch_crypto_mentions(self) -> list[dict]:
        return self._fetch_subreddits(config.CRYPTO_SUBREDDITS)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _fetch_subreddits(self, subreddits: list[str]) -> list[dict]:
        if not self._available:
            return []
        mentions: list[dict] = []
        for sub_name in subreddits:
            try:
                sub = self._reddit.subreddit(sub_name)
                posts = list(sub.hot(limit=config.REDDIT_POST_LIMIT))
                for post in posts:
                    text = f"{post.title}. {post.selftext[:500]}"
                    tickers = _extract_tickers(text)
                    if not tickers:
                        continue
                    weight = _score_to_weight(post.score)
                    for ticker in tickers:
                        mentions.append({
                            "ticker": ticker,
                            "text": text,
                            "source": f"reddit/r/{sub_name}",
                            "published_at": str(post.created_utc),
                            "weight": weight,
                        })
            except Exception as exc:
                log.warning("Reddit fetch error [r/%s]: %s", sub_name, exc)
        log.debug("Reddit: fetched %d mentions from %s", len(mentions), subreddits)
        return mentions
