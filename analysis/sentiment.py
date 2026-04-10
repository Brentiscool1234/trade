"""
Sentiment analysis engine.

Default (always available): VADER — lexicon-based, zero model download, fast.
Optional upgrade:           FinBERT — transformer trained on financial text,
                            ~440 MB download, needs `pip install transformers torch`.
                            Enable with USE_FINBERT=true in .env

Both backends return a score in [-1, +1]:
  -1 = very negative   0 = neutral   +1 = very positive

Public function
---------------
aggregate_sentiment(mentions) -> dict[ticker, SentimentResult]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import config
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class SentimentResult:
    ticker: str
    score: float          # weighted average sentiment, -1 to +1
    confidence: float     # 0–1; higher with more mentions and source agreement
    mention_count: int
    sources: list[str]    # unique sources that contributed


# ── Backend initialisation ───────────────────────────────────────────────────

_vader_analyser = None
_finbert_pipeline = None


def _get_vader():
    global _vader_analyser
    if _vader_analyser is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _vader_analyser = SentimentIntensityAnalyzer()
            log.info("VADER sentiment analyser loaded")
        except ImportError:
            log.error("vaderSentiment not installed — run: pip install vaderSentiment")
    return _vader_analyser


def _get_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is None:
        try:
            from transformers import pipeline
            _finbert_pipeline = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                top_k=None,
            )
            log.info("FinBERT pipeline loaded")
        except ImportError:
            log.warning(
                "transformers/torch not installed — falling back to VADER. "
                "Run: pip install transformers torch"
            )
        except Exception as exc:
            log.warning("FinBERT load failed (%s) — falling back to VADER", exc)
    return _finbert_pipeline


# ── Scoring functions ─────────────────────────────────────────────────────────

def _vader_score(text: str) -> float:
    """Return compound score in [-1, +1] using VADER."""
    analyser = _get_vader()
    if analyser is None:
        return 0.0
    # Truncate to 512 chars — VADER is word-based so length is fine
    vs = analyser.polarity_scores(text[:1024])
    return vs["compound"]


def _finbert_score(text: str) -> float:
    """Return score in [-1, +1] using FinBERT (positive − negative confidence)."""
    pipe = _get_finbert()
    if pipe is None:
        return _vader_score(text)
    try:
        # FinBERT max token length is 512; truncate aggressively
        result = pipe(text[:512], truncation=True)[0]
        scores = {item["label"]: item["score"] for item in result}
        positive = scores.get("positive", 0.0)
        negative = scores.get("negative", 0.0)
        return positive - negative      # range: -1 to +1
    except Exception as exc:
        log.debug("FinBERT inference failed: %s — using VADER", exc)
        return _vader_score(text)


def _score_text(text: str) -> float:
    if config.USE_FINBERT:
        return _finbert_score(text)
    return _vader_score(text)


# ── Confidence calculation ────────────────────────────────────────────────────

def _compute_confidence(scores: list[float], mention_count: int) -> float:
    """
    Confidence grows with:
      - more mentions (log-scaled, capped at 1.0)
      - higher agreement between individual scores (low variance)
    """
    if not scores:
        return 0.0

    # Volume component: saturates around 20 mentions
    volume_conf = min(1.0, math.log10(mention_count + 1) / math.log10(21))

    # Agreement component: low std-dev → high agreement
    if len(scores) > 1:
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = math.sqrt(variance)
        agreement_conf = max(0.0, 1.0 - std)   # std in [0, 2] → conf in [-1, 1] clipped
    else:
        agreement_conf = 0.5    # single mention: medium confidence

    return round((volume_conf + agreement_conf) / 2, 3)


# ── Public API ────────────────────────────────────────────────────────────────

def aggregate_sentiment(mentions: list[dict]) -> dict[str, SentimentResult]:
    """
    Process a list of mention dicts (from data collectors) and return a
    mapping of ticker → SentimentResult.

    Each mention dict must have keys: ticker, text, source, weight.
    """
    # Group by ticker
    by_ticker: dict[str, list[dict]] = {}
    for m in mentions:
        ticker = m["ticker"]
        by_ticker.setdefault(ticker, []).append(m)

    results: dict[str, SentimentResult] = {}

    for ticker, ticker_mentions in by_ticker.items():
        if len(ticker_mentions) < config.MIN_MENTION_COUNT:
            log.debug(
                "Skipping %s — only %d mention(s) (min %d)",
                ticker, len(ticker_mentions), config.MIN_MENTION_COUNT,
            )
            continue

        weighted_scores: list[float] = []
        raw_scores: list[float] = []
        sources: set[str] = set()

        for m in ticker_mentions:
            raw = _score_text(m["text"])
            weight = float(m.get("weight", 1.0))
            weighted_scores.append(raw * weight)
            raw_scores.append(raw)
            sources.add(m.get("source", "unknown"))

        total_weight = sum(float(m.get("weight", 1.0)) for m in ticker_mentions)
        avg_score = sum(weighted_scores) / total_weight if total_weight else 0.0
        avg_score = max(-1.0, min(1.0, avg_score))     # clamp

        confidence = _compute_confidence(raw_scores, len(ticker_mentions))

        results[ticker] = SentimentResult(
            ticker=ticker,
            score=round(avg_score, 4),
            confidence=confidence,
            mention_count=len(ticker_mentions),
            sources=sorted(sources),
        )
        log.debug(
            "Sentiment [%s]: score=%.3f  conf=%.2f  mentions=%d  sources=%s",
            ticker, avg_score, confidence, len(ticker_mentions), sorted(sources),
        )

    return results
