"""
Signal generator — combines sentiment and technical scores into
a final trading recommendation.

Pipeline
--------
1. Receive all raw mentions from data collectors.
2. Run sentiment aggregation  → SentimentResult per ticker.
3. Run technical analysis     → TechnicalResult per ticker.
4. Combine scores             → TradeSignal per ticker.
5. Filter by thresholds       → actionable BUY / SELL / HOLD signals.

Composite score formula
-----------------------
  composite = SENTIMENT_WEIGHT * sentiment.score
            + TECHNICAL_WEIGHT * technical.score

  where SENTIMENT_WEIGHT = 0.55, TECHNICAL_WEIGHT = 0.45  (from config)

  composite > BUY_SIGNAL_THRESHOLD  (0.35)  → BUY
  composite < SELL_SIGNAL_THRESHOLD (-0.25) → SELL
  otherwise                                 → HOLD
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import config
from analysis.sentiment import SentimentResult, aggregate_sentiment
from analysis.technical import TechnicalResult, analyse as technical_analyse
from utils.logger import get_logger

log = get_logger(__name__)


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradeSignal:
    ticker: str
    action: Action
    composite_score: float        # -1 to +1
    sentiment_score: float        # -1 to +1  (0 if no sentiment data)
    technical_score: float        # -1 to +1  (0 if no technical data)
    sentiment_confidence: float   # 0–1
    current_price: float
    mention_count: int
    sources: list[str] = field(default_factory=list)
    notes: str = ""


# ── Core logic ────────────────────────────────────────────────────────────────

def _combine(
    ticker: str,
    sentiment: Optional[SentimentResult],
    technical: Optional[TechnicalResult],
) -> Optional[TradeSignal]:
    """
    Build a TradeSignal for *ticker*.
    Returns None if neither sentiment nor technical data is available.
    """
    # Require at least one source
    if sentiment is None and technical is None:
        return None

    # If only one source is available, reduce the composite weight
    if sentiment is None:
        composite = technical.score * 0.5   # half-weight — limited conviction
        s_score, s_conf, mentions, sources = 0.0, 0.0, 0, []
        notes = "no-sentiment"
    elif technical is None:
        composite = sentiment.score * 0.5
        s_score = sentiment.score
        s_conf = sentiment.confidence
        mentions = sentiment.mention_count
        sources = sentiment.sources
        notes = "no-technical"
    else:
        composite = (
            config.SENTIMENT_WEIGHT * sentiment.score
            + config.TECHNICAL_WEIGHT * technical.score
        )
        s_score = sentiment.score
        s_conf = sentiment.confidence
        mentions = sentiment.mention_count
        sources = sentiment.sources
        notes = ""

    composite = max(-1.0, min(1.0, composite))
    tech_score = technical.score if technical else 0.0
    price = technical.current_price if technical else 0.0

    # Confidence gate: low-confidence sentiment should dampen the signal
    if sentiment and sentiment.confidence < 0.2 and technical:
        # Shift weight toward technical when sentiment is weak
        composite = 0.3 * sentiment.score + 0.7 * technical.score
        notes = "low-sentiment-confidence"

    # Action decision
    if composite >= config.BUY_SIGNAL_THRESHOLD:
        action = Action.BUY
    elif composite <= config.SELL_SIGNAL_THRESHOLD:
        action = Action.SELL
    else:
        action = Action.HOLD

    return TradeSignal(
        ticker=ticker,
        action=action,
        composite_score=round(composite, 4),
        sentiment_score=round(s_score, 4),
        technical_score=round(tech_score, 4),
        sentiment_confidence=round(s_conf, 3),
        current_price=price,
        mention_count=mentions,
        sources=sources,
        notes=notes,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def generate_signals(mentions: list[dict]) -> list[TradeSignal]:
    """
    Full pipeline: raw mentions → actionable TradeSignal list.

    *mentions* is the combined output of all data collectors:
      [{"ticker": "AAPL", "text": "...", "source": "...", "weight": 1.0, ...}, ...]

    Returns only BUY and SELL signals (HOLD signals are filtered out).
    """
    if not mentions:
        log.info("No mentions to process")
        return []

    # Step 1: sentiment per ticker
    sentiment_results = aggregate_sentiment(mentions)
    log.info("Sentiment computed for %d ticker(s)", len(sentiment_results))

    # Step 2: technical analysis for tickers that passed the mention threshold
    all_tickers = list(sentiment_results.keys())
    # Also include fully watched tickers even if below mention threshold
    # (they'll have no sentiment data but can generate technical-only signals
    #  at reduced conviction)
    technical_results: dict[str, TechnicalResult] = {}
    for ticker in all_tickers:
        r = technical_analyse(ticker)
        if r:
            technical_results[ticker] = r

    log.info("Technical analysis computed for %d ticker(s)", len(technical_results))

    # Step 3: combine
    signals: list[TradeSignal] = []
    all_candidate_tickers = set(sentiment_results) | set(technical_results)

    for ticker in all_candidate_tickers:
        sig = _combine(
            ticker,
            sentiment_results.get(ticker),
            technical_results.get(ticker),
        )
        if sig is None:
            continue
        signals.append(sig)
        log.debug(
            "Signal [%s]: %s  composite=%.3f  sent=%.3f  tech=%.3f",
            ticker, sig.action.value, sig.composite_score,
            sig.sentiment_score, sig.technical_score,
        )

    # Step 4: return only actionable signals, sorted by conviction
    actionable = [s for s in signals if s.action != Action.HOLD]
    actionable.sort(key=lambda s: abs(s.composite_score), reverse=True)

    log.info(
        "Signals generated: %d total, %d actionable (BUY/SELL)",
        len(signals), len(actionable),
    )
    return actionable
