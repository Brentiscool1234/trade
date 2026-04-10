"""
Signal generator — combines sentiment and technical scores into a final
trading recommendation, with multi-layer safety filters.

Pipeline
--------
1. Receive raw mentions from data collectors.
2. Run sentiment aggregation  → SentimentResult per ticker.
3. Run technical analysis     → TechnicalResult per ticker.
4. Combine scores             → composite score.
5. Apply safety filters       → may downgrade BUY/SELL to HOLD.
6. Return actionable signals sorted by conviction.

Composite score formula
-----------------------
  composite = SENTIMENT_WEIGHT * sentiment.score
            + TECHNICAL_WEIGHT * technical.score

Safety filters (applied in order)
----------------------------------
  1. Confidence gate     — low sentiment confidence shifts weight to technical
  2. Confluence check    — both signals must agree in direction for entry
                           (REQUIRE_SIGNAL_CONFLUENCE in config)
  3. Trend filter        — BUY blocked when asset is in confirmed downtrend
                           unless composite > 0.60 (very high conviction)
                           (TREND_FILTER_ENABLED in config)
  4. Death Cross block   — BUY blocked if EMA50 just crossed below EMA200
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
    sentiment_score: float        # -1 to +1  (0 if no data)
    technical_score: float        # -1 to +1  (0 if no data)
    sentiment_confidence: float   # 0–1
    current_price: float
    atr: float                    # ATR for position sizing (0 if unavailable)
    trend_direction: str          # "uptrend" | "downtrend" | "sideways"
    mention_count: int
    sources: list[str] = field(default_factory=list)
    notes: str = ""               # pipe-separated filter notes, e.g. "trend-blocked"


# ── Core combination logic ────────────────────────────────────────────────────

def _combine(
    ticker: str,
    sentiment: Optional[SentimentResult],
    technical: Optional[TechnicalResult],
) -> Optional[TradeSignal]:
    """Build a TradeSignal for *ticker*, applying all safety filters."""

    if sentiment is None and technical is None:
        return None

    note_parts: list[str] = []

    # ── Step 1: raw composite score ───────────────────────────────────────────
    if sentiment is None:
        # No social data — technical only, half weight
        composite = technical.score * 0.5
        s_score, s_conf, mentions, sources = 0.0, 0.0, 0, []
        note_parts.append("no-sentiment")
    elif technical is None:
        # No price data — sentiment only, half weight
        composite = sentiment.score * 0.5
        s_score = sentiment.score
        s_conf = sentiment.confidence
        mentions = sentiment.mention_count
        sources = sentiment.sources
        note_parts.append("no-technical")
    else:
        composite = (
            config.SENTIMENT_WEIGHT * sentiment.score
            + config.TECHNICAL_WEIGHT * technical.score
        )
        s_score = sentiment.score
        s_conf = sentiment.confidence
        mentions = sentiment.mention_count
        sources = sentiment.sources

    composite = max(-1.0, min(1.0, composite))
    tech_score = technical.score if technical else 0.0
    price = technical.current_price if technical else 0.0
    atr = technical.atr if technical else 0.0
    trend_dir = technical.trend_direction if technical else "sideways"

    # ── Step 2: confidence gate ───────────────────────────────────────────────
    # When sentiment confidence is low, lean harder on technical data.
    if sentiment and technical and sentiment.confidence < 0.25:
        composite = 0.25 * sentiment.score + 0.75 * technical.score
        note_parts.append("low-sent-conf")

    # ── Step 3: determine raw action ─────────────────────────────────────────
    if composite >= config.BUY_SIGNAL_THRESHOLD:
        action = Action.BUY
    elif composite <= config.SELL_SIGNAL_THRESHOLD:
        action = Action.SELL
    else:
        action = Action.HOLD

    # ── Step 4: confluence filter ─────────────────────────────────────────────
    # Prevents entering when sentiment and technical disagree.
    if config.REQUIRE_SIGNAL_CONFLUENCE and sentiment and technical:
        if action == Action.BUY:
            both_bullish = sentiment.score > 0.05 and technical.score > 0.0
            if not both_bullish:
                log.debug(
                    "Confluence blocked BUY [%s]: sent=%.3f tech=%.3f",
                    ticker, sentiment.score, technical.score,
                )
                action = Action.HOLD
                note_parts.append("no-confluence-buy")

        elif action == Action.SELL:
            both_bearish = sentiment.score < -0.05 and technical.score < 0.0
            if not both_bearish:
                log.debug(
                    "Confluence blocked SELL [%s]: sent=%.3f tech=%.3f",
                    ticker, sentiment.score, technical.score,
                )
                action = Action.HOLD
                note_parts.append("no-confluence-sell")

    # ── Step 5: trend filter ──────────────────────────────────────────────────
    # Block buys when the asset is in a confirmed downtrend.
    # Only an extremely strong signal (composite > 0.60) can override.
    if config.TREND_FILTER_ENABLED and technical and action == Action.BUY:
        if trend_dir == "downtrend":
            if composite < 0.60:
                log.debug(
                    "Trend filter blocked BUY [%s]: downtrend  composite=%.3f",
                    ticker, composite,
                )
                action = Action.HOLD
                note_parts.append("trend-blocked")
            else:
                note_parts.append("trend-override")  # logged but allowed

    # ── Step 6: Death Cross block ─────────────────────────────────────────────
    if technical and technical.death_cross and action == Action.BUY:
        log.info("Death Cross blocking BUY for %s", ticker)
        action = Action.HOLD
        note_parts.append("death-cross")

    # Golden Cross — boost log visibility (does not change score)
    if technical and technical.golden_cross and action == Action.BUY:
        note_parts.append("golden-cross")

    return TradeSignal(
        ticker=ticker,
        action=action,
        composite_score=round(composite, 4),
        sentiment_score=round(s_score, 4),
        technical_score=round(tech_score, 4),
        sentiment_confidence=round(s_conf, 3),
        current_price=price,
        atr=atr,
        trend_direction=trend_dir,
        mention_count=mentions,
        sources=sources,
        notes="|".join(note_parts),
    )


# ── Public API ────────────────────────────────────────────────────────────────

def generate_signals(mentions: list[dict]) -> list[TradeSignal]:
    """
    Full pipeline: raw mentions → filtered, actionable TradeSignal list.
    Returns only BUY and SELL signals, sorted by |composite_score| descending.
    """
    if not mentions:
        log.info("No mentions to process")
        return []

    # Step 1: sentiment per ticker
    sentiment_results = aggregate_sentiment(mentions)
    log.info("Sentiment computed for %d ticker(s)", len(sentiment_results))

    # Step 2: technical for tickers that cleared the mention threshold
    technical_results: dict[str, TechnicalResult] = {}
    for ticker in sentiment_results:
        r = technical_analyse(ticker)
        if r:
            technical_results[ticker] = r
    log.info("Technical analysis computed for %d ticker(s)", len(technical_results))

    # Step 3: combine and filter
    signals: list[TradeSignal] = []
    for ticker in set(sentiment_results) | set(technical_results):
        sig = _combine(
            ticker,
            sentiment_results.get(ticker),
            technical_results.get(ticker),
        )
        if sig is None:
            continue
        signals.append(sig)
        log.debug(
            "Signal [%s]: %s  composite=%.3f  sent=%.3f  tech=%.3f  trend=%s  notes=%s",
            ticker, sig.action.value, sig.composite_score,
            sig.sentiment_score, sig.technical_score,
            sig.trend_direction, sig.notes or "—",
        )

    # Step 4: return only actionable signals
    actionable = [s for s in signals if s.action != Action.HOLD]
    actionable.sort(key=lambda s: abs(s.composite_score), reverse=True)

    blocked = len(signals) - len(actionable)
    log.info(
        "Signals: %d total | %d actionable (BUY/SELL) | %d blocked by safety filters",
        len(signals), len(actionable), blocked,
    )
    return actionable
