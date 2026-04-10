"""
Learning engine — analyses closed-trade history and rewrites strategy
parameters so the bot improves over time.

How it works
------------
Every time a position closes, `on_trade_closed()` is called with the full
trade context.  After every LEARN_EVERY_N_TRADES trades (default 10), the
engine runs a full optimisation cycle:

  1. Weight optimisation
     Compares how often the *sentiment* signal predicted the right direction
     vs how often the *technical* signal did.  Adjusts SENTIMENT_WEIGHT and
     TECHNICAL_WEIGHT proportionally, using an EMA so the bot doesn't
     overreact to a single good/bad stretch.

  2. Threshold optimisation
     Buckets trades by their entry composite score (low / mid / high).
     If trades entered at the lowest score band are consistently losing,
     the BUY threshold is raised.  If all bands are profitable, it's eased.

  3. Source-credibility scoring
     Tracks per-source (news / reddit / twitter) win rates.  Sources that
     produce more accurate signals get a higher multiplier in the sentiment
     aggregation; unreliable sources get down-weighted.

  4. Ticker blacklisting
     If a ticker loses money on ≥ 4 of its last 5 trades, it is added to a
     temporary blacklist.  The blacklist is cleared for that ticker after 7
     days so it gets another chance.

After any parameter change, a Discord notification is sent showing exactly
what changed and why, so the operator is always informed.

Minimum trades before learning starts: LEARN_AFTER_N_TRADES (default 10).
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

import notifications.discord as discord
from learning import params, performance_tracker as tracker
from utils.logger import get_logger

log = get_logger(__name__)

# ── Tuning constants ──────────────────────────────────────────────────────────

LEARN_AFTER_N_TRADES = 10      # minimum trades before first optimisation run
LEARN_EVERY_N_TRADES = 10      # re-optimise after every N additional trades
ANALYSIS_WINDOW = 40           # trades to consider per learning cycle

EMA_ALPHA = 0.35               # learning rate — how fast params shift each cycle
                               # 0 = never change, 1 = always jump to new value

WEIGHT_MIN = 0.30              # floor: neither signal can be ignored entirely
WEIGHT_MAX = 0.70              # ceiling: neither signal can fully dominate

BUY_THRESHOLD_MIN = 0.28       # never lower the bar below this
BUY_THRESHOLD_MAX = 0.58       # never raise the bar above this
THRESHOLD_STEP = 0.03          # max single-cycle threshold adjustment

SOURCE_WEIGHT_MIN = 0.25       # unreliable sources get at least this weight
SOURCE_WEIGHT_MAX = 2.5        # excellent sources can get up to 2.5×

BLACKLIST_MIN_TRADES = 3       # need at least 3 trades per ticker to judge
BLACKLIST_LOSS_RATE = 0.75     # blacklist if ≥ 75% of recent trades are losses
BLACKLIST_LOOKBACK = 8         # look at this many recent trades per ticker
BLACKLIST_EXPIRY_DAYS = 7      # auto-remove from blacklist after N days

MIN_TRADES_FOR_SOURCE = 4      # min trades per source before adjusting its weight


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ema_blend(current: float, target: float, alpha: float) -> float:
    """Exponential moving average: new = (1-α)*current + α*target."""
    return (1.0 - alpha) * current + alpha * target


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ── Core optimisation functions ───────────────────────────────────────────────

def _optimise_weights(trades: list[dict]) -> dict | None:
    """
    Return updated {sentiment_weight, technical_weight} or None if inconclusive.

    A signal is "correct" if it pointed in the same direction as the final P&L.
    E.g.: entry_sentiment > 0.1 AND pnl > 0  → sentiment was correct.
    """
    sent_correct = sent_total = tech_correct = tech_total = 0

    for t in trades:
        s = t.get("entry_sentiment", 0.0)
        c = t.get("entry_technical", 0.0)
        won = t["pnl"] > 0

        if abs(s) > 0.05:
            sent_total += 1
            if (s > 0) == won:
                sent_correct += 1

        if abs(c) > 0.05:
            tech_total += 1
            if (c > 0) == won:
                tech_correct += 1

    if sent_total < 3 or tech_total < 3:
        log.debug("Not enough directional trades to optimise weights (%d sent, %d tech)",
                  sent_total, tech_total)
        return None

    sent_acc = sent_correct / sent_total
    tech_acc = tech_correct / tech_total
    total_acc = sent_acc + tech_acc

    if total_acc < 0.05:
        return None

    target_sent = sent_acc / total_acc
    target_tech = 1.0 - target_sent

    current_sent = params.sentiment_weight()
    new_sent = _clamp(_ema_blend(current_sent, target_sent, EMA_ALPHA), WEIGHT_MIN, WEIGHT_MAX)
    new_tech = round(1.0 - new_sent, 4)
    new_sent = round(new_sent, 4)

    if abs(new_sent - current_sent) < 0.005:
        return None   # negligible change — skip

    return {
        "sentiment_weight": new_sent,
        "technical_weight": new_tech,
        "_reason_detail": (
            f"sentiment accuracy {sent_acc:.0%} ({sent_correct}/{sent_total}) | "
            f"technical accuracy {tech_acc:.0%} ({tech_correct}/{tech_total})"
        ),
    }


def _optimise_thresholds(trades: list[dict]) -> dict | None:
    """
    Return updated {buy_threshold} or None if unchanged.

    Trades are bucketed by entry composite score:
      low  : score in [current_threshold, current_threshold + 0.10)
      mid  : score in [current_threshold + 0.10, current_threshold + 0.20)
      high : score ≥ current_threshold + 0.20

    If the low band's win rate is poor (< 42%), raise the threshold.
    If the low band is strong (> 62%) and there are enough samples, lower it.
    """
    current_thr = params.buy_threshold()
    low_band_wins: list[bool] = []
    mid_band_wins: list[bool] = []

    for t in trades:
        score = t.get("entry_composite", 0.0)
        if score <= 0:   # sell-driven trades
            continue
        won = t["pnl"] > 0
        if current_thr <= score < current_thr + 0.10:
            low_band_wins.append(won)
        elif current_thr + 0.10 <= score < current_thr + 0.20:
            mid_band_wins.append(won)

    if len(low_band_wins) < 5:
        log.debug("Not enough low-band trades to optimise threshold (%d)", len(low_band_wins))
        return None

    low_wr = _mean([float(w) for w in low_band_wins])
    new_thr = current_thr

    if low_wr < 0.42:
        # Low-conviction trades are losing more than winning → raise the bar
        new_thr = _clamp(current_thr + THRESHOLD_STEP, BUY_THRESHOLD_MIN, BUY_THRESHOLD_MAX)
        detail = f"low-band win rate {low_wr:.0%} ({sum(low_band_wins)}/{len(low_band_wins)} wins)"
    elif low_wr > 0.62 and len(low_band_wins) >= 8:
        # Low-conviction trades are reliably profitable → ease the bar slightly
        new_thr = _clamp(current_thr - THRESHOLD_STEP, BUY_THRESHOLD_MIN, BUY_THRESHOLD_MAX)
        detail = f"low-band win rate {low_wr:.0%} ({sum(low_band_wins)}/{len(low_band_wins)} wins)"
    else:
        return None

    new_thr = round(new_thr, 3)
    if abs(new_thr - current_thr) < 0.001:
        return None

    return {"buy_threshold": new_thr, "_reason_detail": detail}


def _optimise_source_weights(trades: list[dict]) -> dict | None:
    """
    Return updated source_weights or None if no changes are warranted.
    Sources are scored by the fraction of trades that used them and were profitable.
    """
    source_wins: dict[str, list[bool]] = defaultdict(list)

    for t in trades:
        won = t["pnl"] > 0
        for src in t.get("entry_sources", []):
            # Normalise: 'reddit/r/stocks' → 'reddit'
            key = src.split("/")[0]
            source_wins[key].append(won)

    current_weights: dict = dict(params.get("source_weights", {}))
    new_weights = dict(current_weights)
    changed = False

    for src, results in source_wins.items():
        if len(results) < MIN_TRADES_FOR_SOURCE:
            continue
        accuracy = _mean([float(w) for w in results])
        # Map accuracy → weight: 50% → ~1.0, 70% → ~1.4, 30% → ~0.6
        target_weight = accuracy * 2.0
        current_w = current_weights.get(src, 1.0)
        new_w = _clamp(_ema_blend(current_w, target_weight, EMA_ALPHA),
                       SOURCE_WEIGHT_MIN, SOURCE_WEIGHT_MAX)
        new_w = round(new_w, 3)
        if abs(new_w - current_w) > 0.02:
            new_weights[src] = new_w
            changed = True

    if not changed:
        return None

    # Normalise so average weight stays near 1.0 (doesn't inflate overall sentiment)
    if new_weights:
        avg = _mean(list(new_weights.values()))
        if avg > 0:
            new_weights = {k: round(v / avg, 3) for k, v in new_weights.items()}

    return {"source_weights": new_weights}


def _update_blacklist(all_recent: list[dict]) -> tuple[list[str], list[str]]:
    """
    Return (tickers_added, tickers_removed) to the blacklist.
    """
    ticker_trades: dict[str, list[dict]] = defaultdict(list)
    for t in all_recent:
        ticker_trades[t["ticker"]].append(t)

    current_blacklist: list = list(params.get("ticker_blacklist", []))
    ticker_notes: dict = dict(params.get("ticker_notes", {}))
    now = datetime.now(timezone.utc)
    added: list[str] = []
    removed: list[str] = []

    # Check for expiry (auto-remove after BLACKLIST_EXPIRY_DAYS)
    for ticker in list(current_blacklist):
        note = ticker_notes.get(ticker, {})
        bl_at_str = note.get("blacklisted_at") if isinstance(note, dict) else None
        if bl_at_str:
            try:
                bl_at = datetime.fromisoformat(bl_at_str)
                if now - bl_at > timedelta(days=BLACKLIST_EXPIRY_DAYS):
                    current_blacklist.remove(ticker)
                    removed.append(ticker)
                    log.info("Ticker %s removed from blacklist (expired after %d days)",
                             ticker, BLACKLIST_EXPIRY_DAYS)
            except Exception:
                pass

    # Check for new additions
    for ticker, trades in ticker_trades.items():
        if len(trades) < BLACKLIST_MIN_TRADES:
            continue
        recent_trades = trades[-BLACKLIST_LOOKBACK:]
        loss_rate = sum(1 for t in recent_trades if t["pnl"] < 0) / len(recent_trades)
        if loss_rate >= BLACKLIST_LOSS_RATE and ticker not in current_blacklist:
            current_blacklist.append(ticker)
            ticker_notes[ticker] = {
                "blacklisted_at": now.isoformat(),
                "reason": f"loss rate {loss_rate:.0%} over last {len(recent_trades)} trades",
            }
            added.append(ticker)
            log.warning("Ticker %s added to blacklist: loss_rate=%.0f%%", ticker, loss_rate * 100)

    params.update(
        {"ticker_blacklist": current_blacklist, "ticker_notes": ticker_notes},
        reason="" if not added and not removed else f"blacklist: +{added} -{removed}",
    )
    return added, removed


# ── Main engine class ─────────────────────────────────────────────────────────

class LearningEngine:
    """
    Orchestrates the full learning cycle.
    Instantiate once in main.py; call on_trade_closed() after every position closes.
    """

    def __init__(self) -> None:
        total = tracker.total_count()
        log.info(
            "LearningEngine ready — %d trade(s) in history, "
            "next optimisation at trade #%d",
            total,
            self._next_learn_at(total),
        )

    @staticmethod
    def _next_learn_at(current_total: int) -> int:
        if current_total < LEARN_AFTER_N_TRADES:
            return LEARN_AFTER_N_TRADES
        steps = (current_total - LEARN_AFTER_N_TRADES) // LEARN_EVERY_N_TRADES
        return LEARN_AFTER_N_TRADES + (steps + 1) * LEARN_EVERY_N_TRADES

    # ── Public interface ──────────────────────────────────────────────────────

    def on_trade_closed(
        self,
        ticker: str,
        entry_price: float,
        exit_price: float,
        qty: float,
        pnl: float,
        exit_reason: str,
        signal_meta: Optional[dict] = None,
        hold_hours: float = 0.0,
    ) -> None:
        """
        Called by main.py every time a position closes.
        Records the trade, then fires a learning cycle if enough have accumulated.
        """
        meta = signal_meta or {}
        tracker.record(
            ticker=ticker,
            entry_price=entry_price,
            exit_price=exit_price,
            qty=qty,
            pnl=pnl,
            exit_reason=exit_reason,
            entry_composite=meta.get("composite_score", 0.0),
            entry_sentiment=meta.get("sentiment_score", 0.0),
            entry_technical=meta.get("technical_score", 0.0),
            entry_trend=meta.get("trend_direction", "sideways"),
            entry_sources=meta.get("sources", []),
            hold_hours=hold_hours,
        )

        total = tracker.total_count()
        params.update(
            {"total_trades_seen": total, "win_rate": tracker.win_rate(20)},
            reason="",  # don't log routine counters as change events
        )

        if total >= LEARN_AFTER_N_TRADES and (total - LEARN_AFTER_N_TRADES) % LEARN_EVERY_N_TRADES == 0:
            log.info("Learning threshold reached (%d trades) — running optimisation", total)
            self.run_learning_cycle()

    def run_learning_cycle(self) -> None:
        """
        Full optimisation pass.  Can also be called manually (e.g. from a schedule).
        """
        trades = tracker.recent(ANALYSIS_WINDOW)
        if len(trades) < LEARN_AFTER_N_TRADES:
            log.info("Too few trades for learning (%d < %d)", len(trades), LEARN_AFTER_N_TRADES)
            return

        log.info("=== LEARNING CYCLE START (%d trades analysed) ===", len(trades))
        changes: list[str] = []

        # 1. Signal weights
        w = _optimise_weights(trades)
        if w:
            detail = w.pop("_reason_detail", "")
            params.update(w, reason=f"weight optimisation — {detail}")
            changes.append(
                f"Weights → sent={w['sentiment_weight']:.3f}  tech={w['technical_weight']:.3f}"
                f" ({detail})"
            )

        # 2. Buy threshold
        t = _optimise_thresholds(trades)
        if t:
            detail = t.pop("_reason_detail", "")
            old_thr = params.buy_threshold()
            params.update(t, reason=f"threshold optimisation — {detail}")
            direction = "raised" if t["buy_threshold"] > old_thr else "lowered"
            changes.append(
                f"Buy threshold {direction} {old_thr:.3f} → {t['buy_threshold']:.3f}"
                f" ({detail})"
            )

        # 3. Source weights
        sw = _optimise_source_weights(trades)
        if sw:
            params.update(sw, reason="source credibility update")
            changes.append(
                "Source weights updated → " +
                "  ".join(f"{k}={v:.2f}" for k, v in sw["source_weights"].items())
            )

        # 4. Blacklist
        added, removed = _update_blacklist(trades)
        if added:
            changes.append(f"Blacklisted: {added}")
        if removed:
            changes.append(f"Un-blacklisted (expired): {removed}")

        wr = tracker.win_rate(20)
        avg_pnl = tracker.avg_pnl_pct(20)

        log.info(
            "=== LEARNING CYCLE END ===  %d changes  win_rate=%.0f%%  avg_pnl=%.2f%%",
            len(changes), (wr or 0) * 100, avg_pnl or 0,
        )

        # Send Discord report whenever something actually changed
        if changes:
            discord.learning_report(
                win_rate=wr,
                avg_pnl_pct=avg_pnl,
                total_trades=tracker.total_count(),
                changes=changes,
                current_params={
                    "sentiment_weight": params.sentiment_weight(),
                    "technical_weight": params.technical_weight(),
                    "buy_threshold": params.buy_threshold(),
                    "source_weights": params.get("source_weights", {}),
                    "blacklist": params.get("ticker_blacklist", []),
                },
            )
