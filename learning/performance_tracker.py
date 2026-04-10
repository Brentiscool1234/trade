"""
Performance tracker — appends every closed trade to `trade_history.json`.

Each record captures the signal conditions that drove the entry so the
learning engine can later audit what worked and what didn't.

Record schema
-------------
{
  "ticker":           "NVDA",
  "entry_price":      482.30,
  "exit_price":       523.10,
  "qty":              0.0518,
  "pnl":              2.11,
  "pnl_pct":          8.46,
  "exit_reason":      "take-profit",
  "entry_composite":  0.47,
  "entry_sentiment":  0.61,
  "entry_technical":  0.29,
  "entry_trend":      "uptrend",
  "entry_sources":    ["news", "reddit/r/wallstreetbets"],
  "hold_hours":       6.2,
  "closed_at":        "2025-01-15T16:30:00+00:00"
}
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

from utils.logger import get_logger

log = get_logger(__name__)

_FILE = "trade_history.json"


def _load() -> list[dict]:
    if not os.path.exists(_FILE):
        return []
    try:
        with open(_FILE) as f:
            return json.load(f)
    except Exception as exc:
        log.warning("Could not load trade_history.json (%s)", exc)
        return []


def _save(history: list[dict]) -> None:
    try:
        with open(_FILE, "w") as f:
            json.dump(history, f, indent=2, default=str)
    except Exception as exc:
        log.warning("Could not save trade_history.json: %s", exc)


# ── Public write API ──────────────────────────────────────────────────────────

def record(
    ticker: str,
    entry_price: float,
    exit_price: float,
    qty: float,
    pnl: float,
    exit_reason: str,
    entry_composite: float = 0.0,
    entry_sentiment: float = 0.0,
    entry_technical: float = 0.0,
    entry_trend: str = "sideways",
    entry_sources: Optional[list[str]] = None,
    hold_hours: float = 0.0,
) -> None:
    """Append one closed-trade record."""
    pnl_pct = (exit_price - entry_price) / entry_price * 100 if entry_price else 0.0
    row = {
        "ticker": ticker,
        "entry_price": round(entry_price, 6),
        "exit_price": round(exit_price, 6),
        "qty": round(qty, 6),
        "pnl": round(pnl, 4),
        "pnl_pct": round(pnl_pct, 4),
        "exit_reason": exit_reason,
        "entry_composite": round(entry_composite, 4),
        "entry_sentiment": round(entry_sentiment, 4),
        "entry_technical": round(entry_technical, 4),
        "entry_trend": entry_trend,
        "entry_sources": entry_sources or [],
        "hold_hours": round(hold_hours, 2),
        "closed_at": datetime.now(timezone.utc).isoformat(),
    }
    history = _load()
    history.append(row)
    _save(history)
    log.debug(
        "Trade recorded: %s  pnl=$%.2f (%+.2f%%)  reason=%s",
        ticker, pnl, pnl_pct, exit_reason,
    )


# ── Public read API ───────────────────────────────────────────────────────────

def all_trades() -> list[dict]:
    return _load()


def recent(n: int = 50) -> list[dict]:
    history = _load()
    return history[-n:] if len(history) > n else history


def total_count() -> int:
    return len(_load())


def win_rate(last_n: int = 20) -> Optional[float]:
    trades = recent(last_n)
    if not trades:
        return None
    return sum(1 for t in trades if t["pnl"] > 0) / len(trades)


def avg_pnl_pct(last_n: int = 20) -> Optional[float]:
    trades = recent(last_n)
    if not trades:
        return None
    return sum(t["pnl_pct"] for t in trades) / len(trades)


def trades_by_ticker(ticker: str, last_n: int = 30) -> list[dict]:
    return [t for t in recent(last_n) if t["ticker"] == ticker]
