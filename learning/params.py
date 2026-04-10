"""
Runtime parameter store — learned overlay on top of config defaults.

Loaded automatically from `learned_params.json` at import time.
Updated by `learning/engine.py` as the bot gains experience.

All strategy code (signals/generator.py, analysis/sentiment.py) reads the
adaptive parameters from here instead of from the static config module.
This means the bot silently re-tunes itself every time enough trades close.

Adaptive parameters
-------------------
  sentiment_weight      float  0.55 default   how much to trust sentiment signals
  technical_weight      float  0.45 default   how much to trust technical signals
  buy_threshold         float  0.35 default   min composite score to open a long
  sell_threshold        float -0.25 default   max composite score to close a long
  source_weights        dict   all 1.0        per-source credibility multiplier
  ticker_blacklist      list   []             tickers to skip entirely
  ticker_notes          dict   {}             per-ticker memos from the engine

Meta
----
  version           int     increments on every save (audit trail)
  total_trades_seen int     lifetime closed-trade count used for learning
  win_rate          float   rolling win rate across last 20 trades
  change_log        list    last N parameter-change events with reasoning
"""

from __future__ import annotations

import json
import os
from typing import Any

import config
from utils.logger import get_logger

log = get_logger(__name__)

_FILE = "learned_params.json"
_MAX_CHANGE_LOG = 50          # keep last 50 change-log entries


def _defaults() -> dict:
    return {
        "sentiment_weight": config.SENTIMENT_WEIGHT,
        "technical_weight": config.TECHNICAL_WEIGHT,
        "buy_threshold": config.BUY_SIGNAL_THRESHOLD,
        "sell_threshold": config.SELL_SIGNAL_THRESHOLD,
        "source_weights": {
            "news": 1.0,
            "reddit": 1.0,
            "twitter": 1.0,
        },
        "ticker_blacklist": [],
        "ticker_notes": {},
        "version": 0,
        "total_trades_seen": 0,
        "win_rate": None,
        "change_log": [],
    }


# ── Module-level mutable store ────────────────────────────────────────────────

_store: dict = {}


def _load() -> None:
    global _store
    if os.path.exists(_FILE):
        try:
            with open(_FILE) as f:
                loaded = json.load(f)
            # Merge: loaded values win, defaults fill missing keys
            _store = {**_defaults(), **loaded}
            wr = _store.get("win_rate")
            log.info(
                "Learned params loaded (v%d) — win_rate=%s  "
                "sent=%.3f  tech=%.3f  buy_thr=%.3f",
                _store["version"],
                f"{wr:.2f}" if wr is not None else "n/a",
                _store["sentiment_weight"],
                _store["technical_weight"],
                _store["buy_threshold"],
            )
            return
        except Exception as exc:
            log.warning("Could not load learned_params.json (%s) — using defaults", exc)
    _store = _defaults()


def save() -> None:
    try:
        with open(_FILE, "w") as f:
            json.dump(_store, f, indent=2, default=str)
    except Exception as exc:
        log.warning("Could not save learned_params.json: %s", exc)


def get(key: str, default: Any = None) -> Any:
    return _store.get(key, default)


def update(updates: dict, reason: str = "") -> None:
    """Apply *updates* to the store, log the change, and persist."""
    from datetime import datetime, timezone

    _store.update(updates)
    _store["version"] = _store.get("version", 0) + 1

    if reason:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "changes": {
                k: round(v, 5) if isinstance(v, float) else v
                for k, v in updates.items()
            },
        }
        log_list: list = _store.setdefault("change_log", [])
        log_list.append(entry)
        if len(log_list) > _MAX_CHANGE_LOG:
            _store["change_log"] = log_list[-_MAX_CHANGE_LOG:]

    save()
    log.info(
        "Learned params updated (v%d): %s | reason: %s",
        _store["version"],
        {k: round(v, 4) if isinstance(v, float) else v for k, v in updates.items()},
        reason or "—",
    )


# ── Convenience accessors ─────────────────────────────────────────────────────

def sentiment_weight() -> float:
    return float(_store.get("sentiment_weight", config.SENTIMENT_WEIGHT))


def technical_weight() -> float:
    return float(_store.get("technical_weight", config.TECHNICAL_WEIGHT))


def buy_threshold() -> float:
    return float(_store.get("buy_threshold", config.BUY_SIGNAL_THRESHOLD))


def sell_threshold() -> float:
    return float(_store.get("sell_threshold", config.SELL_SIGNAL_THRESHOLD))


def source_weight(source: str) -> float:
    """
    Learned credibility multiplier for *source*.
    'source' can be a full string like 'reddit/r/stocks' — matched by prefix.
    """
    weights: dict = _store.get("source_weights", {})
    for key, w in weights.items():
        if source.startswith(key):
            return float(w)
    return 1.0


def is_blacklisted(ticker: str) -> bool:
    return ticker in _store.get("ticker_blacklist", [])


def recent_changes(n: int = 5) -> list[dict]:
    log_list = _store.get("change_log", [])
    return log_list[-n:]


# Load on import
_load()
