"""
Discord notifier — sends rich-embed alerts to a Discord channel via webhook.

Message types
-------------
trade_opened   : sent when a BUY order is executed
trade_closed   : sent when a SELL order is executed (includes P&L)
stop_loss      : special red alert for stop-loss exits
take_profit    : green alert for take-profit exits
portfolio_summary : periodic snapshot of the portfolio
error          : unexpected errors worth surfacing

If DISCORD_WEBHOOK_URL is not set, all calls are no-ops (silent).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

import requests

import config
from utils.logger import get_logger

log = get_logger(__name__)

# Discord embed colours (decimal)
_COLOUR_BUY = 0x00C853       # green
_COLOUR_SELL = 0x2196F3      # blue
_COLOUR_STOP = 0xF44336      # red
_COLOUR_PROFIT = 0x76FF03    # lime
_COLOUR_INFO = 0x9E9E9E      # grey
_COLOUR_ERROR = 0xFF6D00     # orange


def _post(payload: dict) -> None:
    url = config.DISCORD_WEBHOOK_URL
    if not url:
        return
    try:
        resp = requests.post(
            url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        if resp.status_code not in (200, 204):
            log.warning("Discord webhook returned %d: %s", resp.status_code, resp.text[:200])
    except Exception as exc:
        log.warning("Discord notification failed: %s", exc)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _embed(title: str, description: str, colour: int, fields: list[dict] | None = None) -> dict:
    embed: dict = {
        "title": title,
        "description": description,
        "color": colour,
        "timestamp": _timestamp(),
        "footer": {"text": "Trading Bot"},
    }
    if fields:
        embed["fields"] = fields
    return {"embeds": [embed]}


# ── Public notification functions ─────────────────────────────────────────────

def trade_opened(
    ticker: str,
    qty: float,
    price: float,
    composite_score: float,
    sources: list[str],
    broker: str,
) -> None:
    mode = "DRY RUN" if config.DRY_RUN else broker.upper()
    fields = [
        {"name": "Qty", "value": f"`{qty:.4f}`", "inline": True},
        {"name": "Price", "value": f"`${price:,.4f}`", "inline": True},
        {"name": "Value", "value": f"`${qty * price:,.2f}`", "inline": True},
        {"name": "Signal Score", "value": f"`{composite_score:+.3f}`", "inline": True},
        {"name": "Broker", "value": f"`{mode}`", "inline": True},
        {"name": "Sources", "value": ", ".join(sources) or "—", "inline": False},
    ]
    _post(_embed(
        title=f"🟢 BUY — {ticker}",
        description=f"New position opened on **{ticker}**",
        colour=_COLOUR_BUY,
        fields=fields,
    ))


def trade_closed(
    ticker: str,
    qty: float,
    entry_price: float,
    exit_price: float,
    pnl: float,
    reason: str,
) -> None:
    pnl_str = f"${pnl:+,.2f}"
    pct = (exit_price - entry_price) / entry_price * 100 if entry_price else 0
    colour = _COLOUR_PROFIT if pnl >= 0 else _COLOUR_STOP
    emoji = "✅" if pnl >= 0 else "🔴"
    fields = [
        {"name": "Qty", "value": f"`{qty:.4f}`", "inline": True},
        {"name": "Entry", "value": f"`${entry_price:,.4f}`", "inline": True},
        {"name": "Exit", "value": f"`${exit_price:,.4f}`", "inline": True},
        {"name": "P&L", "value": f"`{pnl_str}  ({pct:+.2f}%)`", "inline": True},
        {"name": "Reason", "value": f"`{reason}`", "inline": True},
    ]
    _post(_embed(
        title=f"{emoji} SELL — {ticker}",
        description=f"Position closed on **{ticker}**",
        colour=colour,
        fields=fields,
    ))


def stop_loss_alert(ticker: str, entry: float, current: float, pnl: float) -> None:
    pct = (current - entry) / entry * 100 if entry else 0
    _post(_embed(
        title=f"🚨 STOP-LOSS — {ticker}",
        description=(
            f"**{ticker}** dropped **{pct:.1f}%** below entry.\n"
            f"Closing position to protect capital."
        ),
        colour=_COLOUR_STOP,
        fields=[
            {"name": "Entry", "value": f"`${entry:,.4f}`", "inline": True},
            {"name": "Current", "value": f"`${current:,.4f}`", "inline": True},
            {"name": "Loss", "value": f"`${pnl:+,.2f}`", "inline": True},
        ],
    ))


def portfolio_summary(summary: dict) -> None:
    positions = summary.get("open_positions", [])
    pos_lines = []
    for p in positions:
        pnl_str = f"{p['pct_change']:+.2f}%"
        pos_lines.append(
            f"**{p['ticker']}** — {p['qty']:.4f} @ ${p['avg_price']:.4f} | {pnl_str}"
        )
    description = "\n".join(pos_lines) if pos_lines else "_No open positions_"
    cb = "🚨 ACTIVE" if summary.get("circuit_breaker_active") else "✅ Off"
    fields = [
        {"name": "Cash", "value": f"`${summary['cash']:,.2f}`", "inline": True},
        {"name": "Portfolio Value", "value": f"`${summary['portfolio_value']:,.2f}`", "inline": True},
        {"name": "Daily P&L", "value": f"`${summary['daily_pnl']:+,.2f}`", "inline": True},
        {"name": "Consec. Losses", "value": f"`{summary.get('consecutive_losses', 0)}`", "inline": True},
        {"name": "Circuit Breaker", "value": f"`{cb}`", "inline": True},
    ]
    _post(_embed(
        title="📊 Portfolio Summary",
        description=description,
        colour=_COLOUR_INFO,
        fields=fields,
    ))


def error_alert(message: str) -> None:
    _post(_embed(
        title="⚠️ Bot Error",
        description=f"```{message[:1500]}```",
        colour=_COLOUR_ERROR,
    ))


def learning_report(
    win_rate: Optional[float],
    avg_pnl_pct: Optional[float],
    total_trades: int,
    changes: list[str],
    current_params: dict,
) -> None:
    """Sent whenever the learning engine changes strategy parameters."""
    change_text = "\n".join(f"• {c}" for c in changes) if changes else "_No changes_"
    src_weights = current_params.get("source_weights", {})
    src_str = "  ".join(f"{k}={v:.2f}" for k, v in src_weights.items()) or "—"
    blacklist = current_params.get("blacklist", [])
    fields = [
        {"name": "Trades Analysed", "value": f"`{total_trades}`", "inline": True},
        {"name": "Win Rate (last 20)", "value": f"`{win_rate:.0%}`" if win_rate is not None else "`n/a`", "inline": True},
        {"name": "Avg P&L (last 20)", "value": f"`{avg_pnl_pct:+.2f}%`" if avg_pnl_pct is not None else "`n/a`", "inline": True},
        {"name": "Signal Weights", "value": f"`sent={current_params['sentiment_weight']:.3f}  tech={current_params['technical_weight']:.3f}`", "inline": False},
        {"name": "Buy Threshold", "value": f"`{current_params['buy_threshold']:.3f}`", "inline": True},
        {"name": "Source Weights", "value": f"`{src_str}`", "inline": False},
        {"name": "Blacklist", "value": f"`{', '.join(blacklist) or 'none'}`", "inline": True},
    ]
    _post(_embed(
        title="🧠 Bot Self-Improvement Report",
        description=f"**What changed:**\n{change_text}",
        colour=0x7C4DFF,   # purple
        fields=fields,
    ))


def startup_message(dry_run: bool) -> None:
    mode = "**DRY RUN** (no real orders)" if dry_run else "**PAPER TRADING**"
    _post(_embed(
        title="🤖 Trading Bot Started",
        description=(
            f"Mode: {mode}\n"
            f"Assets: US Stocks (Alpaca) + Crypto (Binance Testnet)\n"
            f"Strategy: Sentiment (55%) + Technical (45%)\n"
            f"Risk: 2.5% per trade | SL: 5% | TP: 15%"
        ),
        colour=_COLOUR_INFO,
    ))
