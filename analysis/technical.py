"""
Technical analysis engine.

Indicators computed:
  - RSI (14)                    momentum oscillator
  - MACD (12, 26, 9)           trend / momentum
  - Bollinger Bands (20, 2σ)   mean-reversion / volatility
  - Volume ratio                current vs 20-period average

Each indicator produces a sub-score in [-1, +1].
Final TechnicalResult.score = weighted average of sub-scores.

Data source: yfinance (free, delayed ~15 min for stocks; near real-time for crypto).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)

# Symbol suffix for crypto on yfinance (e.g. BTC → BTC-USD)
_CRYPTO_SUFFIX = "-USD"

_WATCHED_CRYPTO = {
    "BTC", "ETH", "SOL", "ADA", "DOGE", "AVAX", "LINK", "DOT", "MATIC", "UNI",
}


@dataclass
class TechnicalResult:
    ticker: str
    score: float              # aggregate score -1 to +1
    rsi: Optional[float]      # 0–100
    rsi_score: float          # -1 to +1
    macd_score: float         # -1 to +1
    bb_score: float           # -1 to +1
    volume_score: float       # -1 to +1
    current_price: float
    details: dict = field(default_factory=dict)


# ── Indicator helpers ─────────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_series = 100 - 100 / (1 + rs)
    return float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0


def _rsi_score(rsi_val: float) -> float:
    """Map RSI to [-1, +1]: oversold (<30) → +1, overbought (>70) → -1."""
    if rsi_val <= 20:
        return 1.0
    if rsi_val <= 30:
        return 0.6
    if rsi_val <= 45:
        return 0.2
    if rsi_val <= 55:
        return 0.0
    if rsi_val <= 70:
        return -0.2
    if rsi_val <= 80:
        return -0.6
    return -1.0


def _macd(close: pd.Series) -> tuple[float, float]:
    """Return (macd_line, signal_line) using EMA 12/26/9."""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1])


def _macd_score(macd_val: float, signal_val: float, close_val: float) -> float:
    """
    Bullish when MACD > signal; bearish when below.
    Normalised by current price to make scores comparable across assets.
    """
    if close_val == 0:
        return 0.0
    diff = macd_val - signal_val
    normalised = diff / close_val          # small fraction
    # Scale: ±0.5% price-normalised diff → ±1 score
    score = normalised / 0.005
    return max(-1.0, min(1.0, score))


def _bollinger_bands(close: pd.Series, period: int = 20, std_mult: float = 2.0):
    """Return (upper, middle, lower) Bollinger Bands."""
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return float(upper.iloc[-1]), float(sma.iloc[-1]), float(lower.iloc[-1])


def _bb_score(price: float, upper: float, lower: float, middle: float) -> float:
    """
    Price below lower band → buy signal (+1).
    Price above upper band → sell signal (-1).
    Between bands → proportional score.
    """
    band_width = upper - lower
    if band_width <= 0:
        return 0.0
    # Position within bands: 0 = lower, 0.5 = middle, 1 = upper
    position = (price - lower) / band_width
    # Map [0, 1] → [+1, -1]
    return max(-1.0, min(1.0, 1.0 - 2 * position))


def _volume_score(volume: pd.Series) -> float:
    """
    Volume spike above 20-period average amplifies signal.
    Returns 0 (neutral) to +0.5 (high volume confirmation).
    High volume doesn't indicate direction — it only amplifies.
    Used as a minor secondary signal.
    """
    avg_vol = volume.rolling(20).mean().iloc[-1]
    current_vol = float(volume.iloc[-1])
    if avg_vol <= 0:
        return 0.0
    ratio = current_vol / avg_vol
    if ratio >= 3.0:
        return 0.5
    if ratio >= 2.0:
        return 0.3
    if ratio >= 1.5:
        return 0.1
    return 0.0


# ── yfinance wrapper ──────────────────────────────────────────────────────────

def _yf_symbol(ticker: str) -> str:
    """Convert internal ticker to yfinance symbol."""
    if ticker in _WATCHED_CRYPTO:
        return f"{ticker}{_CRYPTO_SUFFIX}"
    return ticker


def _fetch_ohlcv(ticker: str) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        symbol = _yf_symbol(ticker)
        df = yf.download(symbol, period="60d", interval="1d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 30:
            log.debug("Insufficient price data for %s", ticker)
            return None
        return df
    except Exception as exc:
        log.warning("yfinance error for %s: %s", ticker, exc)
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def analyse(ticker: str) -> Optional[TechnicalResult]:
    """
    Run all technical indicators for *ticker*.
    Returns None if price data cannot be fetched.
    """
    df = _fetch_ohlcv(ticker)
    if df is None:
        return None

    # yfinance column names may be MultiIndex; flatten if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"].dropna()
    volume = df["Volume"].dropna() if "Volume" in df.columns else pd.Series(dtype=float)

    if len(close) < 26:
        log.debug("Not enough candles for %s (%d)", ticker, len(close))
        return None

    price = float(close.iloc[-1])

    # --- RSI ---
    rsi_val = _rsi(close)
    rsi_s = _rsi_score(rsi_val)

    # --- MACD ---
    macd_val, signal_val = _macd(close)
    macd_s = _macd_score(macd_val, signal_val, price)

    # --- Bollinger Bands ---
    upper, middle, lower = _bollinger_bands(close)
    bb_s = _bb_score(price, upper, lower, middle)

    # --- Volume ---
    vol_s = _volume_score(volume) if len(volume) >= 20 else 0.0

    # --- Aggregate ---
    # Weights: RSI 35%, MACD 35%, BB 25%, Volume 5%
    aggregate = (
        0.35 * rsi_s
        + 0.35 * macd_s
        + 0.25 * bb_s
        + 0.05 * vol_s
    )
    aggregate = max(-1.0, min(1.0, aggregate))

    result = TechnicalResult(
        ticker=ticker,
        score=round(aggregate, 4),
        rsi=round(rsi_val, 2),
        rsi_score=round(rsi_s, 4),
        macd_score=round(macd_s, 4),
        bb_score=round(bb_s, 4),
        volume_score=round(vol_s, 4),
        current_price=round(price, 6),
        details={
            "macd": round(macd_val, 6),
            "macd_signal": round(signal_val, 6),
            "bb_upper": round(upper, 6),
            "bb_middle": round(middle, 6),
            "bb_lower": round(lower, 6),
        },
    )

    log.debug(
        "Technical [%s]: score=%.3f  RSI=%.1f  MACD_s=%.3f  BB_s=%.3f  Vol_s=%.3f  price=%.4f",
        ticker, aggregate, rsi_val, macd_s, bb_s, vol_s, price,
    )
    return result


def analyse_batch(tickers: list[str]) -> dict[str, TechnicalResult]:
    """Run analyse() for each ticker; skip failures silently."""
    results = {}
    for ticker in tickers:
        r = analyse(ticker)
        if r is not None:
            results[ticker] = r
    return results
