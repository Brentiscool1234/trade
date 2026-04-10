"""
Technical analysis engine.

Indicators computed
-------------------
  RSI (14)                     momentum oscillator
  MACD (12, 26, 9)             trend / momentum
  Bollinger Bands (20, 2σ)     mean-reversion / volatility
  EMA 50 / EMA 200             trend direction & Golden/Death Cross
  ATR (14)                     volatility — used for position sizing
  Volume ratio                 current vs 20-period average

Each indicator produces a sub-score in [-1, +1].
Final TechnicalResult.score is a weighted average.

Weights: RSI 28%, MACD 28%, BB 19%, Trend (EMA) 20%, Volume 5%

Data source: yfinance — 300-day daily bars (needed for reliable EMA 200).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)

_CRYPTO_SUFFIX = "-USD"
_WATCHED_CRYPTO = {
    "BTC", "ETH", "SOL", "ADA", "DOGE", "AVAX", "LINK", "DOT", "MATIC", "UNI",
}


@dataclass
class TechnicalResult:
    ticker: str
    score: float              # aggregate [-1, +1]
    rsi: Optional[float]      # 0–100
    rsi_score: float
    macd_score: float
    bb_score: float
    trend_score: float        # EMA alignment score [-1, +1]
    volume_score: float
    current_price: float
    atr: float                # Average True Range (absolute price units)
    ema50: float
    ema200: float
    trend_direction: str      # "uptrend" | "downtrend" | "sideways"
    golden_cross: bool        # EMA50 crossed above EMA200 in last 5 bars
    death_cross: bool         # EMA50 crossed below EMA200 in last 5 bars
    details: dict = field(default_factory=dict)


# ── Indicator helpers ─────────────────────────────────────────────────────────

def _ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_series = 100 - 100 / (1 + rs)
    return float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0


def _rsi_score(rsi_val: float) -> float:
    """Map RSI to [-1, +1]: deeply oversold = +1, deeply overbought = -1."""
    if rsi_val <= 20:   return  1.0
    if rsi_val <= 30:   return  0.65
    if rsi_val <= 40:   return  0.25
    if rsi_val <= 55:   return  0.0
    if rsi_val <= 65:   return -0.25
    if rsi_val <= 75:   return -0.65
    return -1.0


def _macd(close: pd.Series) -> tuple[float, float, float]:
    """Return (macd_line, signal_line, histogram)."""
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd_line = ema12 - ema26
    signal_line = _ema(macd_line, 9)
    histogram = macd_line - signal_line
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(histogram.iloc[-1])


def _macd_score(macd_val: float, signal_val: float, histogram: float, price: float) -> float:
    """
    Two components:
    1. MACD line vs signal line (crossover direction)
    2. Histogram trend (is the gap widening or narrowing?)
    Normalised by price so scores are comparable across assets.
    """
    if price <= 0:
        return 0.0
    crossover = (macd_val - signal_val) / price
    score = crossover / 0.005     # ±0.5% diff → ±1 score
    return max(-1.0, min(1.0, score))


def _bollinger_bands(close: pd.Series, period: int = 20, std_mult: float = 2.0):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return float(upper.iloc[-1]), float(sma.iloc[-1]), float(lower.iloc[-1])


def _bb_score(price: float, upper: float, lower: float) -> float:
    """Below lower band → +1 (buy), above upper band → -1 (sell)."""
    band_width = upper - lower
    if band_width <= 0:
        return 0.0
    position = (price - lower) / band_width   # 0 = at lower, 1 = at upper
    return max(-1.0, min(1.0, 1.0 - 2 * position))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """Average True Range — measures volatility in absolute price units."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_series = tr.rolling(period).mean()
    val = float(atr_series.iloc[-1])
    return val if not np.isnan(val) else 0.0


def _ema_trend_score(price: float, ema50: float, ema200: float) -> float:
    """
    Score based on price position relative to EMA50/200.
      +1 : price well above both EMAs, EMA50 above EMA200 (strong uptrend)
      -1 : price well below both EMAs, EMA50 below EMA200 (strong downtrend)
       0 : neutral / sideways
    """
    if ema50 <= 0 or ema200 <= 0:
        return 0.0
    # Normalised distance of price from EMA50
    vs_ema50 = (price - ema50) / ema50
    # EMA50 vs EMA200 spread (positive in uptrend)
    ema_spread = (ema50 - ema200) / ema200
    # Combined, scaled: ±5% combined distance = ±1 score
    combined = (vs_ema50 + ema_spread) / 0.05
    return max(-1.0, min(1.0, combined))


def _classify_trend(price: float, ema50: float, ema200: float) -> str:
    """
    Classify current trend for use in signal filters.
    Uses a 1.5% margin to avoid labelling sideways markets as trending.
    """
    if ema50 <= 0 or ema200 <= 0:
        return "sideways"
    margin = 0.015
    above_ema50 = price > ema50 * (1 + margin)
    below_ema50 = price < ema50 * (1 - margin)
    ema50_above_200 = ema50 > ema200 * (1 + margin)
    ema50_below_200 = ema50 < ema200 * (1 - margin)

    if above_ema50 and ema50_above_200:
        return "uptrend"
    if below_ema50 and ema50_below_200:
        return "downtrend"
    return "sideways"


def _volume_score(volume: pd.Series) -> float:
    """Positive score for above-average volume (amplifies existing signals)."""
    avg_vol = volume.rolling(20).mean().iloc[-1]
    current_vol = float(volume.iloc[-1])
    if avg_vol <= 0:
        return 0.0
    ratio = current_vol / avg_vol
    if ratio >= 3.0:  return 0.5
    if ratio >= 2.0:  return 0.3
    if ratio >= 1.5:  return 0.1
    return 0.0


# ── yfinance wrapper ──────────────────────────────────────────────────────────

def _yf_symbol(ticker: str) -> str:
    if ticker in _WATCHED_CRYPTO:
        return f"{ticker}{_CRYPTO_SUFFIX}"
    return ticker


def _fetch_ohlcv(ticker: str) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        symbol = _yf_symbol(ticker)
        # 300 days needed for a warm EMA 200
        df = yf.download(symbol, period="300d", interval="1d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 50:
            log.debug("Insufficient price data for %s (%d candles)", ticker, len(df))
            return None
        return df
    except Exception as exc:
        log.warning("yfinance error for %s: %s", ticker, exc)
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def analyse(ticker: str) -> Optional[TechnicalResult]:
    """
    Run all technical indicators for *ticker*.
    Returns None if price data cannot be fetched or is insufficient.
    """
    df = _fetch_ohlcv(ticker)
    if df is None:
        return None

    # Flatten MultiIndex columns (yfinance sometimes returns them)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"].dropna()
    if len(close) < 26:
        log.debug("Not enough candles for %s (%d)", ticker, len(close))
        return None

    price = float(close.iloc[-1])
    high = df["High"].dropna() if "High" in df.columns else close
    low = df["Low"].dropna() if "Low" in df.columns else close
    volume = df["Volume"].dropna() if "Volume" in df.columns else pd.Series(dtype=float)

    # ── RSI ───────────────────────────────────────────────────────────────────
    rsi_val = _rsi(close)
    rsi_s = _rsi_score(rsi_val)

    # ── MACD ──────────────────────────────────────────────────────────────────
    macd_val, signal_val, histogram = _macd(close)
    macd_s = _macd_score(macd_val, signal_val, histogram, price)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_upper, bb_middle, bb_lower = _bollinger_bands(close)
    bb_s = _bb_score(price, bb_upper, bb_lower)

    # ── EMA 50 / 200 ──────────────────────────────────────────────────────────
    ema50_series = _ema(close, 50)
    ema50_val = float(ema50_series.iloc[-1])

    if len(close) >= 200:
        ema200_series = _ema(close, 200)
    else:
        # Fallback for assets with limited history (e.g. newer crypto)
        ema200_series = _ema(close, min(len(close), 100))
        log.debug("EMA200 fallback for %s (only %d candles)", ticker, len(close))
    ema200_val = float(ema200_series.iloc[-1])

    trend_s = _ema_trend_score(price, ema50_val, ema200_val)
    trend_dir = _classify_trend(price, ema50_val, ema200_val)

    # Golden / Death Cross — check if crossover happened in last 5 bars
    golden_cross = death_cross = False
    if len(close) >= 6:
        prev_ema50 = float(ema50_series.iloc[-6])
        prev_ema200 = float(ema200_series.iloc[-6])
        was_below = prev_ema50 < prev_ema200
        now_above = ema50_val > ema200_val
        golden_cross = was_below and now_above
        death_cross = (not was_below) and (not now_above)
        if golden_cross:
            log.info("GOLDEN CROSS detected for %s — strong bullish signal", ticker)
        if death_cross:
            log.info("DEATH CROSS detected for %s — strong bearish signal", ticker)

    # ── ATR ───────────────────────────────────────────────────────────────────
    atr_val = _atr(high, low, close) if len(high) >= 15 else 0.0

    # ── Volume ────────────────────────────────────────────────────────────────
    vol_s = _volume_score(volume) if len(volume) >= 20 else 0.0

    # ── Aggregate ─────────────────────────────────────────────────────────────
    # Trend gets 20% weight — it's the primary direction filter.
    # RSI + MACD each 28%; BB 19%; Volume 5%.
    aggregate = (
        0.28 * rsi_s
        + 0.28 * macd_s
        + 0.19 * bb_s
        + 0.20 * trend_s
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
        trend_score=round(trend_s, 4),
        volume_score=round(vol_s, 4),
        current_price=round(price, 6),
        atr=round(atr_val, 6),
        ema50=round(ema50_val, 6),
        ema200=round(ema200_val, 6),
        trend_direction=trend_dir,
        golden_cross=golden_cross,
        death_cross=death_cross,
        details={
            "macd": round(macd_val, 6),
            "macd_signal": round(signal_val, 6),
            "macd_histogram": round(histogram, 6),
            "bb_upper": round(bb_upper, 6),
            "bb_middle": round(bb_middle, 6),
            "bb_lower": round(bb_lower, 6),
        },
    )

    log.debug(
        "Technical [%s]: score=%.3f  RSI=%.1f  MACD_s=%.3f  BB_s=%.3f  "
        "trend=%s(%.3f)  ATR=%.4f  price=%.4f",
        ticker, aggregate, rsi_val, macd_s, bb_s,
        trend_dir, trend_s, atr_val, price,
    )
    return result


def analyse_batch(tickers: list[str]) -> dict[str, TechnicalResult]:
    results = {}
    for ticker in tickers:
        r = analyse(ticker)
        if r is not None:
            results[ticker] = r
    return results
