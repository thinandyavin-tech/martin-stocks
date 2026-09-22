"""
Technical analysis calculations:
EMA, RSI, OBV, volume ratio, candlestick patterns, 52W range, relative strength vs SPY.
All functions operate on pandas DataFrames/Series returned by yfinance.
"""

import logging
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Core Indicators ─────────────────────────────────────────────────────────

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> float:
    """Calculate RSI and return current value (0-100)."""
    if len(series) < period + 1:
        return 50.0
    delta = series.diff().dropna()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return round(float(val), 1) if not np.isnan(val) else 50.0


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    obv = (direction * volume).cumsum()
    return obv


def obv_trend(close: pd.Series, volume: pd.Series, lookback: int = 20) -> str:
    """Return 'Rising', 'Falling', or 'Flat' based on OBV slope over lookback days."""
    if len(close) < lookback + 1:
        return "Insufficient data"
    obv = calculate_obv(close, volume)
    recent = obv.iloc[-lookback:]
    x = np.arange(len(recent))
    slope = np.polyfit(x, recent.values.astype(float), 1)[0]
    if slope > 0:
        return "Rising"
    elif slope < 0:
        return "Falling"
    return "Flat"


def volume_ratio(volume: pd.Series, lookback: int = 20) -> float:
    """Current volume vs N-day average volume ratio."""
    if len(volume) < lookback + 1:
        return 1.0
    avg = volume.iloc[-lookback:-1].mean()
    if avg == 0:
        return 1.0
    return round(float(volume.iloc[-1] / avg), 2)


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Volume-Weighted Average Price for intraday data."""
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    vwap = (typical_price * df["Volume"]).cumsum() / df["Volume"].cumsum()
    return vwap


def relative_strength_vs_spy(
    stock_close: pd.Series, spy_close: pd.Series, period: int = 90
) -> float:
    """
    Relative strength ratio: stock return / SPY return over last N days.
    >1 means outperforming, <1 underperforming.
    """
    if len(stock_close) < period or len(spy_close) < period:
        return 1.0
    stock_ret = stock_close.iloc[-1] / stock_close.iloc[-period] - 1
    spy_ret = spy_close.iloc[-1] / spy_close.iloc[-period] - 1
    if spy_ret == 0:
        return 1.0
    return round(float(stock_ret / spy_ret), 2)


def range_position(close: float, low_52w: float, high_52w: float) -> float:
    """Position within 52W range as percentage (0% = at 52W low, 100% = at 52W high)."""
    if high_52w == low_52w:
        return 50.0
    return round((close - low_52w) / (high_52w - low_52w) * 100, 1)


# ─── Candlestick Pattern Detection ───────────────────────────────────────────

def detect_patterns(df: pd.DataFrame) -> str:
    """
    Detect common candlestick patterns from the last 3 candles.
    Returns pattern name or empty string.
    """
    if len(df) < 3:
        return ""

    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values

    patterns = []

    # --- Single candle ---
    # Doji (open ≈ close)
    body = abs(c[-1] - o[-1])
    range_ = h[-1] - l[-1]
    if range_ > 0 and body / range_ < 0.1:
        patterns.append("Doji")

    # Hammer (bullish): small body at top, long lower shadow
    lower_shadow = min(o[-1], c[-1]) - l[-1]
    upper_shadow = h[-1] - max(o[-1], c[-1])
    if (range_ > 0 and body / range_ < 0.35
            and lower_shadow > 2 * body
            and upper_shadow < body):
        if c[-1] < o[-1]:
            patterns.append("Inverted Hammer")
        else:
            patterns.append("Hammer")

    # Shooting Star (bearish): small body at bottom, long upper shadow
    if (range_ > 0 and body / range_ < 0.35
            and upper_shadow > 2 * body
            and lower_shadow < body):
        patterns.append("Shooting Star")

    # Marubozu (strong candle, almost no wicks)
    if range_ > 0 and body / range_ > 0.9:
        if c[-1] > o[-1]:
            patterns.append("Bullish Marubozu")
        else:
            patterns.append("Bearish Marubozu")

    # --- Two candle ---
    if len(df) >= 2:
        prev_body = abs(c[-2] - o[-2])
        # Engulfing
        if (c[-2] < o[-2]  # prev bearish
                and c[-1] > o[-1]  # curr bullish
                and c[-1] > o[-2]
                and o[-1] < c[-2]):
            patterns.append("Bullish Engulfing")
        elif (c[-2] > o[-2]  # prev bullish
              and c[-1] < o[-1]  # curr bearish
              and c[-1] < o[-2]
              and o[-1] > c[-2]):
            patterns.append("Bearish Engulfing")

    # --- Three candle ---
    if len(df) >= 3:
        # Morning Star
        if (c[-3] < o[-3]
                and abs(c[-2] - o[-2]) < abs(c[-3] - o[-3]) * 0.3
                and c[-1] > o[-1]
                and c[-1] > (o[-3] + c[-3]) / 2):
            patterns.append("Morning Star")
        # Evening Star
        if (c[-3] > o[-3]
                and abs(c[-2] - o[-2]) < abs(c[-3] - o[-3]) * 0.3
                and c[-1] < o[-1]
                and c[-1] < (o[-3] + c[-3]) / 2):
            patterns.append("Evening Star")
        # Three White Soldiers
        if all(c[i] > o[i] for i in [-3, -2, -1]) and c[-1] > c[-2] > c[-3]:
            patterns.append("Three White Soldiers")
        # Three Black Crows
        if all(c[i] < o[i] for i in [-3, -2, -1]) and c[-1] < c[-2] < c[-3]:
            patterns.append("Three Black Crows")

    return ", ".join(patterns) if patterns else ""


# ─── Full Technical Analysis Bundle ──────────────────────────────────────────

def run_full_analysis(df: pd.DataFrame, spy_df: Optional[pd.DataFrame] = None) -> dict:
    """
    Run complete technical analysis on a DataFrame (1y daily OHLCV).
    Returns a dict of all indicators.
    """
    if df is None or df.empty or len(df) < 20:
        return {}

    close = df["Close"]
    volume = df["Volume"]
    current_price = float(close.iloc[-1])

    ema20 = calculate_ema(close, 20)
    ema50 = calculate_ema(close, 50)
    ema200 = calculate_ema(close, 200)

    rsi = calculate_rsi(close)
    obv_t = obv_trend(close, volume)
    vol_ratio = volume_ratio(volume)

    pattern = detect_patterns(df.tail(5))

    low_52w = float(close.rolling(252).min().iloc[-1]) if len(close) >= 252 else float(close.min())
    high_52w = float(close.rolling(252).max().iloc[-1]) if len(close) >= 252 else float(close.max())
    range_pct = range_position(current_price, low_52w, high_52w)

    rs = 1.0
    if spy_df is not None and not spy_df.empty:
        rs = relative_strength_vs_spy(close, spy_df["Close"])

    ema20_val = float(ema20.iloc[-1])
    ema50_val = float(ema50.iloc[-1]) if len(ema50.dropna()) > 0 else None
    ema200_val = float(ema200.iloc[-1]) if len(ema200.dropna()) > 0 else None

    # EMA trend labels
    def _ema_label(price, ema):
        if ema is None:
            return "N/A"
        if price > ema:
            return f"${ema:.2f} ↑"
        return f"${ema:.2f} ↓"

    return {
        "current_price": round(current_price, 2),
        "rsi": rsi,
        "ema20": round(ema20_val, 2),
        "ema50": round(ema50_val, 2) if ema50_val else None,
        "ema200": round(ema200_val, 2) if ema200_val else None,
        "ema20_label": _ema_label(current_price, ema20_val),
        "ema50_label": _ema_label(current_price, ema50_val),
        "ema200_label": _ema_label(current_price, ema200_val),
        "above_ema20": current_price > ema20_val,
        "above_ema50": current_price > ema50_val if ema50_val else None,
        "above_ema200": current_price > ema200_val if ema200_val else None,
        "obv_trend": obv_t,
        "volume_ratio": vol_ratio,
        "pattern": pattern or "None",
        "low_52w": round(low_52w, 2),
        "high_52w": round(high_52w, 2),
        "range_pct": range_pct,
        "relative_strength": rs,
    }


def screener_signal(ta: dict) -> Optional[str]:
    """
    Given a technical analysis dict, return a screener signal or None.
    Signals: 'Oversold Bounce', 'Momentum', 'Breakout Watch', 'Overbought'
    """
    if not ta:
        return None
    rsi = ta.get("rsi", 50)
    vol_ratio = ta.get("volume_ratio", 1.0)
    above_ema20 = ta.get("above_ema20", False)
    above_ema50 = ta.get("above_ema50")
    range_pct = ta.get("range_pct", 50)

    if rsi <= 35 and above_ema20 is False:
        return "Oversold Bounce"
    if rsi >= 70 and vol_ratio >= 1.5 and above_ema20:
        return "Overbought"
    if above_ema20 and above_ema50 and vol_ratio >= 1.3 and 50 <= rsi <= 70:
        return "Momentum"
    if range_pct >= 90 and vol_ratio >= 1.2 and above_ema20:
        return "Breakout Watch"
    return None
