"""
Shared formatting and utility helpers.
"""

import discord
from typing import Optional


# ─── Number Formatting ───────────────────────────────────────────────────────

def fmt_price(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"${value:,.2f}"


def fmt_pct(value: Optional[float], sign: bool = True) -> str:
    if value is None:
        return "N/A"
    prefix = "+" if (sign and value > 0) else ""
    return f"{prefix}{value:.2f}%"


def fmt_large(value: Optional[float]) -> str:
    """Format large numbers as $1.23B / $456M / $78K."""
    if value is None:
        return "N/A"
    abs_val = abs(value)
    sign = "-" if value < 0 else ""
    if abs_val >= 1e12:
        return f"{sign}${abs_val / 1e12:.2f}T"
    if abs_val >= 1e9:
        return f"{sign}${abs_val / 1e9:.2f}B"
    if abs_val >= 1e6:
        return f"{sign}${abs_val / 1e6:.2f}M"
    if abs_val >= 1e3:
        return f"{sign}${abs_val / 1e3:.2f}K"
    return f"{sign}${abs_val:.2f}"


def fmt_shares(value: Optional[int]) -> str:
    if value is None:
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


# ─── Signal / Indicator Formatting ───────────────────────────────────────────

def signal_emoji(signal: str) -> str:
    mapping = {
        "STRONG BUY": "🚀",
        "BUY": "🟢",
        "NEUTRAL": "🟡",
        "SELL": "🔴",
        "STRONG SELL": "💀",
    }
    return mapping.get(signal.upper(), "⚪")


def signal_color(signal: str) -> discord.Color:
    mapping = {
        "STRONG BUY": discord.Color.from_str("#2ea043"),
        "BUY": discord.Color.green(),
        "NEUTRAL": discord.Color.gold(),
        "SELL": discord.Color.orange(),
        "STRONG SELL": discord.Color.red(),
    }
    return mapping.get(signal.upper(), discord.Color.light_grey())


def rsi_label(rsi: float) -> str:
    if rsi >= 70:
        return f"{rsi} 🔴 Overbought"
    if rsi <= 30:
        return f"{rsi} 🟢 Oversold"
    return f"{rsi} 🟡 Neutral"


def change_arrow(pct: float) -> str:
    return "▲" if pct >= 0 else "▼"


def change_color_str(pct: float) -> str:
    return "🟢" if pct >= 0 else "🔴"


def score_bar(score: int, max_score: int = 10) -> str:
    """Return a visual bar like: ████░░░░░░ 7/10"""
    filled = round(score / max_score * 10)
    bar = "█" * filled + "░" * (10 - filled)
    return f"`{bar}` {score}/{max_score}"


def vix_label(vix: float) -> tuple[str, str]:
    """Returns (label, tactical_tip) based on VIX level."""
    if vix < 12:
        return "Extreme Greed", "Caution: market may be overextended. Consider trimming longs."
    if vix < 15:
        return "Greed", "Markets calm. Stay the course but watch for complacency."
    if vix < 20:
        return "Neutral", "Normal volatility. Balanced approach appropriate."
    if vix < 25:
        return "Fear", "Elevated volatility. Watch support levels; dips may be buying opportunities."
    if vix < 30:
        return "High Fear", "Significant fear. Consider scaling into quality names on pullbacks."
    if vix < 40:
        return "Extreme Fear", "High volatility event. Historically a contrarian buy signal — proceed carefully."
    return "Market Panic", "VIX extremely elevated. Protect capital; wait for stabilization before adding risk."


def fear_greed_bar(vix: float) -> str:
    """ASCII progress bar: 0 = Extreme Fear (VIX 40+), 100 = Extreme Greed (VIX <12)."""
    # Invert VIX to get fear/greed index (higher VIX = more fear)
    vix = max(10.0, min(50.0, vix))
    score = int((50 - vix) / (50 - 10) * 100)
    score = max(0, min(100, score))
    filled = score // 5
    bar = "█" * filled + "░" * (20 - filled)
    return f"`[{bar}]` {score}/100"


def screener_signal_emoji(signal: str) -> str:
    mapping = {
        "Oversold Bounce": "🟢",
        "Momentum": "🚀",
        "Breakout Watch": "⚡",
        "Overbought": "🔴",
    }
    return mapping.get(signal, "⚪")


# ─── Embed Helpers ───────────────────────────────────────────────────────────

def base_embed(title: str, description: str = "", color: discord.Color = None) -> discord.Embed:
    embed = discord.Embed(
        title=title,
        description=description,
        color=color or discord.Color.from_str("#58a6ff"),
    )
    embed.set_footer(text="Martin's Stocks • Data via Yahoo Finance & Groq AI")
    return embed


def error_embed(message: str) -> discord.Embed:
    return discord.Embed(
        title="❌ Error",
        description=message,
        color=discord.Color.red(),
    )


def truncate(text: str, max_len: int = 1024) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
