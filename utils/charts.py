"""
Chart generation using matplotlib and mplfinance.
- 1D / 5D: area/mountain chart with VWAP line
- 1W / 1M / 1Y / 5Y / MAX: candlestick chart with EMA overlays + buy zone shading
All charts use a dark theme matching Discord's dark mode.
Returns PNG bytes suitable for discord.File.
"""

import asyncio
import io
import logging
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import mplfinance as mpf
from matplotlib.ticker import FuncFormatter

logger = logging.getLogger(__name__)

# Dark theme colors
BG_COLOR = "#0d1117"
PANEL_COLOR = "#161b22"
TEXT_COLOR = "#e6edf3"
GRID_COLOR = "#21262d"
GREEN = "#2ea043"
RED = "#da3633"
ACCENT_GREEN = "#3fb950"
ACCENT_BLUE = "#58a6ff"
ACCENT_ORANGE = "#d29922"
EMA20_COLOR = "#ff7b72"
EMA50_COLOR = "#ffa657"
EMA200_COLOR = "#79c0ff"
VWAP_COLOR = "#f0883e"

matplotlib.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": PANEL_COLOR,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.4,
    "font.family": "DejaVu Sans",
    "font.size": 9,
})


def _run_in_thread(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, lambda: func(*args, **kwargs))


def _hex_to_rgba(hex_color: str, alpha: float) -> tuple:
    """Convert #rrggbb hex to an RGBA tuple for matplotlib."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
    return (r, g, b, alpha)


def _build_summary_infographic(data: dict) -> bytes:
    """Render a dark-theme infographic card for an article summary."""
    import textwrap

    sentiment = data.get("sentiment", "Neutral")
    sent_color = {
        "Bullish": ACCENT_GREEN,
        "Bearish": RED,
    }.get(sentiment, "#8b949e")

    fig = plt.figure(figsize=(10, 5.8), dpi=110)
    fig.patch.set_facecolor(BG_COLOR)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.8)
    ax.axis("off")
    ax.set_facecolor(BG_COLOR)

    def panel(x, y, w, h, edge=GRID_COLOR):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.12",
            facecolor=PANEL_COLOR,
            edgecolor=edge,
            linewidth=0.8,
            zorder=1,
        )
        ax.add_patch(rect)

    def pill(x, y, w, h, bg_hex, edge_hex, label, fontsize=8.5):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.08",
            facecolor=_hex_to_rgba(bg_hex, 0.18),
            edgecolor=edge_hex,
            linewidth=1.2,
            zorder=2,
        ))
        ax.text(x + w / 2, y + h / 2, label,
                fontsize=fontsize, fontweight="bold",
                color=edge_hex, va="center", ha="center", zorder=3)

    # ── Header panel ─────────────────────────────────────────────
    panel(0.25, 4.35, 9.5, 1.25)

    title = data.get("title", "Article Summary")
    wrapped_title = "\n".join(textwrap.wrap(title, width=52))
    ax.text(0.55, 5.42, wrapped_title,
            fontsize=12.5, fontweight="bold", color=TEXT_COLOR,
            va="top", ha="left", linespacing=1.35, zorder=2)

    # Sentiment pill — top right of header
    pill(7.55, 4.52, 1.95, 0.46, sent_color, sent_color,
         f"{'▲' if sentiment == 'Bullish' else '▼' if sentiment == 'Bearish' else '●'}  {sentiment.upper()}",
         fontsize=8.5)

    # ── Section label ────────────────────────────────────────────
    ax.text(0.55, 4.18, "KEY TAKEAWAYS",
            fontsize=7.5, color="#8b949e", fontweight="bold",
            va="top", ha="left", zorder=2)

    # ── Bullets panel ────────────────────────────────────────────
    panel(0.25, 1.35, 9.5, 2.72)

    bullets = data.get("bullets", [])
    y_pos = 3.82
    for bullet in bullets[:4]:
        wrapped = textwrap.wrap(bullet, width=80)
        # Bullet dot
        ax.plot(0.6, y_pos + 0.04, "o", color=ACCENT_BLUE, markersize=5, zorder=3)
        for j, line in enumerate(wrapped[:2]):
            ax.text(0.82, y_pos - j * 0.3, line,
                    fontsize=10, color=TEXT_COLOR,
                    va="top", ha="left", zorder=2)
        y_pos -= 0.62

    # ── Bottom row ───────────────────────────────────────────────
    panel(0.25, 0.2, 9.5, 0.95)

    # Market impact
    impact = data.get("market_impact", "")
    ax.text(0.55, 0.94, "MARKET IMPACT", fontsize=7, color="#8b949e",
            fontweight="bold", va="top", zorder=2)
    ax.text(0.55, 0.68, impact or "—", fontsize=9.5, color=ACCENT_ORANGE,
            fontweight="bold", va="top", zorder=2)

    # Ticker pills
    tickers = data.get("key_tickers", [])
    if tickers:
        ax.text(5.2, 0.94, "MENTIONED", fontsize=7, color="#8b949e",
                fontweight="bold", va="top", zorder=2)
        x_t = 5.2
        for ticker in tickers[:5]:
            w = max(len(ticker) * 0.115 + 0.35, 0.7)
            pill(x_t, 0.27, w, 0.38, ACCENT_BLUE, ACCENT_BLUE, f"${ticker}", fontsize=8)
            x_t += w + 0.18

    # ── Footer ───────────────────────────────────────────────────
    ax.text(5.0, 0.08, "Martin's Stocks  •  Summarized by Groq LLaMA 3.3 70B",
            fontsize=7, color="#484f58", va="bottom", ha="center", zorder=2)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110, facecolor=BG_COLOR)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


async def generate_summary_infographic(data: dict) -> Optional[bytes]:
    """Async wrapper — renders article summary infographic in a thread."""
    try:
        return await _run_in_thread(_build_summary_infographic, data)
    except Exception as e:
        logger.error(f"generate_summary_infographic: {e}")
        return None


def _build_stock_scorecard(ticker: str, ai_result: dict, ta_data: dict, info: dict) -> bytes:
    """Render a dark-theme stock scorecard infographic — signal, 4 score bars, entry info."""
    import textwrap

    signal = ai_result.get("overall_signal", "NEUTRAL")
    signal_col = {
        "STRONG BUY": ACCENT_GREEN,
        "BUY": "#3fb950",
        "NEUTRAL": "#8b949e",
        "SELL": "#f85149",
        "STRONG SELL": RED,
    }.get(signal, "#8b949e")

    scores = [
        ("Momentum", ai_result.get("momentum_score", 5), ACCENT_BLUE),
        ("Smart Money", ai_result.get("smart_money_score", 5), ACCENT_ORANGE),
        ("News Catalyst", ai_result.get("news_score", 5), "#bc8cff"),
        ("Technical", ai_result.get("technical_score", 5), ACCENT_GREEN),
    ]

    fig = plt.figure(figsize=(9, 5.2), dpi=110)
    fig.patch.set_facecolor(BG_COLOR)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    ax.set_facecolor(BG_COLOR)

    def panel(x, y, w, h):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1",
            facecolor=PANEL_COLOR, edgecolor=GRID_COLOR, linewidth=0.8, zorder=1,
        ))

    # ── Header ────────────────────────────────────────────────────────────────
    panel(0.2, 4.0, 8.6, 1.0)

    name = info.get("longName") or info.get("shortName") or ticker
    current_price = ta_data.get("current_price") or info.get("currentPrice") or 0
    prev_close = info.get("previousClose") or current_price
    change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close else 0
    arrow = "▲" if change_pct >= 0 else "▼"
    change_col = ACCENT_GREEN if change_pct >= 0 else RED

    ax.text(0.45, 4.82, f"{ticker}  —  {truncate_label(name, 38)}", fontsize=12,
            fontweight="bold", color=TEXT_COLOR, va="center", zorder=2)
    ax.text(0.45, 4.28, f"${current_price:,.2f}", fontsize=11,
            color=TEXT_COLOR, va="center", fontweight="bold", zorder=2)
    ax.text(1.9, 4.28, f"{arrow} {abs(change_pct):.2f}%", fontsize=10,
            color=change_col, va="center", fontweight="bold", zorder=2)

    # Signal pill (right side of header)
    ax.add_patch(mpatches.FancyBboxPatch(
        (5.8, 4.12), 2.8, 0.72, boxstyle="round,pad=0.08",
        facecolor=_hex_to_rgba(signal_col, 0.2),
        edgecolor=signal_col, linewidth=1.5, zorder=2,
    ))
    ax.text(7.2, 4.48, signal, fontsize=10.5, fontweight="bold",
            color=signal_col, va="center", ha="center", zorder=3)

    # ── Score bars ────────────────────────────────────────────────────────────
    panel(0.2, 1.6, 8.6, 2.2)
    ax.text(0.45, 3.64, "AI SIGNAL LAYERS", fontsize=7.5, color="#8b949e",
            fontweight="bold", va="top", zorder=2)

    bar_y_start = 3.28
    bar_h = 0.28
    bar_max_w = 6.5
    bar_x = 0.45
    label_x = 2.35
    for i, (label, score, color) in enumerate(scores):
        y = bar_y_start - i * 0.52
        # Track (empty bar)
        ax.add_patch(mpatches.FancyBboxPatch(
            (label_x, y - bar_h / 2), bar_max_w, bar_h,
            boxstyle="round,pad=0.02",
            facecolor=_hex_to_rgba(color, 0.12),
            edgecolor=_hex_to_rgba(color, 0.25),
            linewidth=0.5, zorder=2,
        ))
        # Fill
        fill_w = max(0.1, bar_max_w * score / 10)
        ax.add_patch(mpatches.FancyBboxPatch(
            (label_x, y - bar_h / 2), fill_w, bar_h,
            boxstyle="round,pad=0.02",
            facecolor=_hex_to_rgba(color, 0.75),
            edgecolor="none",
            linewidth=0, zorder=3,
        ))
        ax.text(bar_x, y, label, fontsize=9, color=TEXT_COLOR,
                va="center", ha="left", zorder=2)
        ax.text(label_x + bar_max_w + 0.15, y, f"{score}/10", fontsize=9,
                color=color, va="center", fontweight="bold", zorder=2)

    # ── Entry info row ────────────────────────────────────────────────────────
    panel(0.2, 0.22, 8.6, 1.18)

    buy_zones = ai_result.get("buy_zones", [])
    zones_str = "  ·  ".join(buy_zones[:3]) if buy_zones else "—"
    risk = ai_result.get("risk_level", "Medium")
    risk_col = {"Low": ACCENT_GREEN, "Medium": ACCENT_ORANGE, "High": RED}.get(risk, ACCENT_ORANGE)
    entry = ai_result.get("entry_points", "")

    ax.text(0.45, 1.25, "BUY ZONES", fontsize=7, color="#8b949e", fontweight="bold", va="top", zorder=2)
    ax.text(0.45, 1.00, zones_str, fontsize=9.5, color=ACCENT_GREEN, fontweight="bold", va="top", zorder=2)

    ax.text(5.2, 1.25, "RISK", fontsize=7, color="#8b949e", fontweight="bold", va="top", zorder=2)
    ax.text(5.2, 1.00, risk, fontsize=9.5, color=risk_col, fontweight="bold", va="top", zorder=2)

    if entry:
        wrapped = textwrap.fill(entry, width=55)
        ax.text(0.45, 0.68, wrapped, fontsize=8, color="#8b949e", va="top", zorder=2, linespacing=1.3)

    # Footer
    ax.text(4.5, 0.08, "Martin's Stocks  •  AI by Groq LLaMA 3.3 70B  •  Data: Yahoo Finance",
            fontsize=6.5, color="#484f58", va="bottom", ha="center", zorder=2)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110, facecolor=BG_COLOR)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def truncate_label(text: str, max_len: int) -> str:
    """Truncate a string for display in matplotlib."""
    return text if len(text) <= max_len else text[:max_len - 1] + "…"


async def generate_stock_scorecard(
    ticker: str, ai_result: dict, ta_data: dict, info: dict
) -> Optional[bytes]:
    """Async wrapper — renders stock scorecard in a thread."""
    try:
        return await _run_in_thread(_build_stock_scorecard, ticker, ai_result, ta_data, info)
    except Exception as e:
        logger.error(f"generate_stock_scorecard: {e}")
        return None


async def generate_chart(
    ticker: str,
    period: str,
    buy_zones: Optional[list] = None,
) -> Optional[bytes]:
    """
    Main entry point. Dispatches to area chart (1D/5D) or candlestick (1W+).
    buy_zones: list of float price levels to shade on candlestick charts.
    Returns PNG bytes or None on failure.
    """
    from services.market_data import get_history

    # Map Discord button period labels to yfinance period + interval
    period_map = {
        "1D": ("1d", "5m"),
        "5D": ("5d", "15m"),
        "1W": ("5d", "1d"),
        "1M": ("1mo", "1d"),
        "1Y": ("1y", "1d"),
        "5Y": ("5y", "1wk"),
        "MAX": ("max", "1mo"),
    }
    yf_period, interval = period_map.get(period, ("1y", "1d"))

    try:
        df = await get_history(ticker, period=yf_period, interval=interval)
        if df is None or df.empty:
            return None
        # Ensure proper column names
        df.columns = [c.strip().title() for c in df.columns]
        if "Close" not in df.columns:
            return None

        if period in ("1D", "5D"):
            return await _run_in_thread(_draw_area_chart, ticker, df, period)
        else:
            return await _run_in_thread(_draw_candle_chart, ticker, df, period, buy_zones or [])
    except Exception as e:
        logger.error(f"generate_chart({ticker}, {period}): {e}")
        return None


# ─── Area / Mountain Chart (1D / 5D) ─────────────────────────────────────────

def _draw_area_chart(ticker: str, df: pd.DataFrame, period: str) -> bytes:
    from utils.technical import calculate_vwap

    close = df["Close"]
    first_price = float(close.iloc[0])
    last_price = float(close.iloc[-1])
    is_up = last_price >= first_price
    line_color = ACCENT_GREEN if is_up else RED

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)

    x = np.arange(len(close))
    y = close.values.astype(float)

    # Gradient fill
    ax.fill_between(x, y, y.min() * 0.999, alpha=0.35, color=line_color)
    ax.plot(x, y, color=line_color, linewidth=1.5, zorder=5)

    # VWAP
    try:
        vwap = calculate_vwap(df)
        ax.plot(x, vwap.values.astype(float),
                color=VWAP_COLOR, linewidth=1.2, linestyle="--",
                alpha=0.8, label=f"VWAP ${vwap.iloc[-1]:.2f}", zorder=4)
    except Exception:
        pass

    # Current price line
    ax.axhline(last_price, color=TEXT_COLOR, linewidth=0.6, linestyle=":", alpha=0.5)

    # Labels every N points on X-axis
    step = max(1, len(x) // 8)
    ticks = x[::step]
    labels = []
    idx = df.index
    for t in ticks:
        try:
            ts = idx[t]
            if hasattr(ts, "strftime"):
                labels.append(ts.strftime("%H:%M" if period == "1D" else "%m/%d %H:%M"))
            else:
                labels.append(str(ts)[:10])
        except Exception:
            labels.append("")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)

    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:.2f}"))
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.spines[:].set_visible(False)

    change_pct = ((last_price - first_price) / first_price) * 100
    arrow = "▲" if change_pct >= 0 else "▼"
    fig.suptitle(
        f"{ticker}  •  ${last_price:.2f}  {arrow} {abs(change_pct):.2f}%  [{period}]",
        color=TEXT_COLOR, fontsize=13, fontweight="bold", y=0.97
    )
    ax.legend(loc="upper left", framealpha=0.2, fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=BG_COLOR)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─── Candlestick Chart (1W / 1M / 1Y / 5Y / MAX) ────────────────────────────

def _draw_candle_chart(
    ticker: str, df: pd.DataFrame, period: str, buy_zones: list
) -> bytes:
    from utils.technical import calculate_ema

    # mplfinance custom style
    mc = mpf.make_marketcolors(
        up=ACCENT_GREEN, down=RED,
        wick={"up": ACCENT_GREEN, "down": RED},
        edge={"up": ACCENT_GREEN, "down": RED},
        volume={"up": ACCENT_GREEN, "down": RED},
    )
    s = mpf.make_mpf_style(
        marketcolors=mc,
        facecolor=PANEL_COLOR,
        figcolor=BG_COLOR,
        edgecolor=GRID_COLOR,
        gridcolor=GRID_COLOR,
        gridstyle="--",
        gridaxis="both",
        rc={
            "font.size": 9,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
        },
    )

    close = df["Close"]
    add_plots = []

    # EMA overlays
    if len(close) >= 20:
        ema20 = calculate_ema(close, 20)
        add_plots.append(mpf.make_addplot(ema20, color=EMA20_COLOR, width=1.2, label="EMA20"))
    if len(close) >= 50:
        ema50 = calculate_ema(close, 50)
        add_plots.append(mpf.make_addplot(ema50, color=EMA50_COLOR, width=1.2, label="EMA50"))
    if len(close) >= 200:
        ema200 = calculate_ema(close, 200)
        add_plots.append(mpf.make_addplot(ema200, color=EMA200_COLOR, width=1.5, label="EMA200"))

    last_price = float(close.iloc[-1])
    first_price = float(close.iloc[0])
    change_pct = ((last_price - first_price) / first_price) * 100
    arrow = "▲" if change_pct >= 0 else "▼"

    title = f"{ticker}  •  ${last_price:.2f}  {arrow} {abs(change_pct):.2f}%  [{period}]"

    fig, axes = mpf.plot(
        df,
        type="candle",
        style=s,
        title=title,
        ylabel="Price (USD)",
        volume=True,
        addplot=add_plots if add_plots else None,
        figsize=(12, 7),
        returnfig=True,
        tight_layout=True,
        warn_too_much_data=9999,
    )

    ax_main = axes[0]

    # Buy zone shading
    if buy_zones:
        try:
            numeric_zones = []
            for z in buy_zones:
                z_str = str(z).replace("$", "").replace(",", "").strip()
                try:
                    numeric_zones.append(float(z_str))
                except ValueError:
                    pass
            for zone in numeric_zones:
                ax_main.axhspan(zone * 0.995, zone * 1.005,
                                alpha=0.15, color=ACCENT_GREEN,
                                zorder=1, label=f"Buy Zone ${zone:.2f}")
        except Exception as e:
            logger.debug(f"buy_zone shading error: {e}")

    # Legend
    legend_patches = []
    if len(close) >= 20:
        legend_patches.append(mpatches.Patch(color=EMA20_COLOR, label=f"EMA20 ${calculate_ema(close, 20).iloc[-1]:.2f}"))
    if len(close) >= 50:
        legend_patches.append(mpatches.Patch(color=EMA50_COLOR, label=f"EMA50 ${calculate_ema(close, 50).iloc[-1]:.2f}"))
    if len(close) >= 200:
        legend_patches.append(mpatches.Patch(color=EMA200_COLOR, label=f"EMA200 ${calculate_ema(close, 200).iloc[-1]:.2f}"))
    if legend_patches:
        ax_main.legend(handles=legend_patches, loc="upper left",
                       framealpha=0.2, fontsize=8, facecolor=PANEL_COLOR)

    # Style the title
    if fig.texts:
        fig.texts[0].set_color(TEXT_COLOR)
        fig.texts[0].set_fontsize(12)
        fig.texts[0].set_fontweight("bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=BG_COLOR)
    plt.close(fig)
    buf.seek(0)
    return buf.read()
