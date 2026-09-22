"""
Professional analysis commands: /portfolio, /options, /sentiment
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands

from services import market_data as md
from services import news_service as news
from services import groq_service as groq
from utils.permissions import apply_admin_flair
from utils.helpers import fmt_price, fmt_pct, error_embed, truncate

logger = logging.getLogger(__name__)

# Risk-free rate assumption for Sharpe ratio calculation
RISK_FREE_RATE = 0.05


def _run_sync(func, *args, **kwargs):
    """Run a synchronous callable in the default thread pool."""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, lambda: func(*args, **kwargs))


class ProCommandsCog(commands.Cog, name="Pro"):
    def __init__(self, bot):
        self.bot = bot

    # ─── /portfolio ───────────────────────────────────────────────────────────

    @app_commands.command(
        name="portfolio",
        description="Portfolio analysis: returns, volatility, Sharpe ratio, max drawdown",
    )
    @app_commands.describe(
        ticker1="First ticker (required)",
        ticker2="Second ticker",
        ticker3="Third ticker",
        ticker4="Fourth ticker",
        ticker5="Fifth ticker",
    )
    async def portfolio_cmd(
        self,
        interaction: discord.Interaction,
        ticker1: str,
        ticker2: Optional[str] = None,
        ticker3: Optional[str] = None,
        ticker4: Optional[str] = None,
        ticker5: Optional[str] = None,
    ):
        """Calculate portfolio metrics for up to 5 tickers with equal weighting."""
        await interaction.response.defer()

        raw_tickers = [t for t in [ticker1, ticker2, ticker3, ticker4, ticker5] if t]
        tickers = [t.upper().strip() for t in raw_tickers]

        # Fetch 1-year history for all tickers in parallel
        results = await asyncio.gather(
            *[md.get_history(t, period="1y") for t in tickers],
            return_exceptions=True,
        )

        def safe(v):
            return v if not isinstance(v, Exception) else None

        histories = {t: safe(r) for t, r in zip(tickers, results)}
        valid = {t: h for t, h in histories.items() if h is not None and not h.empty}

        if not valid:
            await interaction.followup.send(embed=error_embed("Could not fetch data for any of the provided tickers."))
            return

        # Compute metrics in the thread pool — pandas operations are CPU-bound
        metrics = await _run_sync(_compute_portfolio_metrics, valid)

        weight_pct = round(100 / len(valid), 1)

        embed = discord.Embed(
            title=f"💼 Portfolio Analysis — {' / '.join(valid.keys())}",
            description=f"Equal-weighted portfolio ({weight_pct}% each) • 1-year lookback",
            color=discord.Color.from_str("#58a6ff"),
        )

        # Portfolio-level metrics
        total_return = metrics["portfolio_return"]
        ret_arrow = "▲" if total_return >= 0 else "▼"
        ret_color = "🟢" if total_return >= 0 else "🔴"
        embed.add_field(
            name="📊 Portfolio Metrics",
            value=(
                f"Total Return:     {ret_color} **{ret_arrow} {abs(total_return):.2f}%**\n"
                f"Annualised Vol:   **{metrics['volatility']:.2f}%**\n"
                f"Sharpe Ratio:     **{metrics['sharpe']:.2f}** (rf=5%)\n"
                f"Max Drawdown:     🔴 **{metrics['max_drawdown']:.2f}%**"
            ),
            inline=False,
        )

        # Per-stock contributions
        stock_lines = []
        for t, stock_ret in metrics["individual_returns"].items():
            arrow = "▲" if stock_ret >= 0 else "▼"
            color = "🟢" if stock_ret >= 0 else "🔴"
            contrib = stock_ret * (1 / len(valid))
            stock_lines.append(
                f"**{t}**: {color} {arrow}{abs(stock_ret):.1f}%  (contrib: {contrib:+.2f}%)"
            )
        embed.add_field(
            name="📈 Stock Contributions",
            value="\n".join(stock_lines),
            inline=False,
        )

        # Interpretation
        sharpe = metrics["sharpe"]
        if sharpe >= 1.5:
            sharpe_note = "Excellent risk-adjusted returns"
        elif sharpe >= 1.0:
            sharpe_note = "Good risk-adjusted returns"
        elif sharpe >= 0.5:
            sharpe_note = "Moderate risk-adjusted returns"
        else:
            sharpe_note = "Poor risk-adjusted returns — consider rebalancing"

        embed.add_field(
            name="💡 Assessment",
            value=sharpe_note,
            inline=False,
        )

        embed.set_footer(text="Martin's Stocks • 1Y data via Yahoo Finance • Sharpe uses 5% risk-free rate")
        apply_admin_flair(embed, interaction)
        await interaction.followup.send(embed=embed)

    # ─── /options ─────────────────────────────────────────────────────────────

    @app_commands.command(name="options", description="Options chain snapshot: nearest expiry, top OI calls & puts")
    @app_commands.describe(ticker="Stock ticker symbol (e.g. AAPL)")
    async def options_cmd(self, interaction: discord.Interaction, ticker: str):
        """Show the nearest-expiry options chain with top open-interest strikes."""
        await interaction.response.defer()
        ticker = ticker.upper().strip()

        chain_data = await _run_sync(_fetch_options_chain, ticker)

        if chain_data is None:
            await interaction.followup.send(embed=error_embed(f"Could not fetch options data for **{ticker}**. Options may not be available for this ticker."))
            return

        expiry = chain_data["expiry"]
        calls = chain_data["calls"]
        puts = chain_data["puts"]
        pc_ratio = chain_data["pc_ratio"]

        # Put/call ratio sentiment
        if pc_ratio > 1.2:
            pc_sentiment = "Bearish (heavy put buying)"
        elif pc_ratio < 0.8:
            pc_sentiment = "Bullish (heavy call buying)"
        else:
            pc_sentiment = "Neutral"

        embed = discord.Embed(
            title=f"📋 Options Chain — {ticker}",
            description=f"Nearest expiry: **{expiry}**  •  Put/Call Ratio: **{pc_ratio:.2f}** ({pc_sentiment})",
            color=discord.Color.from_str("#58a6ff"),
        )

        # Top 3 calls by open interest
        if calls:
            call_lines = []
            for c in calls[:3]:
                call_lines.append(
                    f"Strike **${c['strike']:.0f}**  |  Last: ${c['last']:.2f}  |  "
                    f"IV: {c['iv']:.0%}  |  OI: {c['oi']:,}"
                )
            embed.add_field(
                name="📈 Top Calls (by Open Interest)",
                value="\n".join(call_lines),
                inline=False,
            )
        else:
            embed.add_field(name="📈 Top Calls", value="No data", inline=False)

        # Top 3 puts by open interest
        if puts:
            put_lines = []
            for p in puts[:3]:
                put_lines.append(
                    f"Strike **${p['strike']:.0f}**  |  Last: ${p['last']:.2f}  |  "
                    f"IV: {p['iv']:.0%}  |  OI: {p['oi']:,}"
                )
            embed.add_field(
                name="📉 Top Puts (by Open Interest)",
                value="\n".join(put_lines),
                inline=False,
            )
        else:
            embed.add_field(name="📉 Top Puts", value="No data", inline=False)

        embed.set_footer(text="Martin's Stocks • Options data via Yahoo Finance • Prices may be delayed")
        apply_admin_flair(embed, interaction)
        await interaction.followup.send(embed=embed)

    # ─── /sentiment ───────────────────────────────────────────────────────────

    @app_commands.command(name="sentiment", description="AI-powered news sentiment analysis for a ticker")
    @app_commands.describe(ticker="Stock ticker symbol (e.g. AAPL)")
    async def sentiment_cmd(self, interaction: discord.Interaction, ticker: str):
        """Score recent headlines through Groq and show an aggregated sentiment bar."""
        await interaction.response.defer()
        ticker = ticker.upper().strip()

        headlines = await news.get_stock_news(ticker, limit=10)

        if not headlines:
            await interaction.followup.send(embed=error_embed(f"No recent news found for **{ticker}**."))
            return

        # Score each headline in parallel — Groq is async so gather is safe
        scored = await asyncio.gather(
            *[_score_headline(h) for h in headlines[:10]],
            return_exceptions=True,
        )

        def safe_score(v):
            return v if not isinstance(v, Exception) else {"label": "Neutral", "score": 50}

        scored_items = [safe_score(s) for s in scored]

        # Aggregate
        labels = [s["label"] for s in scored_items]
        total = len(labels)
        bull_count = labels.count("Positive")
        bear_count = labels.count("Negative")
        neut_count = labels.count("Neutral")

        bull_pct = bull_count / total * 100
        bear_pct = bear_count / total * 100
        neut_pct = neut_count / total * 100

        # Weighted sentiment score 0-100 (Positive=100, Neutral=50, Negative=0)
        raw_scores = [s["score"] for s in scored_items]
        weighted_score = sum(raw_scores) / len(raw_scores)

        # ASCII sentiment bar (20 chars wide)
        bar_length = 20
        bull_fill = round(bull_pct / 100 * bar_length)
        bear_fill = round(bear_pct / 100 * bar_length)
        neut_fill = bar_length - bull_fill - bear_fill
        sentiment_bar = "🟢" * bull_fill + "⚪" * neut_fill + "🔴" * bear_fill

        # Overall label
        if weighted_score >= 65:
            overall_label = "Bullish"
            embed_color = discord.Color.from_str("#2ea043")
        elif weighted_score <= 35:
            overall_label = "Bearish"
            embed_color = discord.Color.from_str("#da3633")
        else:
            overall_label = "Neutral"
            embed_color = discord.Color.gold()

        embed = discord.Embed(
            title=f"💬 News Sentiment — {ticker}",
            description=(
                f"Overall: **{overall_label}**  •  Score: **{weighted_score:.0f}/100**\n\n"
                f"{sentiment_bar}\n"
                f"🟢 Bullish {bull_pct:.0f}%  ⚪ Neutral {neut_pct:.0f}%  🔴 Bearish {bear_pct:.0f}%"
            ),
            color=embed_color,
        )

        # Headline breakdown
        breakdown_lines = []
        for item, scored_item in zip(headlines[:10], scored_items):
            icon = {"Positive": "🟢", "Negative": "🔴", "Neutral": "⚪"}.get(scored_item["label"], "⚪")
            breakdown_lines.append(f"{icon} {truncate(item['title'], 75)}")

        embed.add_field(
            name=f"📰 Headline Breakdown ({total} articles scored)",
            value="\n".join(breakdown_lines),
            inline=False,
        )

        embed.set_footer(text="Martin's Stocks • Sentiment via Groq LLaMA 3.3 70B • Data: Yahoo Finance / Google News")
        apply_admin_flair(embed, interaction)
        await interaction.followup.send(embed=embed)


# ─── Sync helpers (run in thread pool) ────────────────────────────────────────

def _compute_portfolio_metrics(histories: dict) -> dict:
    """
    Compute annualised portfolio metrics from a dict of {ticker: DataFrame}.
    Returns total_return, individual_returns, volatility, sharpe, max_drawdown.
    """
    import pandas as pd
    import numpy as np

    # Build daily returns DataFrame, one column per ticker
    returns_df = pd.DataFrame({
        t: h["Close"].pct_change().dropna()
        for t, h in histories.items()
    }).dropna()

    n = len(histories)
    weight = 1 / n

    # Portfolio daily return = equal-weight average
    port_daily = returns_df.mean(axis=1)

    # Annualised metrics (252 trading days)
    total_return = float((1 + port_daily).prod() - 1) * 100
    ann_vol = float(port_daily.std() * (252 ** 0.5)) * 100
    ann_return = float(port_daily.mean() * 252) * 100
    sharpe = (ann_return / 100 - RISK_FREE_RATE) / (ann_vol / 100) if ann_vol else 0

    # Max drawdown
    cumulative = (1 + port_daily).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = float(drawdown.min()) * 100  # negative number

    # Individual 1-year returns
    individual = {}
    for t, h in histories.items():
        prices = h["Close"].dropna()
        if len(prices) >= 2:
            individual[t] = float((prices.iloc[-1] / prices.iloc[0] - 1) * 100)
        else:
            individual[t] = 0.0

    return {
        "portfolio_return": total_return,
        "volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "individual_returns": individual,
    }


def _fetch_options_chain(ticker: str) -> dict | None:
    """
    Fetch the nearest-expiry options chain for a ticker.
    Returns dict with expiry, calls, puts, pc_ratio — or None on failure.
    """
    import yfinance as yf

    try:
        t = yf.Ticker(ticker)
        expirations = t.options
        if not expirations:
            return None

        expiry = expirations[0]  # Nearest expiry
        chain = t.option_chain(expiry)
        calls_df = chain.calls
        puts_df = chain.puts

        def _top_by_oi(df, n=3) -> list[dict]:
            if df is None or df.empty:
                return []
            df_sorted = df.sort_values("openInterest", ascending=False)
            rows = []
            for _, row in df_sorted.head(n).iterrows():
                rows.append({
                    "strike": float(row.get("strike", 0)),
                    "last": float(row.get("lastPrice", 0)),
                    "iv": float(row.get("impliedVolatility", 0)),
                    "oi": int(row.get("openInterest", 0)),
                })
            return rows

        top_calls = _top_by_oi(calls_df)
        top_puts = _top_by_oi(puts_df)

        # Put/call ratio by total open interest
        total_call_oi = int(calls_df["openInterest"].sum()) if not calls_df.empty else 0
        total_put_oi  = int(puts_df["openInterest"].sum()) if not puts_df.empty else 0
        pc_ratio = (total_put_oi / total_call_oi) if total_call_oi > 0 else 0

        return {
            "expiry": expiry,
            "calls": top_calls,
            "puts": top_puts,
            "pc_ratio": pc_ratio,
        }
    except Exception as e:
        logger.error(f"_fetch_options_chain({ticker}): {e}")
        return None


async def _score_headline(article: dict) -> dict:
    """
    Score a news headline as Positive/Negative/Neutral using Groq.
    Returns dict with label and score (0-100).
    """
    title = article.get("title", "")
    if not title:
        return {"label": "Neutral", "score": 50}

    prompt = (
        f"Classify this financial news headline sentiment for a stock investor.\n"
        f"Headline: {title}\n\n"
        f"Respond in EXACTLY this format:\n"
        f"LABEL: [Positive / Negative / Neutral]\n"
        f"SCORE: [0-100 where 0=very bearish, 50=neutral, 100=very bullish]"
    )

    client = groq._get_client()
    try:
        response = await client.chat.completions.create(
            model=groq.MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial sentiment classifier. Respond only in the requested format.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=30,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        label = "Neutral"
        score = 50
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("LABEL:"):
                val = line[6:].strip().capitalize()
                if val in ("Positive", "Negative", "Neutral"):
                    label = val
            elif line.startswith("SCORE:"):
                try:
                    score = min(100, max(0, int(line.split(":")[1].strip().split()[0])))
                except Exception:
                    pass
        return {"label": label, "score": score}
    except Exception as e:
        logger.warning(f"_score_headline: {e}")
        return {"label": "Neutral", "score": 50}


async def setup(bot):
    await bot.add_cog(ProCommandsCog(bot))
