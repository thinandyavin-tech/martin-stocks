"""
Thai stock market commands.
/stocksth — full analysis for SET-listed stocks (Yahoo Finance .BK tickers).
Prices displayed in THB (฿).
"""

from __future__ import annotations

import asyncio
import io
import logging

import discord
from discord import app_commands
from discord.ext import commands

from services import market_data as md
from services import groq_service as groq
from services import news_service as news
from utils import technical as ta
from utils import charts
from utils.charts import generate_stock_scorecard
from utils.permissions import apply_admin_flair
from utils.helpers import (
    fmt_pct, fmt_large, fmt_shares,
    signal_emoji, signal_color, rsi_label, score_bar,
    error_embed, truncate,
)

logger = logging.getLogger(__name__)

# SET Index — used for relative strength comparison instead of SPY
SET_INDEX = "^SET.BK"

TIMEFRAMES = ["1D", "5D", "1W", "1M", "1Y", "5Y", "MAX"]


def fmt_thb(value) -> str:
    """Format a number as Thai Baht."""
    if value is None:
        return "N/A"
    try:
        f = float(value)
        if f >= 1_000_000_000:
            return f"฿{f/1_000_000_000:.2f}B"
        if f >= 1_000_000:
            return f"฿{f/1_000_000:.2f}M"
        if f >= 1_000:
            return f"฿{f:,.2f}"
        return f"฿{f:.2f}"
    except (TypeError, ValueError):
        return "N/A"


def fmt_thb_large(value) -> str:
    """Format a large THB number (market cap etc)."""
    if value is None:
        return "N/A"
    try:
        f = float(value)
        if f >= 1_000_000_000_000:
            return f"฿{f/1_000_000_000_000:.2f}T"
        if f >= 1_000_000_000:
            return f"฿{f/1_000_000_000:.2f}B"
        if f >= 1_000_000:
            return f"฿{f/1_000_000:.2f}M"
        return f"฿{f:,.0f}"
    except (TypeError, ValueError):
        return "N/A"


class ThaiChartView(discord.ui.View):
    """7-button chart timeframe selector for Thai stocks."""

    def __init__(self, ticker: str, current_period: str = "1D", buy_zones: list = None):
        super().__init__(timeout=300)
        self.ticker = ticker
        self.current_period = current_period
        self.buy_zones = buy_zones or []
        self._refresh()

    def _refresh(self):
        self.clear_items()
        for tf in TIMEFRAMES:
            btn = discord.ui.Button(
                label=tf,
                style=discord.ButtonStyle.primary if tf == self.current_period else discord.ButtonStyle.secondary,
                custom_id=f"th_chart_{self.ticker}_{tf}",
            )
            btn.callback = self._make_callback(tf)
            self.add_item(btn)

    def _make_callback(self, period: str):
        async def callback(interaction: discord.Interaction):
            await interaction.response.defer()
            self.current_period = period
            self._refresh()
            chart_bytes = await charts.generate_chart(self.ticker, period, self.buy_zones)
            if not chart_bytes:
                await interaction.followup.send("Could not generate chart.", ephemeral=True)
                return
            file = discord.File(io.BytesIO(chart_bytes), filename=f"{self.ticker}_{period}.png")
            await interaction.message.edit(attachments=[file], view=self)
        return callback


class ThaiNewsSummarizeView(discord.ui.View):
    """Per-article AI Summarize buttons for Thai stock news."""

    def __init__(self, articles: list[dict]):
        super().__init__(timeout=180)
        for i, article in enumerate(articles[:5]):
            btn = discord.ui.Button(
                label=f"Summarize #{i + 1}",
                style=discord.ButtonStyle.secondary,
                custom_id=f"th_summarize_{i}",
                emoji="🤖",
            )
            btn.callback = self._make_callback(article)
            self.add_item(btn)

    def _make_callback(self, article: dict):
        async def callback(interaction: discord.Interaction):
            await interaction.response.defer(ephemeral=True)
            url = article.get("url", "")
            title = article.get("title", "")
            if not url:
                await interaction.followup.send("No URL available.", ephemeral=True)
                return
            summary = await groq.fetch_and_summarize_url(url)
            embed = discord.Embed(
                title="📰 AI Summary",
                description=f"**{truncate(title, 100)}**\n\n{summary}",
                color=discord.Color.from_str("#58a6ff"),
            )
            embed.add_field(name="Source", value=f"[Read full article]({url})", inline=False)
            await interaction.followup.send(embed=embed, ephemeral=True)
        return callback


async def thai_ticker_autocomplete(
    interaction: discord.Interaction, current: str
) -> list[app_commands.Choice[str]]:
    """Autocomplete for SET-listed Thai stock tickers."""
    if not current:
        return []
    results = await md.search_thai_tickers(current)
    return [
        app_commands.Choice(
            name=f"{r['symbol']} — {r['name'][:45]}" if r.get("name") else r["symbol"],
            value=r["symbol"],
        )
        for r in results[:25]
        if r.get("symbol")
    ]


class ThaiStocksCog(commands.Cog, name="ThaiStocks"):
    def __init__(self, bot):
        self.bot = bot

    @app_commands.command(
        name="stocksth",
        description="Thai stock analysis (SET) — search by ticker or company name",
    )
    @app_commands.describe(query="Thai ticker or company name (e.g. PTT, KBANK, or PTT Public)")
    @app_commands.autocomplete(query=thai_ticker_autocomplete)
    async def stocksth_cmd(self, interaction: discord.Interaction, query: str):
        await interaction.response.defer()

        ticker = await md.resolve_thai_ticker(query)
        if not ticker:
            await interaction.followup.send(embed=discord.Embed(
                title="❌ Not Found",
                description=(
                    f"Could not find a Thai SET-listed stock for **{query}**.\n\n"
                    "Try the ticker symbol directly (e.g. `PTT`, `KBANK`, `CPALL`) "
                    "or the full company name."
                ),
                color=discord.Color.red(),
            ))
            return

        # Fetch all data in parallel
        results = await asyncio.gather(
            md.get_stock_info(ticker),
            md.get_history(ticker, period="1y"),
            md.get_history(SET_INDEX, period="1y"),
            news.get_stock_news(ticker),
            md.get_analyst_data(ticker),
            md.get_earnings(ticker),
            md.get_insider_transactions(ticker),
            md.get_institutional_holders(ticker),
            return_exceptions=True,
        )

        def safe(v):
            return v if not isinstance(v, Exception) else None

        info         = safe(results[0]) or {}
        hist         = safe(results[1])
        set_hist     = safe(results[2])
        news_items   = safe(results[3]) or []
        analyst      = safe(results[4]) or {}
        earnings     = safe(results[5]) or {}
        insiders     = safe(results[6]) or []
        institutions = safe(results[7]) or []

        ta_data = {}
        if hist is not None and not hist.empty:
            ta_data = ta.run_full_analysis(hist, set_hist)

        # ── Display values ────────────────────────────────────────────────────
        display_ticker = ticker.replace(".BK", "")
        name = info.get("longName") or info.get("shortName") or display_ticker

        current_price = ta_data.get("current_price") or info.get("currentPrice") or info.get("regularMarketPrice") or 0
        prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose") or current_price
        change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close else 0
        arrow = "▲" if change_pct >= 0 else "▼"

        pe    = info.get("trailingPE") or info.get("forwardPE")
        beta  = info.get("beta")
        eps   = info.get("trailingEps")
        div_y = info.get("dividendYield")

        low_52  = ta_data.get("low_52w") or info.get("fiftyTwoWeekLow", 0)
        high_52 = ta_data.get("high_52w") or info.get("fiftyTwoWeekHigh", 0)
        range_pct = ta_data.get("range_pct", 0) or 0
        bar_filled = round(range_pct / 10)
        range_bar = "▏" + "─" * bar_filled + "●" + "─" * (10 - bar_filled) + "▕"

        rsi_val = ta_data.get("rsi", 50)

        # ── AI analysis ───────────────────────────────────────────────────────
        ai_result = await groq.analyze_stock(ticker, {
            "technical": ta_data,
            "smart_money": {"insiders": insiders, "institutions": institutions},
            "news": news_items,
            "info": info,
        })

        buy_zones = ai_result.get("buy_zones", [])
        signal = ai_result.get("overall_signal", "NEUTRAL")

        chart_bytes, scorecard_bytes = await asyncio.gather(
            charts.generate_chart(ticker, "1D", buy_zones),
            generate_stock_scorecard(display_ticker, ai_result, ta_data, info),
        )

        # ── Main embed ────────────────────────────────────────────────────────
        main_embed = discord.Embed(
            title=f"{signal_emoji(signal)} {display_ticker} — {name}",
            description=(
                f"**{fmt_thb(current_price)}**  {arrow} **{abs(change_pct):.2f}%** today\n"
                f"AI Signal: **{signal}**  •  Risk: **{ai_result.get('risk_level', 'Medium')}**\n"
                f"*SET-listed • Prices in Thai Baht (฿) 🇹🇭*"
            ),
            color=signal_color(signal),
        )
        main_embed.set_footer(
            text=f"Martin's Stocks • {ticker} • Data: Yahoo Finance / SET + Groq LLaMA 3.3 70B"
        )

        embed_pe  = round(pe, 1) if pe else "N/A"
        embed_eps = fmt_thb(eps)
        embed_div = fmt_pct(div_y * 100) if div_y else "None"

        main_embed.add_field(
            name="📈 Fundamentals (THB)",
            value=(
                f"Market Cap:  **{fmt_thb_large(info.get('marketCap'))}**\n"
                f"P/E (TTM):   **{embed_pe}**\n"
                f"EPS (TTM):   **{embed_eps}**\n"
                f"Beta:        **{round(beta, 2) if beta else 'N/A'}**\n"
                f"Div. Yield:  **{embed_div}**\n"
                f"Sector:      {info.get('sector', 'N/A')}"
            ),
            inline=True,
        )

        main_embed.add_field(
            name="📊 Technicals",
            value=(
                f"RSI:     **{rsi_label(rsi_val)}** ({rsi_val:.0f})\n"
                f"EMA 20:  `{ta_data.get('ema20_label', 'N/A')}`\n"
                f"EMA 50:  `{ta_data.get('ema50_label', 'N/A')}`\n"
                f"EMA 200: `{ta_data.get('ema200_label', 'N/A')}`\n"
                f"Volume:  **{ta_data.get('volume_ratio', 'N/A')}x** avg\n"
                f"Pattern: `{ta_data.get('pattern', 'None')}`\n"
                f"RS/SET:  **{ta_data.get('relative_strength', 'N/A')}x**"
            ),
            inline=True,
        )

        main_embed.add_field(
            name="📏 52-Week Range",
            value=(
                f"Low:      **{fmt_thb(low_52)}**\n"
                f"High:     **{fmt_thb(high_52)}**\n"
                f"Position: **{range_pct}%**\n"
                f"`{range_bar}`"
            ),
            inline=True,
        )

        main_embed.add_field(
            name="🏆 Signal Layers",
            value=(
                f"Momentum    {score_bar(ai_result.get('momentum_score', 5))}\n"
                f"Smart Money {score_bar(ai_result.get('smart_money_score', 5))}\n"
                f"News        {score_bar(ai_result.get('news_score', 5))}\n"
                f"Technical   {score_bar(ai_result.get('technical_score', 5))}"
            ),
            inline=True,
        )

        thesis = ai_result.get("thesis", "")
        if thesis:
            main_embed.add_field(name="🤖 AI Thesis", value=truncate(thesis, 500), inline=False)

        zones_str = "  ·  ".join(buy_zones) if buy_zones else "N/A"
        main_embed.add_field(
            name="🎯 Entry Strategy",
            value=(
                f"**Buy Zones:** {zones_str}\n"
                f"{truncate(ai_result.get('entry_points', ''), 200)}"
            ),
            inline=False,
        )

        apply_admin_flair(main_embed, interaction)

        files_main = []
        if scorecard_bytes:
            main_embed.set_image(url=f"attachment://{display_ticker}_scorecard.png")
            files_main.append(
                discord.File(io.BytesIO(scorecard_bytes), filename=f"{display_ticker}_scorecard.png")
            )

        await interaction.followup.send(embed=main_embed, files=files_main)

        # ── Smart money embed ─────────────────────────────────────────────────
        sm_embed = discord.Embed(
            title=f"🏦 Smart Money — {display_ticker}",
            color=discord.Color.from_str("#58a6ff"),
        )
        if institutions:
            sm_embed.add_field(
                name="Institutional Holders",
                value="\n".join(
                    f"• **{i['holder']}** — {i['pct_out']:.1f}% ({fmt_thb_large(i['value'])})"
                    for i in institutions[:5]
                ),
                inline=False,
            )
        insider_lines = []
        for ins in (insiders or [])[:8]:
            is_buy = "buy" in ins.get("transaction", "").lower() or "purchase" in ins.get("transaction", "").lower()
            emoji  = "🟢 BUY" if is_buy else "🔴 SELL"
            insider_lines.append(f"{emoji} **{ins['name']}** ({ins['title']}) — {fmt_shares(ins['shares'])} shares")
        sm_embed.add_field(
            name="Insider Transactions",
            value="\n".join(insider_lines) if insider_lines else "No recent filings",
            inline=False,
        )

        # ── News embed ────────────────────────────────────────────────────────
        news_embed = discord.Embed(
            title=f"📰 Latest News — {display_ticker}",
            color=discord.Color.from_str("#21262d"),
        )
        for i, a in enumerate(news_items[:5]):
            news_embed.add_field(
                name=f"{i+1}. {truncate(a['title'], 90)}",
                value=f"*{a['source']}* • {a['published']}\n[Read article]({a['url']})",
                inline=False,
            )

        chart_view = ThaiChartView(ticker, current_period="1D", buy_zones=buy_zones)
        news_view  = ThaiNewsSummarizeView(news_items) if news_items else discord.ui.View()

        await interaction.channel.send(embeds=[sm_embed, news_embed], view=news_view)

        if chart_bytes:
            chart_file = discord.File(io.BytesIO(chart_bytes), filename=f"{display_ticker}_1D.png")
            await interaction.channel.send(file=chart_file, view=chart_view)


async def setup(bot):
    await bot.add_cog(ThaiStocksCog(bot))
