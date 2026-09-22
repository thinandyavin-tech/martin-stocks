"""
Personal tools commands:
/watchlist add|remove|show
/alert set|list|remove
/news
/summarize
"""

from __future__ import annotations

import asyncio
import io
import logging

import discord
from discord import app_commands
from discord.ext import commands

from services import market_data as md
from services import news_service as news
from services import groq_service as groq
from utils import technical as ta
from utils.charts import generate_summary_infographic
from utils.permissions import apply_admin_flair
from utils.helpers import (
    fmt_price, fmt_pct, rsi_label, change_arrow, change_color_str,
    base_embed, error_embed, truncate,
)

logger = logging.getLogger(__name__)


class SummarizeView(discord.ui.View):
    """Per-article Summarize buttons for /news."""

    def __init__(self, articles: list[dict]):
        super().__init__(timeout=180)
        for i, article in enumerate(articles[:5]):
            btn = discord.ui.Button(
                label=f"Summarize #{i + 1}",
                style=discord.ButtonStyle.secondary,
                custom_id=f"news_summarize_{i}",
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
                await interaction.followup.send("No URL available for this article.", ephemeral=True)
                return
            data = await groq.fetch_and_summarize_url_rich(url)
            img = await generate_summary_infographic(data)
            embed = discord.Embed(
                title="📰 AI Summary",
                description=f"**{truncate(title, 100)}**\n\n{data['summary_text']}",
                color=discord.Color.from_str("#58a6ff"),
            )
            embed.add_field(name="Source", value=f"[Read full article]({url})", inline=False)
            if img:
                embed.set_image(url="attachment://summary.png")
                await interaction.followup.send(
                    embed=embed,
                    file=discord.File(io.BytesIO(img), filename="summary.png"),
                    ephemeral=True,
                )
            else:
                await interaction.followup.send(embed=embed, ephemeral=True)
        return callback


class PersonalToolsCog(commands.Cog, name="Personal"):
    def __init__(self, bot):
        self.bot = bot

    watchlist_group = app_commands.Group(name="watchlist", description="Manage your personal stock watchlist")
    alert_group = app_commands.Group(name="alert", description="Manage your price alerts")

    # ─── /watchlist ───────────────────────────────────────────────────────────

    @watchlist_group.command(name="add", description="Add a ticker to your watchlist (max 20)")
    @app_commands.describe(ticker="Ticker symbol to add (e.g. AAPL)")
    async def watchlist_add(self, interaction: discord.Interaction, ticker: str):
        await interaction.response.defer(ephemeral=True)
        ticker = ticker.upper().strip()

        price = await md.get_price(ticker)
        if price is None:
            await interaction.followup.send(
                embed=error_embed(f"Could not find a valid ticker for **{ticker}**."),
                ephemeral=True,
            )
            return

        success, msg = await self.bot.db.add_to_watchlist(str(interaction.user.id), ticker)
        color = discord.Color.green() if success else discord.Color.red()
        embed = discord.Embed(description=msg, color=color)
        await interaction.followup.send(embed=embed, ephemeral=True)

    @watchlist_group.command(name="remove", description="Remove a ticker from your watchlist")
    @app_commands.describe(ticker="Ticker symbol to remove")
    async def watchlist_remove(self, interaction: discord.Interaction, ticker: str):
        await interaction.response.defer(ephemeral=True)
        ticker = ticker.upper().strip()
        success, msg = await self.bot.db.remove_from_watchlist(str(interaction.user.id), ticker)
        color = discord.Color.green() if success else discord.Color.red()
        embed = discord.Embed(description=msg, color=color)
        await interaction.followup.send(embed=embed, ephemeral=True)

    @watchlist_group.command(name="show", description="Show your watchlist with live prices and indicators")
    async def watchlist_show(self, interaction: discord.Interaction):
        await interaction.response.defer()
        tickers = await self.bot.db.get_watchlist(str(interaction.user.id))

        if not tickers:
            await interaction.followup.send(
                embed=discord.Embed(
                    description="Your watchlist is empty. Use `/watchlist add` to get started.",
                    color=discord.Color.gold(),
                ),
                ephemeral=True,
            )
            return

        embed = discord.Embed(
            title=f"📋 {interaction.user.display_name}'s Watchlist ({len(tickers)}/20)",
            color=discord.Color.from_str("#58a6ff"),
        )

        async def _get_ticker_data(t):
            info = await md.get_stock_info(t)
            hist = await md.get_history(t, period="1mo")
            ta_data = ta.run_full_analysis(hist) if hist is not None and not hist.empty else {}
            return t, info, ta_data

        results = await asyncio.gather(*[_get_ticker_data(t) for t in tickers], return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                continue
            ticker, info, ta_data = result
            price = ta_data.get("current_price") or info.get("currentPrice") or info.get("regularMarketPrice")
            prev = info.get("previousClose") or info.get("regularMarketPreviousClose")
            chg_pct = ((price - prev) / prev * 100) if (price and prev) else 0
            rsi = ta_data.get("rsi", 50)
            trend = "📈 Bullish" if ta_data.get("above_ema20") else "📉 Bearish"

            embed.add_field(
                name=f"{ticker} — {fmt_price(price)}",
                value=(
                    f"{change_color_str(chg_pct)} {change_arrow(chg_pct)}{abs(chg_pct):.2f}% today\n"
                    f"RSI: {rsi} | Trend: {trend}"
                ),
                inline=True,
            )

        embed.set_footer(text="Martin's Stocks • Live prices via Yahoo Finance")
        apply_admin_flair(embed, interaction)
        await interaction.followup.send(embed=embed)

    # ─── /alert ───────────────────────────────────────────────────────────────

    @alert_group.command(name="set", description="Set a price alert for a stock")
    @app_commands.describe(
        ticker="Stock ticker (e.g. AAPL)",
        price="Target price to alert at",
        direction="Alert when price goes above or below",
    )
    @app_commands.choices(direction=[
        app_commands.Choice(name="Above (price rises to target)", value="above"),
        app_commands.Choice(name="Below (price falls to target)", value="below"),
    ])
    async def alert_set(
        self,
        interaction: discord.Interaction,
        ticker: str,
        price: float,
        direction: str,
    ):
        await interaction.response.defer(ephemeral=True)
        ticker = ticker.upper().strip()

        current_price = await md.get_price(ticker)
        if current_price is None:
            await interaction.followup.send(
                embed=error_embed(f"Could not find ticker **{ticker}**."),
                ephemeral=True,
            )
            return

        alert_id = await self.bot.db.add_alert(
            user_id=str(interaction.user.id),
            channel_id=str(interaction.channel_id),
            ticker=ticker,
            target_price=price,
            direction=direction,
        )

        direction_text = "rises above" if direction == "above" else "falls below"
        embed = discord.Embed(
            title="✅ Alert Set",
            description=(
                f"I'll ping you in this channel when **{ticker}** {direction_text} **${price:.2f}**.\n\n"
                f"Current price: **{fmt_price(current_price)}**\n"
                f"Alert ID: `#{alert_id}`"
            ),
            color=discord.Color.green(),
        )
        await interaction.followup.send(embed=embed, ephemeral=True)

    @alert_group.command(name="list", description="List all your active price alerts")
    async def alert_list(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        alerts = await self.bot.db.get_user_alerts(str(interaction.user.id))

        if not alerts:
            await interaction.followup.send(
                embed=discord.Embed(
                    description="You have no active alerts. Use `/alert set` to create one.",
                    color=discord.Color.gold(),
                ),
                ephemeral=True,
            )
            return

        embed = discord.Embed(
            title=f"🔔 Your Active Alerts ({len(alerts)})",
            color=discord.Color.from_str("#58a6ff"),
        )
        for alert in alerts:
            direction_text = "above" if alert["direction"] == "above" else "below"
            embed.add_field(
                name=f"#{alert['id']} — {alert['ticker']}",
                value=(
                    f"Trigger: price goes **{direction_text}** **${alert['target_price']:.2f}**\n"
                    f"Set: {alert['created_at']}"
                ),
                inline=False,
            )
        await interaction.followup.send(embed=embed, ephemeral=True)

    @alert_group.command(name="remove", description="Remove a price alert by ID")
    @app_commands.describe(alert_id="Alert ID from /alert list")
    async def alert_remove(self, interaction: discord.Interaction, alert_id: int):
        await interaction.response.defer(ephemeral=True)
        success = await self.bot.db.remove_alert(str(interaction.user.id), alert_id)
        if success:
            embed = discord.Embed(
                description=f"✅ Alert `#{alert_id}` removed.",
                color=discord.Color.green(),
            )
        else:
            embed = discord.Embed(
                description=f"❌ Alert `#{alert_id}` not found or doesn't belong to you.",
                color=discord.Color.red(),
            )
        await interaction.followup.send(embed=embed, ephemeral=True)

    # ─── /news ────────────────────────────────────────────────────────────────

    @app_commands.command(name="news", description="Top financial headlines with AI summarize buttons")
    async def news_cmd(self, interaction: discord.Interaction):
        await interaction.response.defer()
        articles = await news.get_all_news(limit_per_feed=4)

        if not articles:
            await interaction.followup.send(embed=error_embed("Could not fetch news. Try again in a moment."))
            return

        embed = discord.Embed(
            title="📰 Top Financial Headlines",
            color=discord.Color.from_str("#58a6ff"),
        )

        for i, article in enumerate(articles[:10]):
            embed.add_field(
                name=f"{i + 1}. {truncate(article['title'], 90)}",
                value=(
                    f"*{article['source']}* • {article['published']}\n"
                    f"[Read article]({article['url']})"
                ),
                inline=False,
            )

        embed.set_footer(text="Click Summarize to get an AI summary of any article")
        apply_admin_flair(embed, interaction)
        await interaction.followup.send(embed=embed, view=SummarizeView(articles[:10]))

    # ─── /summarize ───────────────────────────────────────────────────────────

    @app_commands.command(name="summarize", description="AI summary + infographic of any article URL")
    @app_commands.describe(url="URL of the article to summarize")
    async def summarize_cmd(self, interaction: discord.Interaction, url: str):
        await interaction.response.defer()

        if not url.startswith("http"):
            await interaction.followup.send(embed=error_embed("Please provide a valid URL starting with http(s)://"))
            return

        await interaction.followup.send(embed=discord.Embed(
            description="⏳ Fetching and summarizing article...",
            color=discord.Color.from_str("#161b22"),
        ))

        data, img = await asyncio.gather(
            groq.fetch_and_summarize_url_rich(url),
            asyncio.sleep(0),  # placeholder so gather runs
        )
        img = await generate_summary_infographic(data)

        sent_color = {"Bullish": 0x2ea043, "Bearish": 0xda3633}.get(data["sentiment"], 0x58a6ff)

        result_embed = discord.Embed(
            title=f"📰 {data['title']}",
            description=data["summary_text"],
            color=sent_color,
        )
        result_embed.add_field(
            name="Sentiment",
            value=f"{'🟢' if data['sentiment'] == 'Bullish' else '🔴' if data['sentiment'] == 'Bearish' else '⚪'} {data['sentiment']}",
            inline=True,
        )
        result_embed.add_field(
            name="Market Impact",
            value=data.get("market_impact") or "—",
            inline=True,
        )
        if data.get("key_tickers"):
            result_embed.add_field(
                name="Tickers",
                value=" ".join(f"`${t}`" for t in data["key_tickers"][:6]),
                inline=True,
            )
        result_embed.add_field(name="Source", value=f"[View original]({url})", inline=False)
        result_embed.set_footer(text="Martin's Stocks • Summarized by Groq LLaMA 3.3 70B")
        apply_admin_flair(result_embed, interaction)

        if img:
            result_embed.set_image(url="attachment://summary.png")
            await interaction.channel.send(
                embed=result_embed,
                file=discord.File(io.BytesIO(img), filename="summary.png"),
            )
        else:
            await interaction.channel.send(embed=result_embed)


async def setup(bot):
    await bot.add_cog(PersonalToolsCog(bot))
