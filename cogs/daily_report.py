"""
Daily AI Report cog.
- /report set — designate this channel for daily morning reports
- /report remove — stop reports in this channel
- send_daily_report() — called by APScheduler each morning at 8AM UTC
  Scans 50 stocks, runs Groq AI on each, posts formatted Discord embeds.
"""

import asyncio
import logging
from datetime import datetime, timezone

import discord
from discord import app_commands
from discord.ext import commands

from services import market_data as md
from services import groq_service as groq
from services import news_service as news
from utils import technical as ta
from utils.helpers import (
    fmt_price, fmt_pct, fmt_large,
    signal_emoji, change_arrow, change_color_str, truncate,
)
from services.market_data import STOCK_UNIVERSE

logger = logging.getLogger(__name__)

# Number of stocks included in the daily report
REPORT_UNIVERSE = STOCK_UNIVERSE[:30]  # top 30 most relevant


class DailyReportCog(commands.Cog, name="DailyReport"):
    def __init__(self, bot):
        self.bot = bot

    report_group = app_commands.Group(name="report", description="Configure the daily AI market report")

    @report_group.command(name="set", description="Post the daily AI report in this channel every morning (8AM UTC)")
    async def report_set(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)

        if not interaction.guild:
            await interaction.followup.send(
                embed=discord.Embed(description="❌ Server channels only.", color=discord.Color.red()),
                ephemeral=True,
            )
            return

        await self.bot.db.set_daily_report_channel(
            str(interaction.channel_id),
            str(interaction.guild_id),
        )

        embed = discord.Embed(
            title="✅ Daily AI Report Configured",
            description=(
                f"The daily AI market report will be posted in <#{interaction.channel_id}> every morning at **8:00 AM UTC**.\n\n"
                "Each report covers:\n"
                "• 🌍 Macro overview (SPY, NASDAQ, VIX)\n"
                "• 📰 Top news headlines\n"
                "• 🤖 AI-scored signals for 30 major stocks\n"
                "• 🎯 Buy zones and tactical setups\n\n"
                "Use `/report remove` to stop reports."
            ),
            color=discord.Color.green(),
        )
        await interaction.followup.send(embed=embed, ephemeral=True)

    @report_group.command(name="remove", description="Stop daily AI reports in this channel")
    async def report_remove(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        await self.bot.db.remove_daily_report_channel(str(interaction.channel_id))
        embed = discord.Embed(
            description=f"🔇 Daily reports disabled in <#{interaction.channel_id}>.",
            color=discord.Color.orange(),
        )
        await interaction.followup.send(embed=embed, ephemeral=True)

    @report_group.command(name="now", description="Generate and send the daily report right now (for testing)")
    @app_commands.checks.has_permissions(manage_channels=True)
    async def report_now(self, interaction: discord.Interaction):
        await interaction.response.defer()
        await interaction.followup.send(
            embed=discord.Embed(
                description="⏳ Generating AI report... this takes 1-2 minutes.",
                color=discord.Color.from_str("#161b22"),
            )
        )
        await send_daily_report(self.bot, channel_override=interaction.channel)


async def send_daily_report(bot, channel_override=None):
    """
    Core report logic. Fetches macro data + news, runs AI analysis on REPORT_UNIVERSE,
    then posts formatted embeds to all configured channels.
    """
    logger.info("Starting daily report generation...")
    start_time = datetime.now(tz=timezone.utc)

    # ─── Step 1: Fetch macro and news ────────────────────────────────────────
    macro_data, all_news, spy_hist = await asyncio.gather(
        md.get_macro_data(),
        news.get_all_news(limit_per_feed=3),
        md.get_history("SPY", period="1y"),
        return_exceptions=True,
    )
    macro_data = macro_data if not isinstance(macro_data, Exception) else {}
    all_news = all_news if not isinstance(all_news, Exception) else []
    spy_hist = spy_hist if not isinstance(spy_hist, Exception) else None

    # ─── Step 2: Analyze stocks ───────────────────────────────────────────────
    async def _analyze_ticker(ticker: str) -> dict:
        try:
            info = await md.get_stock_info(ticker)
            hist = await md.get_history(ticker, period="1y")
            if hist is None or hist.empty:
                return None
            ta_data = ta.run_full_analysis(hist, spy_hist)
            ticker_news = [
                a for a in all_news
                if ticker.lower() in a.get("title", "").lower()
            ][:3]
            ai_result = await groq.analyze_for_daily_report(
                ticker,
                {"technical": ta_data, "info": info, "news": ticker_news},
            )
            return {
                "ticker": ticker,
                "ta": ta_data,
                "info": info,
                "ai": ai_result,
            }
        except Exception as e:
            logger.warning(f"daily_report analyze {ticker}: {e}")
            return None

    # Batched to avoid rate limits
    stock_results = []
    batch_size = 5
    for i in range(0, len(REPORT_UNIVERSE), batch_size):
        batch = REPORT_UNIVERSE[i:i + batch_size]
        batch_results = await asyncio.gather(*[_analyze_ticker(t) for t in batch])
        stock_results.extend([r for r in batch_results if r])
        if i + batch_size < len(REPORT_UNIVERSE):
            await asyncio.sleep(2)  # be kind to rate limits

    # ─── Step 3: Build embeds ─────────────────────────────────────────────────

    date_str = start_time.strftime("%A, %B %d, %Y")

    # Header embed
    header_embed = discord.Embed(
        title=f"🌅 Morning Market Report — {date_str}",
        description=(
            "Good morning! Here's your AI-powered daily market briefing.\n"
            f"Analysis powered by **Groq LLaMA 3.3 70B** | {len(stock_results)} stocks analyzed"
        ),
        color=discord.Color.from_str("#f0883e"),
    )

    # Macro section
    if macro_data:
        spy = macro_data.get("S&P 500", {})
        ndaq = macro_data.get("NASDAQ", {})
        vix = macro_data.get("VIX", {})
        btc = macro_data.get("Bitcoin", {})

        def _fmt_item(item):
            if not item or not item.get("price"):
                return "N/A"
            p = item["price"]
            c = item.get("change_pct", 0)
            cc = change_color_str(c)
            return f"{fmt_price(p)} {cc} {change_arrow(c)}{abs(c):.2f}%"

        header_embed.add_field(
            name="🌍 Macro Overview",
            value=(
                f"S&P 500: **{_fmt_item(spy)}**\n"
                f"NASDAQ: **{_fmt_item(ndaq)}**\n"
                f"VIX: **{vix.get('price', 'N/A'):.2f}** | BTC: **{_fmt_item(btc)}**"
            ),
            inline=False,
        )

    # Top headlines
    if all_news:
        news_lines = [
            f"• [{truncate(a['title'], 80)}]({a['url']}) *— {a['source']}*"
            for a in all_news[:5]
        ]
        header_embed.add_field(
            name="📰 Top Headlines",
            value="\n".join(news_lines),
            inline=False,
        )

    header_embed.set_footer(text="Martin's Stocks • Scroll down for individual stock signals ↓")

    # ─── Signal categorization ────────────────────────────────────────────────
    strong_buys = []
    buys = []
    sells_strong = []
    neutrals = []

    for r in stock_results:
        sig = r["ai"].get("signal", "NEUTRAL").upper()
        if sig == "STRONG BUY":
            strong_buys.append(r)
        elif sig == "BUY":
            buys.append(r)
        elif sig in ("SELL", "STRONG SELL"):
            sells_strong.append(r)
        else:
            neutrals.append(r)

    # Sort by AI score descending
    for lst in [strong_buys, buys, sells_strong, neutrals]:
        lst.sort(key=lambda x: x["ai"].get("score", 0), reverse=True)

    # ─── Build stock signal embed ─────────────────────────────────────────────
    signals_embed = discord.Embed(
        title="🤖 AI Stock Signals Today",
        color=discord.Color.from_str("#2ea043"),
    )

    def _stock_line(r: dict) -> str:
        ticker = r["ticker"]
        ai = r["ai"]
        ta_d = r["ta"]
        info = r["info"]
        price = ta_d.get("current_price") or info.get("currentPrice", "?")
        sig = ai.get("signal", "NEUTRAL")
        score = ai.get("score", 5)
        thesis = truncate(ai.get("thesis", ""), 120)
        buy_zone = ai.get("buy_zone", "N/A")
        rsi = ta_d.get("rsi", "?")
        return (
            f"**{ticker}** `{fmt_price(price)}` {signal_emoji(sig)} Score: {score}/10\n"
            f"RSI: {rsi} | Buy Zone: {buy_zone}\n"
            f"_{thesis}_"
        )

    if strong_buys:
        lines = [_stock_line(r) for r in strong_buys[:5]]
        signals_embed.add_field(
            name=f"🚀 STRONG BUY ({len(strong_buys)})",
            value="\n\n".join(lines)[:1024],
            inline=False,
        )
    if buys:
        lines = [_stock_line(r) for r in buys[:5]]
        signals_embed.add_field(
            name=f"🟢 BUY ({len(buys)})",
            value="\n\n".join(lines)[:1024],
            inline=False,
        )
    if sells_strong:
        lines = [_stock_line(r) for r in sells_strong[:3]]
        signals_embed.add_field(
            name=f"🔴 SELL / STRONG SELL ({len(sells_strong)})",
            value="\n\n".join(lines)[:1024],
            inline=False,
        )

    # ─── Top picks detail embeds ──────────────────────────────────────────────
    top_picks = (strong_buys + buys)[:3]
    detail_embeds = []

    for r in top_picks:
        ticker = r["ticker"]
        ai = r["ai"]
        ta_d = r["ta"]
        info = r["info"]
        price = ta_d.get("current_price") or info.get("currentPrice", "N/A")
        sig = ai.get("signal", "NEUTRAL")

        detail = discord.Embed(
            title=f"{signal_emoji(sig)} {ticker} — {info.get('longName') or ticker}",
            description=ai.get("thesis", ""),
            color=discord.Color.green(),
        )
        detail.add_field(
            name="📊 Key Metrics",
            value=(
                f"Price: **{fmt_price(price)}**\n"
                f"RSI: {ta_d.get('rsi', 'N/A')} | "
                f"OBV: {ta_d.get('obv_trend', 'N/A')}\n"
                f"Vol: {ta_d.get('volume_ratio', 'N/A')}x avg\n"
                f"EMA: {'Bullish ✅' if ta_d.get('above_ema20') else 'Bearish ❌'}"
            ),
            inline=True,
        )
        detail.add_field(
            name="🎯 Entry",
            value=f"Buy Zone: **{ai.get('buy_zone', 'N/A')}**\nSignal Score: **{ai.get('score', 5)}/10**",
            inline=True,
        )
        detail_embeds.append(detail)

    elapsed = (datetime.now(tz=timezone.utc) - start_time).total_seconds()
    signals_embed.set_footer(text=f"Generated in {elapsed:.1f}s • Martin's Stocks")

    # ─── Dispatch to channels ─────────────────────────────────────────────────
    if channel_override:
        channels = [channel_override]
    else:
        channel_ids = await bot.db.get_daily_report_channels()
        channels = []
        for cid in channel_ids:
            ch = bot.get_channel(int(cid))
            if ch:
                channels.append(ch)

    all_embeds = [header_embed, signals_embed] + detail_embeds[:3]

    for channel in channels:
        try:
            await channel.send(embeds=all_embeds)
            logger.info(f"Daily report sent to #{channel.name} ({channel.id})")
        except discord.Forbidden:
            logger.warning(f"No permission to post in channel {channel.id}")
        except Exception as e:
            logger.error(f"Failed to send daily report to {channel.id}: {e}")


async def setup(bot):
    await bot.add_cog(DailyReportCog(bot))
