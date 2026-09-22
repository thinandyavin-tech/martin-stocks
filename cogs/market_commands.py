"""
Market intelligence commands:
/movers, /macro, /fear, /screener
"""

from __future__ import annotations

import asyncio
import logging

import discord
from discord import app_commands
from discord.ext import commands

from services import market_data as md
from services import groq_service as groq
from utils import technical as ta
from utils.permissions import apply_admin_flair
from utils.helpers import (
    fmt_price, fmt_pct, fmt_large,
    change_arrow, change_color_str,
    vix_label, fear_greed_bar,
    screener_signal_emoji, base_embed, error_embed,
)
from services.market_data import STOCK_UNIVERSE

logger = logging.getLogger(__name__)


class MarketCog(commands.Cog, name="Market"):
    def __init__(self, bot):
        self.bot = bot

    # ─── /movers ─────────────────────────────────────────────────────────────

    @app_commands.command(name="movers", description="Top 5 gainers and losers today")
    async def movers_cmd(self, interaction: discord.Interaction):
        await interaction.response.defer()
        data = await md.get_movers()

        embed = discord.Embed(
            title="📊 Top Movers — Today",
            color=discord.Color.from_str("#58a6ff"),
        )

        gainers = data.get("gainers", [])
        losers = data.get("losers", [])

        if gainers:
            lines = [
                f"{i}. **{g['ticker']}** — {fmt_price(g['price'])} 🟢 +{g['change_pct']:.2f}%"
                for i, g in enumerate(gainers, 1)
            ]
            embed.add_field(name="🚀 Top Gainers", value="\n".join(lines), inline=True)
        else:
            embed.add_field(name="🚀 Top Gainers", value="Data unavailable", inline=True)

        if losers:
            lines = [
                f"{i}. **{l['ticker']}** — {fmt_price(l['price'])} 🔴 {l['change_pct']:.2f}%"
                for i, l in enumerate(losers, 1)
            ]
            embed.add_field(name="📉 Top Losers", value="\n".join(lines), inline=True)
        else:
            embed.add_field(name="📉 Top Losers", value="Data unavailable", inline=True)

        embed.set_footer(text="Martin's Stocks • Data via Yahoo Finance (15min delay)")
        apply_admin_flair(embed, interaction)
        await interaction.followup.send(embed=embed)

    # ─── /macro ──────────────────────────────────────────────────────────────

    @app_commands.command(name="macro", description="Live macro dashboard: indices, VIX, crypto, commodities")
    async def macro_cmd(self, interaction: discord.Interaction):
        await interaction.response.defer()
        data = await md.get_macro_data()

        embed = discord.Embed(
            title="🌍 Macro Dashboard",
            color=discord.Color.from_str("#58a6ff"),
        )

        categories = {
            "📈 US Indices": ["S&P 500", "NASDAQ", "Dow Jones"],
            "😨 Volatility": ["VIX"],
            "🪙 Crypto": ["Bitcoin", "Ethereum"],
            "🥇 Commodities": ["Gold", "Oil (WTI)"],
        }

        for category, names in categories.items():
            lines = []
            for name in names:
                item = data.get(name)
                if not item:
                    continue
                price = item.get("price")
                chg = item.get("change_pct", 0)
                cc = change_color_str(chg)
                arrow = change_arrow(chg)
                if name == "VIX":
                    label, _ = vix_label(price) if price else ("N/A", "")
                    lines.append(f"**{name}**: {price:.2f} {cc} {arrow}{abs(chg):.2f}% — *{label}*")
                else:
                    lines.append(f"**{name}**: {fmt_price(price)} {cc} {arrow}{abs(chg):.2f}%")
            if lines:
                embed.add_field(name=category, value="\n".join(lines), inline=False)

        embed.set_footer(text="Martin's Stocks • Live data via Yahoo Finance")
        apply_admin_flair(embed, interaction)
        await interaction.followup.send(embed=embed)

    # ─── /fear ───────────────────────────────────────────────────────────────

    @app_commands.command(name="fear", description="VIX-based Fear & Greed meter with tactical tip")
    async def fear_cmd(self, interaction: discord.Interaction):
        await interaction.response.defer()
        vix = await md.get_vix()

        if vix is None:
            await interaction.followup.send(embed=error_embed("Could not fetch VIX data."))
            return

        label, tip = vix_label(vix)
        bar = fear_greed_bar(vix)

        color_map = {
            "Extreme Greed": discord.Color.from_str("#2ea043"),
            "Greed": discord.Color.green(),
            "Neutral": discord.Color.gold(),
            "Fear": discord.Color.orange(),
            "High Fear": discord.Color.red(),
            "Extreme Fear": discord.Color.from_str("#da3633"),
            "Market Panic": discord.Color.from_str("#8b0000"),
        }
        color = color_map.get(label, discord.Color.blurple())

        embed = discord.Embed(
            title="😱 Fear & Greed Meter",
            color=color,
        )
        embed.add_field(
            name="Current Reading",
            value=(
                f"**VIX Level:** `{vix:.2f}`\n"
                f"**Sentiment:** 🎯 **{label}**\n\n"
                f"Extreme Fear `◄──────────────►` Extreme Greed\n"
                f"{bar}"
            ),
            inline=False,
        )
        embed.add_field(
            name="💡 Tactical Tip",
            value=tip,
            inline=False,
        )
        embed.set_footer(text="Martin's Stocks • VIX data via Yahoo Finance")
        apply_admin_flair(embed, interaction)
        await interaction.followup.send(embed=embed)

    # ─── /screener ───────────────────────────────────────────────────────────

    @app_commands.command(name="screener", description="Scan 50 stocks for technical setups")
    async def screener_cmd(self, interaction: discord.Interaction):
        await interaction.response.defer()

        await interaction.followup.send(
            embed=discord.Embed(
                description=f"⏳ Scanning {len(STOCK_UNIVERSE)} stocks... this takes ~20 seconds.",
                color=discord.Color.from_str("#161b22"),
            )
        )

        spy_hist = await md.get_history("SPY", period="1y")

        async def _analyze_one(ticker: str):
            try:
                hist = await md.get_history(ticker, period="1y")
                if hist is None or hist.empty:
                    return ticker, None
                ta_data = ta.run_full_analysis(hist, spy_hist)
                signal = ta.screener_signal(ta_data)
                return ticker, (signal, ta_data)
            except Exception as e:
                logger.warning(f"screener {ticker}: {e}")
                return ticker, None

        # Run in batches of 10 to avoid rate limits
        results = {}
        batch_size = 10
        for i in range(0, len(STOCK_UNIVERSE), batch_size):
            batch = STOCK_UNIVERSE[i:i + batch_size]
            batch_results = await asyncio.gather(*[_analyze_one(t) for t in batch])
            for ticker, result in batch_results:
                results[ticker] = result
            if i + batch_size < len(STOCK_UNIVERSE):
                await asyncio.sleep(1)

        categories = {
            "Oversold Bounce": [],
            "Momentum": [],
            "Breakout Watch": [],
            "Overbought": [],
        }
        for ticker, result in results.items():
            if result and result[0]:
                signal, ta_data = result
                categories[signal].append((ticker, ta_data))

        embed = discord.Embed(
            title=f"🔬 Market Screener — {len(STOCK_UNIVERSE)} Stocks Scanned",
            color=discord.Color.from_str("#58a6ff"),
        )

        has_results = False
        for signal_name, stocks in categories.items():
            if stocks:
                has_results = True
                emoji = screener_signal_emoji(signal_name)
                lines = []
                for ticker, ta_data in stocks[:8]:
                    rsi = ta_data.get("rsi", "?")
                    vol = ta_data.get("volume_ratio", 1.0)
                    rs = ta_data.get("relative_strength", 1.0)
                    lines.append(
                        f"**{ticker}** — RSI: {rsi} | Vol: {vol}x | RS: {rs}x"
                    )
                embed.add_field(
                    name=f"{emoji} {signal_name} ({len(stocks)})",
                    value="\n".join(lines),
                    inline=False,
                )

        if not has_results:
            embed.description = "No strong signals detected in current market conditions."

        embed.set_footer(text=f"Martin's Stocks • {sum(1 for r in results.values() if r and r[0])} signals found")
        apply_admin_flair(embed, interaction)
        await interaction.channel.send(embed=embed)


async def setup(bot):
    await bot.add_cog(MarketCog(bot))
