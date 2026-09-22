"""
/stock, /compare, /earnings, /insider commands.
/stock features interactive chart with 7 timeframe buttons that swap the image in-place.
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Optional

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
    fmt_price, fmt_pct, fmt_large, fmt_shares,
    signal_emoji, signal_color, rsi_label, score_bar,
    change_arrow, change_color_str, base_embed, error_embed, truncate,
)

logger = logging.getLogger(__name__)

TIMEFRAMES = ["1D", "5D", "1W", "1M", "1Y", "5Y", "MAX"]


async def ticker_autocomplete(
    interaction: discord.Interaction, current: str
) -> list[app_commands.Choice[str]]:
    """Autocomplete for stock ticker/name search."""
    if not current:
        return []
    results = await md.search_tickers(current)
    return [
        app_commands.Choice(
            name=f"{r['symbol']} — {r['name'][:45]}" if r.get("name") else r["symbol"],
            value=r["symbol"],
        )
        for r in results[:25]
        if r.get("symbol")
    ]


class ChartView(discord.ui.View):
    """
    7-button chart timeframe selector.
    Clicking a button regenerates the chart image and edits the message in-place.
    """

    def __init__(self, ticker: str, current_period: str = "1D", buy_zones: list = None):
        super().__init__(timeout=300)
        self.ticker = ticker
        self.current_period = current_period
        self.buy_zones = buy_zones or []
        self._add_buttons()

    def _add_buttons(self):
        self.clear_items()
        for tf in TIMEFRAMES:
            btn = discord.ui.Button(
                label=tf,
                style=discord.ButtonStyle.primary if tf == self.current_period else discord.ButtonStyle.secondary,
                custom_id=f"chart_{self.ticker}_{tf}",
            )
            btn.callback = self._make_callback(tf)
            self.add_item(btn)

    def _make_callback(self, period: str):
        async def callback(interaction: discord.Interaction):
            await interaction.response.defer()
            self.current_period = period
            self._add_buttons()  # Refresh to highlight active button

            chart_bytes = await charts.generate_chart(self.ticker, period, self.buy_zones)
            if not chart_bytes:
                await interaction.followup.send("Could not generate chart.", ephemeral=True)
                return

            file = discord.File(io.BytesIO(chart_bytes), filename=f"{self.ticker}_{period}.png")
            await interaction.message.edit(attachments=[file], view=self)
        return callback


class NewsButtonView(discord.ui.View):
    """Displays per-article AI Summarize buttons."""

    def __init__(self, articles: list[dict]):
        super().__init__(timeout=180)
        for i, article in enumerate(articles[:5]):
            btn = discord.ui.Button(
                label=f"Summarize #{i + 1}",
                style=discord.ButtonStyle.secondary,
                custom_id=f"summarize_{i}",
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
                title=f"📰 Summary: {truncate(title, 80)}",
                description=summary,
                color=discord.Color.from_str("#58a6ff"),
            )
            embed.add_field(name="Source", value=f"[Read full article]({url})", inline=False)
            await interaction.followup.send(embed=embed, ephemeral=True)
        return callback


class StockCog(commands.Cog, name="Stock"):
    def __init__(self, bot):
        self.bot = bot

    # ─── /stock ──────────────────────────────────────────────────────────────

    @app_commands.command(name="stock", description="Stock analysis — price, chart, news and AI deep dive")
    @app_commands.describe(query="Ticker symbol or company name (e.g. AAPL or Apple)")
    @app_commands.autocomplete(query=ticker_autocomplete)
    async def stock_cmd(self, interaction: discord.Interaction, query: str):
        await interaction.response.defer()

        ticker = await md.resolve_ticker(query)
        if not ticker:
            await interaction.followup.send(embed=error_embed(f"Could not find a ticker for **{query}**."))
            return

        # Fetch all data in parallel
        results = await asyncio.gather(
            md.get_stock_info(ticker),
            md.get_history(ticker, period="1y"),
            md.get_history("SPY", period="1y"),
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
        spy_hist     = safe(results[2])
        news_items   = safe(results[3]) or []
        analyst      = safe(results[4]) or {}
        earnings     = safe(results[5]) or {}
        insiders     = safe(results[6]) or []
        institutions = safe(results[7]) or []

        ta_data = {}
        if hist is not None and not hist.empty:
            ta_data = ta.run_full_analysis(hist, spy_hist)

        # ── Display values ────────────────────────────────────────────────────
        name = info.get("longName") or info.get("shortName") or ticker
        current_price = ta_data.get("current_price") or info.get("currentPrice") or info.get("regularMarketPrice") or 0
        prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose") or current_price
        change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close else 0
        arrow = "▲" if change_pct >= 0 else "▼"

        pre_price = info.get("preMarketPrice")
        pre_change_pct = info.get("preMarketChangePercent") or 0

        pe = info.get("trailingPE") or info.get("forwardPE")
        beta = info.get("beta")
        eps = info.get("trailingEps")
        div_yield = info.get("dividendYield")

        low_52  = ta_data.get("low_52w") or info.get("fiftyTwoWeekLow", 0)
        high_52 = ta_data.get("high_52w") or info.get("fiftyTwoWeekHigh", 0)
        range_pct = ta_data.get("range_pct", 0)
        bar_filled = round((range_pct or 0) / 10)
        range_bar = "▏" + "─" * bar_filled + "●" + "─" * (10 - bar_filled) + "▕"

        rsi_val = ta_data.get("rsi", 50)

        # ── AI analysis (market-context enriched) ────────────────────────────
        ai_result = await groq.analyze_stock(ticker, {
            "technical": ta_data,
            "smart_money": {"insiders": insiders, "institutions": institutions},
            "news": news_items,
            "info": info,
        }, db=self.bot.db)

        buy_zones = ai_result.get("buy_zones", [])
        signal = ai_result.get("overall_signal", "NEUTRAL")

        # Record prediction for accuracy tracking
        if current_price > 0:
            try:
                await self.bot.db.record_prediction(
                    ticker=ticker,
                    signal=signal,
                    score=ai_result.get("technical_score", 5),
                    price=current_price,
                    market_regime=ai_result.get("market_regime", ""),
                    source="stock_cmd",
                )
            except Exception:
                pass

        chart_bytes, scorecard_bytes = await asyncio.gather(
            charts.generate_chart(ticker, "1D", buy_zones),
            generate_stock_scorecard(ticker, ai_result, ta_data, info),
        )

        # ── Main embed ────────────────────────────────────────────────────────
        pre_line = ""
        if pre_price:
            pm_arrow = "▲" if pre_change_pct >= 0 else "▼"
            pre_line = f"\nPre-market: **{fmt_price(pre_price)}**  {pm_arrow} **{abs(pre_change_pct):.2f}%**"

        main_embed = discord.Embed(
            title=f"{signal_emoji(signal)} {ticker} — {name}",
            description=(
                f"**{fmt_price(current_price)}**  {arrow} **{abs(change_pct):.2f}%** today"
                f"{pre_line}\n"
                f"AI Signal: **{signal}**  •  Risk: **{ai_result.get('risk_level', 'Medium')}**"
            ),
            color=signal_color(signal),
        )
        main_embed.set_footer(text="Martin's Stocks • Data: Yahoo Finance + Groq LLaMA 3.3 70B")

        embed_pe  = round(pe, 1) if pe else "N/A"
        embed_eps = f"${round(eps, 2)}" if eps else "N/A"
        embed_beta = round(beta, 2) if beta else "N/A"
        embed_div  = fmt_pct(div_yield * 100) if div_yield else "None"
        main_embed.add_field(
            name="📈 Fundamentals",
            value=(
                f"Market Cap: **{fmt_large(info.get('marketCap'))}**\n"
                f"P/E (TTM):  **{embed_pe}**\n"
                f"EPS (TTM):  **{embed_eps}**\n"
                f"Beta:       **{embed_beta}**\n"
                f"Div. Yield: **{embed_div}**\n"
                f"Sector:     {info.get('sector', 'N/A')}"
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
                f"RS/SPY:  **{ta_data.get('relative_strength', 'N/A')}x**"
            ),
            inline=True,
        )

        main_embed.add_field(
            name="📏 52-Week Range",
            value=(
                f"Low:  **{fmt_price(low_52)}**\n"
                f"High: **{fmt_price(high_52)}**\n"
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
                f"Technical   {score_bar(ai_result.get('technical_score', 5))}\n"
                f"Macro Fit   {score_bar(ai_result.get('macro_fit_score', 5))}"
            ),
            inline=True,
        )

        thesis = ai_result.get("thesis", "")
        if thesis:
            conf = ai_result.get("confidence", "")
            regime = ai_result.get("market_regime", "")
            horizon = ai_result.get("time_horizon", "")
            header = "🤖 AI Thesis"
            if conf:
                header += f"  •  Conviction: **{conf}**"
            if regime:
                header += f"  •  Regime: `{regime}`"
            if horizon:
                header += f"  •  Horizon: `{horizon}`"
            main_embed.add_field(name=header, value=truncate(thesis, 450), inline=False)

        # Scenarios: bull / bear side-by-side
        if ai_result.get("bull_case"):
            main_embed.add_field(
                name="🐂 Bull Case",
                value=truncate(ai_result["bull_case"], 200),
                inline=True,
            )
        if ai_result.get("bear_case"):
            main_embed.add_field(
                name="🐻 Bear Case",
                value=truncate(ai_result["bear_case"], 200),
                inline=True,
            )

        if ai_result.get("catalyst"):
            main_embed.add_field(
                name="⚡ Next Catalyst",
                value=truncate(ai_result["catalyst"], 200),
                inline=True,
            )

        if ai_result.get("invalidation"):
            main_embed.add_field(
                name="🚫 Invalidation",
                value=truncate(ai_result["invalidation"], 200),
                inline=True,
            )

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
            main_embed.set_image(url=f"attachment://{ticker}_scorecard.png")
            files_main.append(discord.File(io.BytesIO(scorecard_bytes), filename=f"{ticker}_scorecard.png"))

        await interaction.followup.send(embed=main_embed, files=files_main)

        # ── Smart money embed ─────────────────────────────────────────────────
        sm_embed = discord.Embed(title=f"🏦 Smart Money — {ticker}", color=discord.Color.from_str("#58a6ff"))
        if institutions:
            sm_embed.add_field(
                name="Institutional Holders",
                value="\n".join(
                    f"• **{i['holder']}** — {i['pct_out']:.1f}% ({fmt_large(i['value'])})"
                    for i in institutions[:5]
                ),
                inline=False,
            )
        insider_lines = []
        for ins in (insiders or [])[:8]:
            is_buy = "buy" in ins.get("transaction", "").lower() or "purchase" in ins.get("transaction", "").lower()
            action = "🟢 BUY" if is_buy else "🔴 SELL"
            insider_lines.append(f"{action} **{ins['name']}** ({ins['title']}) — {fmt_shares(ins['shares'])} shares")
        sm_embed.add_field(
            name="Insider Transactions (SEC Filings)",
            value="\n".join(insider_lines) if insider_lines else "No recent filings",
            inline=False,
        )

        # ── News embed with AI summarize buttons ──────────────────────────────
        news_embed = discord.Embed(title=f"📰 Latest News — {ticker}", color=discord.Color.from_str("#21262d"))
        for i, a in enumerate(news_items[:5]):
            news_embed.add_field(
                name=f"{i+1}. {truncate(a['title'], 90)}",
                value=f"*{a['source']}* • {a['published']}\n[Read article]({a['url']})",
                inline=False,
            )

        chart_view = ChartView(ticker, current_period="1D", buy_zones=buy_zones)
        news_view = NewsButtonView(news_items) if news_items else discord.ui.View()

        await interaction.channel.send(embeds=[sm_embed, news_embed], view=news_view)

        if chart_bytes:
            chart_file = discord.File(io.BytesIO(chart_bytes), filename=f"{ticker}_1D.png")
            await interaction.channel.send(file=chart_file, view=chart_view)

    # ─── /compare ────────────────────────────────────────────────────────────

    @app_commands.command(name="compare", description="Side-by-side comparison of two stocks")
    @app_commands.describe(stock1="First stock ticker or name", stock2="Second stock ticker or name")
    async def compare_cmd(self, interaction: discord.Interaction, stock1: str, stock2: str):
        await interaction.response.defer()

        t1, t2 = await asyncio.gather(
            md.resolve_ticker(stock1), md.resolve_ticker(stock2)
        )
        if not t1:
            await interaction.followup.send(embed=error_embed(f"Could not resolve: **{stock1}**"))
            return
        if not t2:
            await interaction.followup.send(embed=error_embed(f"Could not resolve: **{stock2}**"))
            return

        async def _get_data(ticker):
            info = await md.get_stock_info(ticker)
            hist = await md.get_history(ticker, period="1y")
            spy_hist = await md.get_history("SPY", period="1y")
            ta_data = ta.run_full_analysis(hist, spy_hist) if hist is not None and not hist.empty else {}
            return info, ta_data

        (info1, ta1), (info2, ta2) = await asyncio.gather(_get_data(t1), _get_data(t2))

        def _price(info, ta_d):
            return ta_d.get("current_price") or info.get("currentPrice") or info.get("regularMarketPrice")

        def _pe(info):
            return info.get("trailingPE") or info.get("forwardPE")

        p1, p2 = _price(info1, ta1), _price(info2, ta2)
        low1, high1 = ta1.get("low_52w"), ta1.get("high_52w")
        low2, high2 = ta2.get("low_52w"), ta2.get("high_52w")

        embed = discord.Embed(
            title=f"⚔️ {t1} vs {t2} — Side-by-Side Comparison",
            color=discord.Color.from_str("#58a6ff"),
        )

        def _row(label, v1, v2):
            return f"**{label}**\n`{t1}:` {v1}\n`{t2}:` {v2}"

        range1 = f"${low1}—${high1}"
        range2 = f"${low2}—${high2}"
        rpos1 = f"{ta1.get('range_pct', 'N/A')}%"
        rpos2 = f"{ta2.get('range_pct', 'N/A')}%"
        embed.add_field(
            name="💰 Price & Performance",
            value=(
                f"{_row('Price', fmt_price(p1), fmt_price(p2))}\n\n"
                f"{_row('52W Range', range1, range2)}\n\n"
                f"{_row('52W Position', rpos1, rpos2)}"
            ),
            inline=True,
        )

        ema_trend1 = "Bullish" if ta1.get("above_ema20") else "Bearish"
        ema_trend2 = "Bullish" if ta2.get("above_ema20") else "Bearish"
        vol1 = f"{ta1.get('volume_ratio', 'N/A')}x"
        vol2 = f"{ta2.get('volume_ratio', 'N/A')}x"
        rs1 = f"{ta1.get('relative_strength', 'N/A')}x"
        rs2 = f"{ta2.get('relative_strength', 'N/A')}x"
        embed.add_field(
            name="📊 Technical",
            value=(
                f"{_row('RSI', ta1.get('rsi', 'N/A'), ta2.get('rsi', 'N/A'))}\n\n"
                f"{_row('EMA Trend', ema_trend1, ema_trend2)}\n\n"
                f"{_row('Volume', vol1, vol2)}\n\n"
                f"{_row('RS vs SPY', rs1, rs2)}"
            ),
            inline=True,
        )

        embed.add_field(
            name="🏢 Fundamentals",
            value=(
                f"{_row('Market Cap', fmt_large(info1.get('marketCap')), fmt_large(info2.get('marketCap')))}\n\n"
                f"{_row('P/E Ratio', round(_pe(info1), 1) if _pe(info1) else 'N/A', round(_pe(info2), 1) if _pe(info2) else 'N/A')}\n\n"
                f"{_row('Sector', info1.get('sector', 'N/A'), info2.get('sector', 'N/A'))}"
            ),
            inline=False,
        )

        embed.set_footer(text="Martin's Stocks • Data via Yahoo Finance")
        apply_admin_flair(embed, interaction)
        await interaction.followup.send(embed=embed)

    # ─── /earnings ───────────────────────────────────────────────────────────

    @app_commands.command(name="earnings", description="Earnings calendar and EPS history")
    @app_commands.describe(ticker="Stock ticker symbol (e.g. AAPL)")
    async def earnings_cmd(self, interaction: discord.Interaction, ticker: str):
        await interaction.response.defer()
        ticker = ticker.upper().strip()
        data = await md.get_earnings(ticker)

        embed = discord.Embed(
            title=f"📅 Earnings — {ticker}",
            color=discord.Color.from_str("#58a6ff"),
        )
        embed.add_field(
            name="Next Earnings Date",
            value=f"📆 **{data['next_earnings_date']}**",
            inline=False,
        )

        quarterly = data.get("quarterly", [])
        if quarterly:
            rows = []
            for q in quarterly:
                beat = ""
                if q.get("surprise_pct") is not None:
                    sp = q["surprise_pct"]
                    beat = f"{'🟢 Beat' if sp >= 0 else '🔴 Miss'} {fmt_pct(sp)}"
                est = f"Est: ${q['estimate']}" if q.get("estimate") is not None else "Est: N/A"
                rows.append(
                    f"**{q['date']}** | Act: ${q['actual']} | {est} | {beat}"
                )
            embed.add_field(
                name="Last 4 Quarters EPS",
                value="\n".join(rows),
                inline=False,
            )
        else:
            embed.add_field(name="EPS History", value="No data available.", inline=False)

        embed.set_footer(text="Martin's Stocks • Data via Yahoo Finance")
        apply_admin_flair(embed, interaction)
        await interaction.followup.send(embed=embed)

    # ─── /insider ────────────────────────────────────────────────────────────

    @app_commands.command(name="insider", description="Recent SEC-filed insider transactions")
    @app_commands.describe(ticker="Stock ticker symbol (e.g. AAPL)")
    async def insider_cmd(self, interaction: discord.Interaction, ticker: str):
        await interaction.response.defer()
        ticker = ticker.upper().strip()
        transactions = await md.get_insider_transactions(ticker)

        embed = discord.Embed(
            title=f"🔍 Insider Transactions — {ticker}",
            color=discord.Color.from_str("#58a6ff"),
        )

        if not transactions:
            embed.description = "No recent insider transactions found."
        else:
            buys = [t for t in transactions if "buy" in t.get("transaction", "").lower() or "purchase" in t.get("transaction", "").lower()]
            sells = [t for t in transactions if "sell" in t.get("transaction", "").lower()]

            net_shares = sum(
                t.get("shares", 0) * (1 if "buy" in t.get("transaction", "").lower() or "purchase" in t.get("transaction", "").lower() else -1)
                for t in transactions[:10]
            )

            embed.description = (
                f"**{len(buys)}** recent buys | **{len(sells)}** recent sells\n"
                f"Net position change: **{'+' if net_shares >= 0 else ''}{fmt_shares(int(net_shares))} shares**"
            )

            for t in transactions[:10]:
                is_buy = "buy" in t.get("transaction", "").lower() or "purchase" in t.get("transaction", "").lower()
                action_emoji = "🟢" if is_buy else "🔴"
                embed.add_field(
                    name=f"{action_emoji} {t['name']} — {t['transaction']}",
                    value=(
                        f"**Role:** {t['title'] or 'N/A'}\n"
                        f"**Shares:** {fmt_shares(t['shares'])}\n"
                        f"**Value:** {fmt_large(t['value'])}\n"
                        f"**Date:** {t['date']}\n"
                        f"**Type:** {t['ownership']}"
                    ),
                    inline=True,
                )

        embed.set_footer(text="Martin's Stocks • SEC filings via Yahoo Finance")
        apply_admin_flair(embed, interaction)
        await interaction.followup.send(embed=embed)


async def setup(bot):
    await bot.add_cog(StockCog(bot))
