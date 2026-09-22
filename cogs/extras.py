"""
Extra commands: /analyst, /dividend, /top
"""

from __future__ import annotations

import asyncio
import datetime
import logging

import discord
from discord import app_commands
from discord.ext import commands

from services import market_data as md
from services import groq_service as groq
from services import news_service as news
from utils import technical as ta
from utils.permissions import apply_admin_flair
from utils.helpers import (
    fmt_price, fmt_pct, fmt_large, signal_emoji, signal_color,
    score_bar, base_embed, error_embed, truncate,
)

logger = logging.getLogger(__name__)

# Stocks scanned by /top
TOP_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO",
    "JPM", "V", "UNH", "XOM", "LLY", "JNJ", "WMT", "MA", "PG", "HD",
    "MRK", "ORCL", "BAC", "ABBV", "CVX", "KO", "PEP", "COST", "ADBE",
    "CRM", "AMD", "NFLX",
]


class ExtrasCog(commands.Cog, name="Extras"):
    def __init__(self, bot):
        self.bot = bot

    # ─── /analyst ────────────────────────────────────────────────────────────

    @app_commands.command(name="analyst", description="Wall Street analyst ratings and price targets")
    @app_commands.describe(ticker="Stock ticker symbol (e.g. AAPL)")
    async def analyst_cmd(self, interaction: discord.Interaction, ticker: str):
        await interaction.response.defer()
        ticker = ticker.upper().strip()

        data, info = await asyncio.gather(
            md.get_analyst_data(ticker),
            md.get_stock_info(ticker),
        )

        current = data.get("current_price") or info.get("currentPrice") or 0
        target_mean = data.get("target_mean")
        upside = ((target_mean - current) / current * 100) if (target_mean and current) else None

        sb = data.get("strong_buy", 0)
        b  = data.get("buy", 0)
        h  = data.get("hold", 0)
        s  = data.get("sell", 0)
        ss = data.get("strong_sell", 0)
        total = sb + b + h + s + ss

        rec = data.get("recommendation", "N/A").replace("_", " ").title()
        upside_str = f"{'▲' if upside and upside >= 0 else '▼'} {abs(upside):.1f}%" if upside is not None else "N/A"
        upside_col = discord.Color.green() if (upside and upside >= 0) else discord.Color.red()

        embed = discord.Embed(
            title=f"🎯 Analyst Ratings — {ticker}",
            description=f"**Consensus:** {rec}  •  **Analysts:** {data.get('num_analysts') or total or 'N/A'}",
            color=upside_col,
        )

        embed.add_field(
            name="💰 Price Targets",
            value=(
                f"Current:  **{fmt_price(current)}**\n"
                f"Mean:     **{fmt_price(target_mean)}** ({upside_str})\n"
                f"High:     **{fmt_price(data.get('target_high'))}**\n"
                f"Low:      **{fmt_price(data.get('target_low'))}**"
            ),
            inline=True,
        )

        if total > 0:
            def _bar(count):
                filled = round(count / total * 10)
                return "█" * filled + "░" * (10 - filled) + f"  {count}"

            embed.add_field(
                name="📊 Rating Breakdown",
                value=(
                    f"🟢 Strong Buy  {_bar(sb)}\n"
                    f"🟩 Buy         {_bar(b)}\n"
                    f"⬜ Hold        {_bar(h)}\n"
                    f"🟥 Sell        {_bar(s)}\n"
                    f"🔴 Strong Sell {_bar(ss)}"
                ),
                inline=True,
            )
        else:
            embed.add_field(name="Rating Breakdown", value="No detailed rating data available.", inline=True)

        embed.set_footer(text="Martin's Stocks • Data via Yahoo Finance")
        apply_admin_flair(embed, interaction)
        await interaction.followup.send(embed=embed)

    # ─── /dividend ───────────────────────────────────────────────────────────

    @app_commands.command(name="dividend", description="Dividend yield, ex-date, and payout history")
    @app_commands.describe(ticker="Stock ticker symbol (e.g. AAPL)")
    async def dividend_cmd(self, interaction: discord.Interaction, ticker: str):
        await interaction.response.defer()
        ticker = ticker.upper().strip()

        data = await md.get_dividend_data(ticker)

        yield_pct = data.get("dividend_yield")
        rate = data.get("dividend_rate")
        ex_date = data.get("ex_dividend_date") or "N/A"
        payout = data.get("payout_ratio")
        five_yr = data.get("five_year_avg_yield")

        if not yield_pct and not rate:
            await interaction.followup.send(embed=discord.Embed(
                description=f"**{ticker}** does not pay a dividend.",
                color=discord.Color.from_str("#8b949e"),
            ))
            return

        embed = discord.Embed(
            title=f"💸 Dividend Info — {ticker}",
            color=discord.Color.from_str("#2ea043"),
        )

        embed.add_field(
            name="Current Dividend",
            value=(
                f"Yield:        **{fmt_pct(yield_pct * 100) if yield_pct else 'N/A'}**\n"
                f"Annual Rate:  **${rate:.4f}** / share\n"
                f"Ex-Date:      **{ex_date}**\n"
                f"Payout Ratio: **{fmt_pct(payout * 100) if payout else 'N/A'}**\n"
                f"5Y Avg Yield: **{fmt_pct(five_yr) if five_yr else 'N/A'}**"
            ),
            inline=False,
        )

        history = data.get("history", [])
        if history:
            lines = [f"`{h['date']}` — **${h['amount']:.4f}**" for h in reversed(history[-8:])]
            embed.add_field(
                name="📅 Recent Payment History",
                value="\n".join(lines),
                inline=False,
            )

        embed.set_footer(text="Martin's Stocks • Data via Yahoo Finance")
        apply_admin_flair(embed, interaction)
        await interaction.followup.send(embed=embed)

    # ─── /top ─────────────────────────────────────────────────────────────────

    @app_commands.command(name="top", description="AI-ranked top stock picks from the market universe")
    async def top_cmd(self, interaction: discord.Interaction):
        await interaction.response.defer()

        await interaction.followup.send(embed=discord.Embed(
            description=f"⏳ Scanning {len(TOP_UNIVERSE)} stocks and running AI scoring… this takes ~30 seconds.",
            color=discord.Color.from_str("#161b22"),
        ))

        spy_hist = await md.get_history("SPY", period="1y")

        async def _score_ticker(ticker: str) -> dict | None:
            try:
                info, hist, news_items = await asyncio.gather(
                    md.get_stock_info(ticker),
                    md.get_history(ticker, period="1y"),
                    news.get_stock_news(ticker),
                    return_exceptions=True,
                )
                if isinstance(info, Exception): info = {}
                if isinstance(hist, Exception) or hist is None or hist.empty: return None
                if isinstance(news_items, Exception): news_items = []

                ta_data = ta.run_full_analysis(hist, spy_hist)
                ai = await groq.analyze_for_daily_report(ticker, {
                    "technical": ta_data, "info": info, "news": news_items,
                })
                overall = (
                    ta_data.get("rsi", 50) / 10 +
                    ta_data.get("volume_ratio", 1) +
                    ai.get("score", 5)
                )
                return {
                    "ticker": ticker,
                    "signal": ai.get("signal", "NEUTRAL"),
                    "score": ai.get("score", 5),
                    "thesis": ai.get("thesis", ""),
                    "buy_zone": ai.get("buy_zone", "N/A"),
                    "overall": overall,
                    "price": ta_data.get("current_price") or info.get("currentPrice"),
                    "rsi": ta_data.get("rsi"),
                    "name": info.get("shortName") or ticker,
                }
            except Exception as e:
                logger.warning(f"/top: {ticker} skipped: {e}")
                return None

        results = []
        for i in range(0, len(TOP_UNIVERSE), 8):
            batch = TOP_UNIVERSE[i:i + 8]
            batch_results = await asyncio.gather(*[_score_ticker(t) for t in batch])
            results.extend(r for r in batch_results if r is not None)
            await asyncio.sleep(1)

        if not results:
            await interaction.channel.send(embed=error_embed("Could not retrieve data. Try again later."))
            return

        bullish = [r for r in results if "buy" in r["signal"].lower()]
        bullish.sort(key=lambda x: x["score"], reverse=True)
        top5 = bullish[:5] if bullish else sorted(results, key=lambda x: x["score"], reverse=True)[:5]

        embed = discord.Embed(
            title="🏆 Top AI Stock Picks",
            description=f"Best-scoring stocks from a {len(results)}-stock universe, ranked by Groq AI conviction.",
            color=discord.Color.from_str("#3fb950"),
        )

        medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
        for i, stock in enumerate(top5):
            embed.add_field(
                name=f"{medals[i]} {stock['ticker']} — {truncate(stock['name'], 28)}",
                value=(
                    f"Signal: **{stock['signal']}**  •  Score: **{stock['score']}/10**\n"
                    f"Price: **{fmt_price(stock['price'])}**  •  RSI: **{stock['rsi']}**\n"
                    f"Buy Zone: **{stock['buy_zone']}**\n"
                    f"_{truncate(stock['thesis'], 110)}_"
                ),
                inline=False,
            )

        embed.set_footer(text=f"Martin's Stocks • {len(results)} stocks scanned • AI by Groq LLaMA 3.3 70B")
        apply_admin_flair(embed, interaction)
        await interaction.channel.send(embed=embed)


    # ─── /recommendations ─────────────────────────────────────────────────────

    @app_commands.command(
        name="recommendations",
        description="Top 20 AI stock recommendations across 7 industries — scored on news, technicals, premarket & more",
    )
    async def recommendations_cmd(self, interaction: discord.Interaction):
        await interaction.response.defer()

        # 7 sectors × 7 stocks = 49-stock universe for sector diversity
        SECTOR_UNIVERSE = {
            "⚡ Technology":    ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMD", "AVGO"],
            "💊 Healthcare":    ["UNH", "LLY", "JNJ", "MRK", "ABBV", "TMO", "ABT"],
            "🏦 Finance":       ["JPM", "BAC", "V", "MA", "GS", "MS", "BRK-B"],
            "⛽ Energy":        ["XOM", "CVX", "COP", "SLB", "EOG", "VLO", "PSX"],
            "🛒 Consumer":      ["TSLA", "AMZN", "COST", "WMT", "HD", "MCD", "NKE"],
            "🏭 Industrials":   ["CAT", "HON", "GE", "RTX", "UPS", "DE", "LMT"],
            "📡 Communication": ["NFLX", "DIS", "T", "VZ", "CMCSA", "CHTR", "SNAP"],
        }
        all_tickers = [t for tickers in SECTOR_UNIVERSE.values() for t in tickers]
        ticker_to_sector = {t: s for s, tickers in SECTOR_UNIVERSE.items() for t in tickers}

        await interaction.followup.send(embed=discord.Embed(
            description=(
                f"⏳ Scanning **{len(all_tickers)} stocks** across **7 industries** with live AI scoring…\n"
                f"Analysing: news sentiment, premarket momentum, RSI, relative strength & more. ~60s"
            ),
            color=discord.Color.from_str("#161b22"),
        ))

        spy_hist = await md.get_history("SPY", period="1y")

        def _composite(ai_score: float, signal: str, pre_chg: float | None,
                        rsi: float | None, rs: float | None, vol_ratio: float | None) -> float:
            """Composite score combining AI conviction, premarket momentum, technicals."""
            score = float(ai_score)
            if pre_chg is not None:
                score += min(pre_chg * 0.3, 0.8) if pre_chg > 0 else max(pre_chg * 0.2, -0.5)
            if rsi:
                if 40 <= rsi <= 65:
                    score += 0.5
                elif rsi > 80:
                    score -= 0.5
                elif rsi < 30:
                    score += 0.3
            if rs and rs > 1.0:
                score += min((rs - 1.0) * 0.5, 0.5)
            if vol_ratio and vol_ratio > 1.5:
                score += 0.3
            sig = signal.upper()
            if "STRONG BUY" in sig:
                score *= 1.15
            elif "BUY" in sig:
                score *= 1.05
            elif "STRONG SELL" in sig:
                score *= 0.55
            elif "SELL" in sig:
                score *= 0.70
            return round(min(score, 10.0), 2)

        async def _score(ticker: str) -> dict | None:
            try:
                info, hist, news_items = await asyncio.gather(
                    md.get_stock_info(ticker),
                    md.get_history(ticker, period="1y"),
                    news.get_stock_news(ticker),
                    return_exceptions=True,
                )
                if isinstance(info, Exception): info = {}
                if isinstance(hist, Exception) or hist is None or hist.empty: return None
                if isinstance(news_items, Exception): news_items = []

                ta_data = ta.run_full_analysis(hist, spy_hist)
                ai = await groq.analyze_for_daily_report(ticker, {
                    "technical": ta_data, "info": info, "news": news_items,
                })

                current = ta_data.get("current_price") or info.get("currentPrice") or 0
                prev = info.get("previousClose") or current
                day_chg = ((current - prev) / prev * 100) if prev else 0
                pre_price = info.get("preMarketPrice")
                pre_chg = info.get("preMarketChangePercent") or 0
                rsi = ta_data.get("rsi")
                rs = ta_data.get("relative_strength", 1.0)
                vol = ta_data.get("volume_ratio")
                ai_score = ai.get("score", 5)
                signal = ai.get("signal", "NEUTRAL")

                return {
                    "ticker": ticker,
                    "sector": ticker_to_sector.get(ticker, "Other"),
                    "name": info.get("shortName") or ticker,
                    "signal": signal,
                    "ai_score": ai_score,
                    "composite": _composite(ai_score, signal, pre_chg, rsi, rs, vol),
                    "thesis": ai.get("thesis", ""),
                    "buy_zone": ai.get("buy_zone", "N/A"),
                    "price": current,
                    "day_chg": day_chg,
                    "pre_price": pre_price,
                    "pre_chg": pre_chg,
                    "rsi": rsi,
                    "rs": rs,
                    "vol": vol,
                }
            except Exception as e:
                logger.warning(f"/recommendations: {ticker} skipped: {e}")
                return None

        # Process in batches of 7 (one sector at a time) to respect Groq rate limits
        all_results = []
        for i in range(0, len(all_tickers), 7):
            batch = all_tickers[i:i + 7]
            batch_out = await asyncio.gather(*[_score(t) for t in batch])
            all_results.extend(r for r in batch_out if r is not None)
            await asyncio.sleep(1)

        if not all_results:
            await interaction.channel.send(embed=error_embed("Could not retrieve data. Try again later."))
            return

        # Enforce diversity: pick up to 4 per sector, then take global top 20
        sector_counts: dict[str, int] = {}
        diverse: list[dict] = []
        for r in sorted(all_results, key=lambda x: x["composite"], reverse=True):
            sec = r["sector"]
            if sector_counts.get(sec, 0) < 4:
                diverse.append(r)
                sector_counts[sec] = sector_counts.get(sec, 0) + 1
            if len(diverse) == 20:
                break

        today = datetime.datetime.now().strftime("%b %d, %Y")
        medals = ["🥇", "🥈", "🥉"] + [f"{n}️⃣" for n in range(4, 11)]

        # ── Embed 1: Overview — top 20 grouped by sector ─────────────────────
        overview = discord.Embed(
            title=f"⭐ Daily Recommendations — {today}",
            description=(
                f"**Top 20 picks** from **{len(all_results)}** stocks across **7 industries**, "
                f"scored by Groq AI on news, premarket momentum, RSI, relative strength & volume. "
                f"Max 4 per sector for diversification."
            ),
            color=discord.Color.from_str("#3fb950"),
        )

        # Group the top 20 back by sector for the field layout
        sector_picks: dict[str, list[dict]] = {}
        for r in diverse:
            sector_picks.setdefault(r["sector"], []).append(r)

        rank = 1
        for sector_label, picks in sorted(
            sector_picks.items(), key=lambda kv: max(p["composite"] for p in kv[1]), reverse=True
        ):
            lines = []
            for p in sorted(picks, key=lambda x: x["composite"], reverse=True):
                medal = medals[rank - 1] if rank <= len(medals) else f"**#{rank}**"
                day_dot = "🟢" if p["day_chg"] >= 0 else "🔴"
                day_arrow = "▲" if p["day_chg"] >= 0 else "▼"
                line = (
                    f"{medal} **{p['ticker']}** {day_dot}{day_arrow}{abs(p['day_chg']):.1f}%"
                    f"  •  Score **{p['composite']}/10**  •  RSI {p['rsi']}"
                )
                if p.get("pre_price"):
                    pm_dot = "🟢" if p["pre_chg"] >= 0 else "🔴"
                    pm_arrow = "▲" if p["pre_chg"] >= 0 else "▼"
                    line += f"  •  Pre {pm_dot}{pm_arrow}{abs(p['pre_chg']):.1f}%"
                lines.append(line)
                rank += 1
            overview.add_field(name=sector_label, value="\n".join(lines), inline=False)

        overview.set_footer(
            text=f"Martin's Stocks • {len(all_results)} stocks scanned • Composite score = AI + premarket + RSI + RS + volume"
        )
        apply_admin_flair(overview, interaction)
        await interaction.channel.send(embed=overview)

        # ── Embed 2: Deep analysis on top 5 ──────────────────────────────────
        top5 = diverse[:5]
        deep = discord.Embed(
            title="🔬 Deep Analysis — Top 5 Picks",
            description="Full AI thesis, entry zones, and technical profile for the highest-conviction picks.",
            color=discord.Color.from_str("#58a6ff"),
        )
        for i, s in enumerate(top5):
            day_arrow = "▲" if s["day_chg"] >= 0 else "▼"
            price_line = f"**{fmt_price(s['price'])}** {day_arrow}{abs(s['day_chg']):.2f}% today"
            if s.get("pre_price"):
                pm_arrow = "▲" if s["pre_chg"] >= 0 else "▼"
                price_line += f"  |  Pre-mkt **{fmt_price(s['pre_price'])}** {pm_arrow}{abs(s['pre_chg']):.2f}%"
            deep.add_field(
                name=f"{medals[i]} {s['ticker']} — {truncate(s['name'], 30)}  [{s['sector']}]",
                value=(
                    f"{price_line}\n"
                    f"Signal: **{s['signal']}**  •  Composite: **{s['composite']}/10**  •  AI: **{s['ai_score']}/10**\n"
                    f"RSI: **{s['rsi']}**  •  RS vs SPY: **{s['rs']}x**  •  Volume: **{s['vol']}x**\n"
                    f"Buy Zone: **{s['buy_zone']}**\n"
                    f"_{truncate(s['thesis'], 180)}_"
                ),
                inline=False,
            )
        deep.set_footer(text="Martin's Stocks • AI analysis by Groq LLaMA 3.3 70B")
        apply_admin_flair(deep, interaction)
        await interaction.channel.send(embed=deep)


async def setup(bot):
    await bot.add_cog(ExtrasCog(bot))
