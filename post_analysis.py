"""
One-shot script: run full NVDA analysis and POST the results to Discord.
Run from the project root: python3 post_analysis.py
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import sys
from pathlib import Path

# Must happen before any project imports so env vars are available
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import aiohttp

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from services import market_data as md
from services import groq_service as groq
from services import news_service as news
from utils import technical as ta
from utils import charts
from utils.charts import generate_stock_scorecard
from utils.helpers import (
    fmt_price, fmt_pct, fmt_large, fmt_shares,
    signal_emoji, signal_color, rsi_label, score_bar,
    change_arrow, truncate,
)
from db.database import Database

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s: %(message)s")
logger = logging.getLogger("post_analysis")

TICKER = "NVDA"
CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID", "")


def _color_int(signal: str) -> int:
    """Map signal name to Discord color integer."""
    colors = {
        "STRONG BUY": 0x00FF7F,
        "BUY":        0x57F287,
        "NEUTRAL":    0xFEE75C,
        "SELL":       0xFF4444,
        "STRONG SELL":0xFF0000,
    }
    return colors.get(signal.upper(), 0x5865F2)


def build_main_embed(ticker: str, info: dict, ta_data: dict, ai_result: dict) -> dict:
    """Build the main analysis embed as a Discord API dict."""
    name = info.get("longName") or info.get("shortName") or ticker
    current_price = ta_data.get("current_price") or info.get("currentPrice") or info.get("regularMarketPrice") or 0
    prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose") or current_price
    change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close else 0
    arrow = "▲" if change_pct >= 0 else "▼"

    pre_price = info.get("preMarketPrice")
    pre_change_pct = info.get("preMarketChangePercent") or 0

    pre_line = ""
    if pre_price:
        pm_arrow = "▲" if pre_change_pct >= 0 else "▼"
        pre_line = f"\nPre-market: **{fmt_price(pre_price)}**  {pm_arrow} **{abs(pre_change_pct):.2f}%**"

    signal = ai_result.get("overall_signal", "NEUTRAL")

    pe = info.get("trailingPE") or info.get("forwardPE")
    beta = info.get("beta")
    eps = info.get("trailingEps")
    div_yield = info.get("dividendYield")
    rsi_val = ta_data.get("rsi", 50)
    low_52  = ta_data.get("low_52w") or info.get("fiftyTwoWeekLow", 0)
    high_52 = ta_data.get("high_52w") or info.get("fiftyTwoWeekHigh", 0)
    range_pct = ta_data.get("range_pct", 0)
    bar_filled = round((range_pct or 0) / 10)
    range_bar = "▏" + "─" * bar_filled + "●" + "─" * (10 - bar_filled) + "▕"

    buy_zones = ai_result.get("buy_zones", [])
    zones_str = "  ·  ".join(buy_zones) if buy_zones else "N/A"

    conf   = ai_result.get("confidence", "")
    regime = ai_result.get("market_regime", "")
    thesis_header = "🤖 AI Thesis"
    if conf:
        thesis_header += f"  •  Confidence: **{conf}**"
    if regime:
        thesis_header += f"  •  Regime: `{regime}`"

    fields = [
        {
            "name": "📈 Fundamentals",
            "value": (
                f"Market Cap: **{fmt_large(info.get('marketCap'))}**\n"
                f"P/E (TTM):  **{round(pe, 1) if pe else 'N/A'}**\n"
                f"EPS (TTM):  **{'$' + str(round(eps, 2)) if eps else 'N/A'}**\n"
                f"Beta:       **{round(beta, 2) if beta else 'N/A'}**\n"
                f"Div. Yield: **{fmt_pct(div_yield * 100) if div_yield else 'None'}**\n"
                f"Sector:     {info.get('sector', 'N/A')}"
            ),
            "inline": True,
        },
        {
            "name": "📊 Technicals",
            "value": (
                f"RSI:     **{rsi_label(rsi_val)}** ({rsi_val:.0f})\n"
                f"EMA 20:  `{ta_data.get('ema20_label', 'N/A')}`\n"
                f"EMA 50:  `{ta_data.get('ema50_label', 'N/A')}`\n"
                f"EMA 200: `{ta_data.get('ema200_label', 'N/A')}`\n"
                f"Volume:  **{ta_data.get('volume_ratio', 'N/A')}x** avg\n"
                f"Pattern: `{ta_data.get('pattern', 'None')}`\n"
                f"RS/SPY:  **{ta_data.get('relative_strength', 'N/A')}x**"
            ),
            "inline": True,
        },
        {
            "name": "📏 52-Week Range",
            "value": (
                f"Low:  **{fmt_price(low_52)}**\n"
                f"High: **{fmt_price(high_52)}**\n"
                f"Position: **{range_pct}%**\n"
                f"`{range_bar}`"
            ),
            "inline": True,
        },
        {
            "name": "🏆 Signal Layers",
            "value": (
                f"Momentum    {score_bar(ai_result.get('momentum_score', 5))}\n"
                f"Smart Money {score_bar(ai_result.get('smart_money_score', 5))}\n"
                f"News        {score_bar(ai_result.get('news_score', 5))}\n"
                f"Technical   {score_bar(ai_result.get('technical_score', 5))}\n"
                f"Macro Fit   {score_bar(ai_result.get('macro_fit_score', 5))}"
            ),
            "inline": True,
        },
    ]

    thesis = ai_result.get("thesis", "")
    if thesis:
        fields.append({"name": thesis_header, "value": truncate(thesis, 1024), "inline": False})

    if ai_result.get("catalyst"):
        fields.append({"name": "⚡ Next Catalyst", "value": truncate(ai_result["catalyst"], 200), "inline": True})

    if ai_result.get("bear_case"):
        fields.append({"name": "🐻 Bear Case", "value": truncate(ai_result["bear_case"], 200), "inline": True})

    fields.append({
        "name": "🎯 Entry Strategy",
        "value": f"**Buy Zones:** {zones_str}\n{truncate(ai_result.get('entry_points', ''), 200)}",
        "inline": False,
    })

    return {
        "title": f"{signal_emoji(signal)} {ticker} — {name}",
        "description": (
            f"**{fmt_price(current_price)}**  {arrow} **{abs(change_pct):.2f}%** today"
            f"{pre_line}\n"
            f"AI Signal: **{signal}**  •  Risk: **{ai_result.get('risk_level', 'Medium')}**"
        ),
        "color": _color_int(signal),
        "fields": fields,
        "footer": {"text": "Martin's Stocks • Data: Yahoo Finance + Groq LLaMA 3.3 70B"},
        "image": {"url": f"attachment://{ticker}_scorecard.png"},
    }


def build_smart_money_embed(ticker: str, insiders: list, institutions: list) -> dict:
    """Build the smart money embed dict."""
    fields = []

    if institutions:
        fields.append({
            "name": "Institutional Holders",
            "value": "\n".join(
                f"• **{i['holder']}** — {i['pct_out']:.1f}% ({fmt_large(i['value'])})"
                for i in institutions[:5]
            ),
            "inline": False,
        })

    insider_lines = []
    for ins in (insiders or [])[:8]:
        tx = ins.get("transaction", "").lower()
        is_buy = "buy" in tx or "purchase" in tx
        action = "🟢 BUY" if is_buy else "🔴 SELL"
        insider_lines.append(
            f"{action} **{ins['name']}** ({ins['title']}) — {fmt_shares(ins['shares'])} shares"
        )
    fields.append({
        "name": "Insider Transactions (SEC Filings)",
        "value": "\n".join(insider_lines) if insider_lines else "No recent filings",
        "inline": False,
    })

    return {
        "title": f"🏦 Smart Money — {ticker}",
        "color": 0x58A6FF,
        "fields": fields,
    }


def build_news_embed(ticker: str, news_items: list) -> dict:
    """Build the news embed dict."""
    fields = []
    for i, a in enumerate(news_items[:5]):
        fields.append({
            "name": f"{i+1}. {truncate(a['title'], 90)}",
            "value": f"*{a['source']}* • {a['published']}\n[Read article]({a['url']})",
            "inline": False,
        })
    return {
        "title": f"📰 Latest News — {ticker}",
        "color": 0x21262D,
        "fields": fields,
    }


async def post_to_discord(
    session: aiohttp.ClientSession,
    token: str,
    channel_id: str,
    embeds: list[dict],
    file_bytes: bytes | None = None,
    filename: str = "scorecard.png",
) -> dict:
    """POST a message (with optional file) to a Discord channel via REST API."""
    url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
    headers = {"Authorization": f"Bot {token}"}

    if file_bytes:
        form = aiohttp.FormData()
        form.add_field(
            "payload_json",
            json.dumps({"embeds": embeds}),
            content_type="application/json",
        )
        form.add_field(
            "files[0]",
            file_bytes,
            filename=filename,
            content_type="image/png",
        )
        async with session.post(url, data=form, headers=headers) as resp:
            body = await resp.json()
            if resp.status not in (200, 201):
                logger.error(f"Discord API error {resp.status}: {body}")
            return body
    else:
        headers["Content-Type"] = "application/json"
        async with session.post(url, json={"embeds": embeds}, headers=headers) as resp:
            body = await resp.json()
            if resp.status not in (200, 201):
                logger.error(f"Discord API error {resp.status}: {body}")
            return body


async def main() -> None:
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        sys.exit("DISCORD_TOKEN not set in .env")

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        sys.exit("GROQ_API_KEY not set in .env")

    logger.info(f"Fetching data for {TICKER}...")

    # Open DB (needed for market context accuracy injection)
    db = Database()
    await db.initialize()

    results = await asyncio.gather(
        md.get_stock_info(TICKER),
        md.get_history(TICKER, period="1y"),
        md.get_history("SPY", period="1y"),
        news.get_stock_news(TICKER),
        md.get_insider_transactions(TICKER),
        md.get_institutional_holders(TICKER),
        return_exceptions=True,
    )

    def safe(v):
        return v if not isinstance(v, Exception) else None

    info         = safe(results[0]) or {}
    hist         = safe(results[1])
    spy_hist     = safe(results[2])
    news_items   = safe(results[3]) or []
    insiders     = safe(results[4]) or []
    institutions = safe(results[5]) or []

    ta_data = {}
    if hist is not None and not hist.empty:
        ta_data = ta.run_full_analysis(hist, spy_hist)

    logger.info("Running AI analysis...")
    ai_result = await groq.analyze_stock(TICKER, {
        "technical": ta_data,
        "smart_money": {"insiders": insiders, "institutions": institutions},
        "news": news_items,
        "info": info,
    }, db=db)

    logger.info(f"Signal: {ai_result.get('overall_signal')}  Confidence: {ai_result.get('confidence')}")

    logger.info("Generating charts...")
    buy_zones = ai_result.get("buy_zones", [])
    chart_bytes, scorecard_bytes = await asyncio.gather(
        charts.generate_chart(TICKER, "1D", buy_zones),
        generate_stock_scorecard(TICKER, ai_result, ta_data, info),
    )

    main_embed      = build_main_embed(TICKER, info, ta_data, ai_result)
    sm_embed        = build_smart_money_embed(TICKER, insiders, institutions)
    news_embed      = build_news_embed(TICKER, news_items)

    logger.info("Posting to Discord...")
    async with aiohttp.ClientSession() as session:
        # Message 1: main embed + scorecard image
        r1 = await post_to_discord(
            session, token, CHANNEL_ID,
            embeds=[main_embed],
            file_bytes=scorecard_bytes,
            filename=f"{TICKER}_scorecard.png",
        )
        logger.info(f"Message 1 id={r1.get('id')}")

        # Message 2: smart money + news embeds
        r2 = await post_to_discord(
            session, token, CHANNEL_ID,
            embeds=[sm_embed, news_embed],
        )
        logger.info(f"Message 2 id={r2.get('id')}")

        # Message 3: chart image (plain file, no embed)
        if chart_bytes:
            form = aiohttp.FormData()
            form.add_field(
                "payload_json",
                json.dumps({"content": f"📊 **{TICKER}** — 1D chart"}),
                content_type="application/json",
            )
            form.add_field(
                "files[0]",
                chart_bytes,
                filename=f"{TICKER}_1D.png",
                content_type="image/png",
            )
            url = f"https://discord.com/api/v10/channels/{CHANNEL_ID}/messages"
            headers = {"Authorization": f"Bot {token}"}
            async with session.post(url, data=form, headers=headers) as resp:
                r3 = await resp.json()
                logger.info(f"Message 3 id={r3.get('id')}")

    logger.info("Done — all messages posted.")


if __name__ == "__main__":
    asyncio.run(main())
