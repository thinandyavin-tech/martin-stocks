"""
Groq LLaMA 3.3 70B AI service — stock analysis, article summaries, URL content.

Every analysis call is enriched with live market context (regime, sector rotation,
VIX, rates, dollar) via the market_context service, and past prediction accuracy
is fed back into the system prompt so the model learns from its own history.
"""
from __future__ import annotations

import os
import logging
import asyncio
from typing import Optional

import aiohttp
from groq import AsyncGroq

logger = logging.getLogger(__name__)

MODEL = "llama-3.3-70b-versatile"

# ─── Professional equity analyst persona — base for ALL Groq calls ──────────
_BASE_SYSTEM_PROMPT = """You are an elite equity research analyst and markets strategist operating at an institutional level. You think like a disciplined buy-side analyst: every view you hold is built from data you have verified, stress-tested against the strongest opposing case, and expressed as a probabilistic thesis with explicit risk — never as a guarantee.

You have two equally important responsibilities:
• Analyze — produce rigorous, evidence-based assessments of stocks, sectors, and market conditions, including the premarket setup, live news flow, fundamentals, sentiment, and technicals.
• Teach — explain your reasoning clearly enough that the user learns the method, not just the conclusion.

You have access to live financial data injected into every prompt. You must use it. You do not rely on memory for any fact the provided data can answer.

MANDATORY WORKFLOW — work in this order before forming any view:
1. Quote & premarket setup — assess the current/last price vs prior close. If outside regular US hours (9:30 AM–4:00 PM ET), treat any premarket move as provisional.
2. News scan — identify what is driving the stock right now. Judge each item as material, rumor, or noise. Prioritize the freshest timestamps.
3. Sentiment — treat as a confirming or contrarian input, never gospel.
4. Fundamentals — assess profitability, growth, balance-sheet health, and valuation vs peers and own history.
5. Earnings context — flag if a print is imminent; it sharply raises near-term risk.
6. Positioning signals — analyst targets, insider activity, institutional flows for confirming or conflicting evidence.
7. Technicals — trend, key support/resistance, and momentum.
8. Synthesize — only now form a thesis.

If data is missing, stale, or unavailable, say so explicitly and lower your confidence. Never fill a gap with a guess.

HOW TO EXPRESS A CALL — probabilistic theses, not guarantees:
For every call output exactly these fields (used for structured parsing):
SIGNAL: [STRONG BUY / BUY / NEUTRAL / SELL / STRONG SELL]
THESIS: [central view, 1-2 sentences]
BULL_CASE: [scenario with rough probability, e.g. "60%: if X then ~$Y"]
BASE_CASE: [scenario with rough probability]
BEAR_CASE: [scenario with rough probability — these three sum to ~100%]
CONVICTION: [Low / Medium / High — one line why]
TIME_HORIZON: [intraday / swing / position]
CATALYST: [upcoming event that could move it]
INVALIDATION: [specific price level or event that proves the thesis wrong — MANDATORY]
ENTRY_POINTS: [suggested buy zones or levels]
TECHNICAL_SCORE: [1-10]
MOMENTUM_SCORE: [1-10]
NEWS_SCORE: [1-10]
SMART_MONEY_SCORE: [1-10]
MACRO_FIT_SCORE: [1-10]
RISK_LEVEL: [Low / Medium / High]
BUY_ZONE_1: [price or range, e.g. "$130–$133"]
BUY_ZONE_2: [price or range, optional]
BUY_ZONE_3: [price or range, optional]
MARKET_REGIME: [copy the regime label from injected context, e.g. BULL]

ANALYTICAL DISCIPLINE:
• Build and weigh the strongest bear case against your own bull view before concluding.
• Watch for and name your own biases: recency, anchoring, confirmation, narrative-chasing.
• Prefer base rates and ranges over point predictions.
• Calibrate honestly — if data is thin or conflicting, say conviction is Low.
• Separate the company from the stock from the trade — a great company can be a poor stock at the wrong price.
• Always label each data point as FACT (from provided data), INFERENCE (your reasoning), or OPINION (your judgment).
• Factor the current MARKET REGIME into every signal — a BUY in a bull market may be NEUTRAL in high volatility.
• In PANIC or BEAR regimes, only issue BUY for exceptional oversold setups with clear catalysts.
• Format responses exactly as instructed. Output only the structured fields above — no extra commentary outside the format."""


def _build_system_prompt(market_context_block: str = "", accuracy_str: str = "") -> str:
    """Compose the full system prompt: expert persona + live market context + accuracy history."""
    parts = [_BASE_SYSTEM_PROMPT]
    if market_context_block:
        parts.append(f"\nCURRENT MARKET CONDITIONS:\n{market_context_block}")
    if accuracy_str:
        parts.append(
            f"\n{accuracy_str}\n"
            "Use this accuracy history to calibrate your conviction — "
            "if past BUY calls have been right only 40% of the time, be more selective."
        )
    return "\n".join(parts)


def _get_client() -> AsyncGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    return AsyncGroq(api_key=api_key)


async def _get_context_and_accuracy(ticker: str = "", db=None) -> tuple[dict, str, str]:
    """
    Fetch live market context + prediction accuracy for a ticker's sector.
    Returns (ctx_dict, context_block_str, accuracy_str).
    """
    from services import market_context as mc

    ctx = {}
    ctx_block = ""
    accuracy_str = ""

    try:
        ctx = await mc.get_market_context()
        sector = mc.TICKER_SECTOR_MAP.get(ticker.upper())
        ctx_block = mc.format_for_prompt(ctx, ticker_sector=sector)
    except Exception as e:
        logger.warning(f"market context unavailable: {e}")

    if db is not None:
        try:
            acc = await db.get_prediction_accuracy(days=30)
            if acc.get("overall_total", 0) >= 10:
                lines = [
                    f"MODEL ACCURACY (last 30 days, {acc['overall_total']} evaluated predictions):",
                    f"  Overall: {acc['overall_rate']}% correct",
                ]
                for sig, s in acc["per_signal"].items():
                    if s["total"] >= 3:
                        lines.append(f"  {sig}: {s['rate']}% ({s['hits']}/{s['total']})")
                accuracy_str = "\n".join(lines)
        except Exception:
            pass

    return ctx, ctx_block, accuracy_str


async def analyze_stock(ticker: str, analysis_data: dict, db=None) -> dict:
    """
    Deep AI analysis of a stock enriched with live market context and accuracy history.
    Returns dict with thesis, 4 sub-scores, overall_signal, buy_zones, risk_level,
    entry_points, bear_case, catalyst, confidence.
    """
    ctx, ctx_block, accuracy_str = await _get_context_and_accuracy(ticker, db)
    prompt = _build_stock_analysis_prompt(ticker, analysis_data, ctx_block)
    regime = ctx.get("regime", "UNKNOWN")
    client = _get_client()

    system = _build_system_prompt(ctx_block, accuracy_str)

    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1400,
            temperature=0.25,
        )
        raw = response.choices[0].message.content
        result = _parse_stock_analysis(raw, ticker)
        result["market_regime"] = regime
        return result
    except Exception as e:
        logger.error(f"analyze_stock({ticker}): {e}")
        return _fallback_analysis(ticker)


def _build_stock_analysis_prompt(ticker: str, data: dict, market_context_block: str = "") -> str:
    ta = data.get("technical", {})
    sm = data.get("smart_money", {})
    news_items = data.get("news", [])
    info = data.get("info", {})

    news_text = "\n".join(f"- {n.get('title', '')}" for n in news_items[:6]) or "No recent news."
    insider_text = "\n".join(
        f"- {i.get('name', '')} ({i.get('title', '')}): {i.get('transaction', '')} "
        f"{i.get('shares', 0):,} shares"
        for i in sm.get("insiders", [])[:5]
    ) or "No recent insider activity."

    pe = info.get("trailingPE") or info.get("forwardPE")
    div = info.get("dividendYield")
    beta = info.get("beta")
    pre_price = info.get("preMarketPrice")
    pre_chg = info.get("preMarketChangePercent")

    premarket_line = (
        f"- Pre-Market Price: ${pre_price:.2f} ({pre_chg:+.2f}%)" if pre_price else ""
    )

    return f"""{market_context_block}

Perform a comprehensive investment analysis for {ticker} — {info.get('longName', ticker)}.

STOCK DATA:
- Current Price:    ${info.get('currentPrice') or info.get('regularMarketPrice', 'N/A')}
{premarket_line}
- 52W Range:        ${info.get('fiftyTwoWeekLow', 'N/A')} → ${info.get('fiftyTwoWeekHigh', 'N/A')}
- 52W Position:     {ta.get('range_pct', 'N/A')}% (0%=52W low, 100%=52W high)
- Market Cap:       ${info.get('marketCap', 0):,.0f}
- P/E (TTM/Fwd):    {round(pe, 1) if pe else 'N/A'}
- Div. Yield:       {f'{div*100:.2f}%' if div else 'None'}
- Beta:             {round(beta, 2) if beta else 'N/A'}
- Sector:           {info.get('sector', 'N/A')}

TECHNICAL INDICATORS:
- RSI (14):         {ta.get('rsi', 'N/A')} — {_rsi_context(ta.get('rsi'))}
- EMA 20/50/200:    {ta.get('ema20_label', 'N/A')} / {ta.get('ema50_label', 'N/A')} / {ta.get('ema200_label', 'N/A')}
- Above EMA 20/50:  {ta.get('above_ema20', 'N/A')} / {ta.get('above_ema50', 'N/A')}
- MACD:             {ta.get('macd_signal', 'N/A')}
- Volume Ratio:     {ta.get('volume_ratio', 'N/A')}x 20-day average
- OBV Trend:        {ta.get('obv_trend', 'N/A')}
- Candlestick:      {ta.get('pattern', 'None detected')}
- RS vs SPY:        {ta.get('relative_strength', 'N/A')}x (>1 = outperforming market)
- Bollinger:        {ta.get('bb_position', 'N/A')}

SMART MONEY FLOWS:
Top Institutional Holders:
{chr(10).join(f'  • {h.get("holder","")}: {h.get("pct_out",0):.1f}% ownership  (${h.get("value",0)/1e6:.0f}M)' for h in sm.get("institutions",[])[:3]) or "  N/A"}

Insider Transactions (SEC filings):
{insider_text}

RECENT NEWS (most recent first):
{news_text}

INSTRUCTIONS:
Work the mandatory workflow in order before forming any view:
1. MACRO FIT — does this stock suit the current regime and rate/dollar environment above?
2. TREND — primary trend direction, key support/resistance levels.
3. MOMENTUM — accelerating or fading? Volume confirmation or divergence?
4. CATALYST — next likely price-moving event (earnings, product, macro).
5. RISK — build the strongest bear case first, then your invalidation level.

Respond using ONLY these exact field labels, one per line, no extra commentary:

SIGNAL: [STRONG BUY / BUY / NEUTRAL / SELL / STRONG SELL]
THESIS: [3-4 sentences: macro fit → trend → momentum → near-term outlook]
BULL_CASE: [probability%, e.g. "55%: if X then price could reach $Y"]
BASE_CASE: [probability%, central scenario]
BEAR_CASE: [probability%, what breaks the thesis and target level]
CONVICTION: [Low / Medium / High — one line why]
TIME_HORIZON: [intraday / swing / position]
CATALYST: [most important upcoming catalyst with approximate timing]
INVALIDATION: [specific price or event that proves the thesis wrong]
ENTRY_POINTS: [specific entry strategy with conditions]
TECHNICAL_SCORE: [1-10]
MOMENTUM_SCORE: [1-10]
NEWS_SCORE: [1-10]
SMART_MONEY_SCORE: [1-10]
MACRO_FIT_SCORE: [1-10]
RISK_LEVEL: [Low / Medium / High]
BUY_ZONE_1: [price or range, e.g. $130–$133]
BUY_ZONE_2: [price or range, optional]
BUY_ZONE_3: [price or range, optional]
MARKET_REGIME: [copy the regime label from the injected context, e.g. BULL]"""


def _rsi_context(rsi) -> str:
    """Return a brief RSI interpretation for the prompt."""
    if rsi is None:
        return "N/A"
    rsi = float(rsi)
    if rsi >= 80:   return "extremely overbought"
    if rsi >= 70:   return "overbought"
    if rsi >= 60:   return "bullish momentum"
    if rsi >= 50:   return "neutral-bullish"
    if rsi >= 40:   return "neutral-bearish"
    if rsi >= 30:   return "oversold"
    return "extremely oversold"


def _parse_stock_analysis(raw: str, ticker: str) -> dict:
    result = {
        "thesis": "",
        "bull_case": "",
        "base_case": "",
        "bear_case": "",
        "catalyst": "",
        "invalidation": "",
        "time_horizon": "",
        "momentum_score": 5,
        "smart_money_score": 5,
        "news_score": 5,
        "technical_score": 5,
        "macro_fit_score": 5,
        "overall_signal": "NEUTRAL",
        "confidence": "Medium",
        "buy_zones": [],
        "risk_level": "Medium",
        "entry_points": "",
        "market_regime": "",
        "raw": raw,
    }
    buy_zone_parts: list[str] = []

    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        key, _, val = line.partition(":")
        val = val.strip()
        k = key.strip().upper()

        match k:
            case "THESIS":
                result["thesis"] = val
            case "BULL_CASE":
                result["bull_case"] = val
            case "BASE_CASE":
                result["base_case"] = val
            case "BEAR_CASE":
                result["bear_case"] = val
            case "CATALYST":
                result["catalyst"] = val
            case "INVALIDATION":
                result["invalidation"] = val
            case "TIME_HORIZON":
                result["time_horizon"] = val
            case "ENTRY_POINTS":
                result["entry_points"] = val
            case "MARKET_REGIME":
                result["market_regime"] = val
            case "SIGNAL" | "OVERALL_SIGNAL":
                result["overall_signal"] = val
            case "CONVICTION" | "CONFIDENCE":
                result["confidence"] = val
            case "RISK_LEVEL":
                result["risk_level"] = val
            case "MOMENTUM_SCORE":
                try: result["momentum_score"] = int(val.split()[0])
                except Exception: pass
            case "SMART_MONEY_SCORE":
                try: result["smart_money_score"] = int(val.split()[0])
                except Exception: pass
            case "NEWS_SCORE":
                try: result["news_score"] = int(val.split()[0])
                except Exception: pass
            case "TECHNICAL_SCORE":
                try: result["technical_score"] = int(val.split()[0])
                except Exception: pass
            case "MACRO_FIT_SCORE":
                try: result["macro_fit_score"] = int(val.split()[0])
                except Exception: pass
            case "BUY_ZONE_1" | "BUY_ZONE_2" | "BUY_ZONE_3":
                if val and val.lower() not in ("n/a", "optional", "none", ""):
                    buy_zone_parts.append(val)
            case "BUY_ZONES":
                # Legacy fallback if model still outputs comma-separated
                buy_zone_parts.extend(z.strip() for z in val.split(",") if z.strip())

    result["buy_zones"] = buy_zone_parts or result["buy_zones"]
    return result


def _fallback_analysis(ticker: str) -> dict:
    return {
        "thesis": f"AI analysis temporarily unavailable for {ticker}.",
        "bull_case": "",
        "base_case": "",
        "bear_case": "",
        "catalyst": "",
        "invalidation": "",
        "time_horizon": "",
        "momentum_score": 5,
        "smart_money_score": 5,
        "news_score": 5,
        "technical_score": 5,
        "macro_fit_score": 5,
        "overall_signal": "NEUTRAL",
        "confidence": "Low",
        "buy_zones": [],
        "risk_level": "Medium",
        "entry_points": "N/A",
        "market_regime": "",
        "raw": "",
    }


async def analyze_for_daily_report(ticker: str, analysis_data: dict, db=None) -> dict:
    """
    Market-aware bulk analysis for daily reports and scanning.
    Injects live market regime and sector context so scores reflect current conditions.
    """
    ta = analysis_data.get("technical", {})
    info = analysis_data.get("info", {})
    news_titles = [n.get("title", "") for n in analysis_data.get("news", [])[:4]]

    ctx, ctx_block, _ = await _get_context_and_accuracy(ticker, db)
    regime = ctx.get("regime", "")

    # Compact sector-specific context
    from services import market_context as mc
    sector = mc.TICKER_SECTOR_MAP.get(ticker.upper(), "")
    sector_line = ""
    if sector and ctx.get("sectors"):
        sec_data = ctx["sectors"].get(sector, {})
        chg = sec_data.get("change_pct")
        if chg is not None:
            sector_line = f"Sector ({sector}): {chg:+.2f}% today\n"

    prompt = f"""Market regime: {ctx.get('regime_label', 'Unknown')}
{sector_line}
Stock: {ticker}
Price: ${info.get('currentPrice') or info.get('regularMarketPrice', 'N/A')}
Pre-market: {f"{info.get('preMarketPrice', 'N/A')} ({info.get('preMarketChangePercent', 0):+.2f}%)" if info.get('preMarketPrice') else 'N/A'}
RSI(14): {ta.get('rsi', 'N/A')} — {_rsi_context(ta.get('rsi'))}
EMA trend: {'Bullish (above EMA20)' if ta.get('above_ema20') else 'Bearish (below EMA20)'}
EMA50: {'above' if ta.get('above_ema50') else 'below'}  EMA200: {'above' if ta.get('above_ema200') else 'below'}
Volume: {ta.get('volume_ratio', 'N/A')}x avg  |  RS vs SPY: {ta.get('relative_strength', 'N/A')}x
52W position: {ta.get('range_pct', 'N/A')}%  |  MACD: {ta.get('macd_signal', 'N/A')}
News: {'; '.join(news_titles) or 'None'}

Given the market regime above, rate this stock. In {regime or 'this'} conditions, be appropriately conservative.

Respond ONLY in this exact format:
SIGNAL: [STRONG BUY / BUY / NEUTRAL / SELL / STRONG SELL]
SCORE: [1-10 conviction score adjusted for market regime]
THESIS: [one sentence — include regime context if relevant]
BUY_ZONE: [single price level or N/A]"""

    system = _build_system_prompt(ctx_block, "")
    # Append concise-output instruction so bulk calls stay fast
    system += "\n\nFor this bulk scan: be extremely concise — output ONLY the four requested lines, no extra text."

    client = _get_client()
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=120,
            temperature=0.2,
        )
        raw = response.choices[0].message.content
        result: dict = {"signal": "NEUTRAL", "score": 5, "thesis": "", "buy_zone": "N/A"}
        for line in raw.strip().splitlines():
            line = line.strip()
            key, _, val = line.partition(":")
            val = val.strip()
            k = key.strip().upper()
            if k == "SIGNAL":
                result["signal"] = val
            elif k == "SCORE":
                try: result["score"] = int(val.split()[0])
                except Exception: pass
            elif k == "THESIS":
                result["thesis"] = val
            elif k == "BUY_ZONE":
                result["buy_zone"] = val
        return result
    except Exception as e:
        logger.error(f"analyze_for_daily_report({ticker}): {e}")
        return {"signal": "NEUTRAL", "score": 5, "thesis": "Analysis unavailable.", "buy_zone": "N/A"}


async def summarize_text(text: str, source_url: str = "") -> str:
    """Summarize article text using Groq. Returns plain bullet-point string."""
    if not text or len(text.strip()) < 50:
        return "Not enough content to summarize."

    text = text[:4000]

    prompt = f"""Summarize this financial news article in 3-4 bullet points. Be concise and focus on market impact.

{'URL: ' + source_url if source_url else ''}

ARTICLE:
{text}

Format:
• [key point 1]
• [key point 2]
• [key point 3]
• [market implication, if any]"""

    client = _get_client()
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a financial news analyst. Summarize concisely."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"summarize_text: {e}")
        return "Summary unavailable — Groq API error."


async def summarize_text_rich(text: str, source_url: str = "") -> dict:
    """Summarize article text and return structured data for infographic rendering."""
    if not text or len(text.strip()) < 50:
        return _fallback_summary()

    text = text[:4000]

    prompt = f"""Analyze this financial news article and respond in EXACTLY this format (keep labels as-is):

{'URL: ' + source_url if source_url else ''}

ARTICLE:
{text}

TITLE: [6-10 word headline capturing the main topic]
SENTIMENT: [Bullish / Bearish / Neutral]
MARKET_IMPACT: [one short phrase, e.g. "Positive for semiconductor stocks"]
TICKERS: [comma-separated ticker symbols mentioned, e.g. AAPL, MSFT — or NONE]
BULLET1: [most important finding, max 18 words]
BULLET2: [second key point, max 18 words]
BULLET3: [third key point, max 18 words]
BULLET4: [market implication or outlook, max 18 words]"""

    system = (
        _BASE_SYSTEM_PROMPT
        + "\n\nFor this task: analyse and summarise the provided article "
        "using your financial expertise. Identify market impact, sentiment, "
        "and key tickers. Respond ONLY in the exact structured format requested."
    )
    client = _get_client()
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=400,
            temperature=0.3,
        )
        return _parse_rich_summary(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"summarize_text_rich: {e}")
        return _fallback_summary()


def _parse_rich_summary(raw: str) -> dict:
    result = _fallback_summary()
    bullets = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if line.startswith("TITLE:"):
            result["title"] = line[6:].strip()
        elif line.startswith("SENTIMENT:"):
            val = line[10:].strip().capitalize()
            if val in ("Bullish", "Bearish", "Neutral"):
                result["sentiment"] = val
        elif line.startswith("MARKET_IMPACT:"):
            result["market_impact"] = line[14:].strip()
        elif line.startswith("TICKERS:"):
            raw_tickers = line[8:].strip()
            if raw_tickers.upper() != "NONE":
                result["key_tickers"] = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]
        elif line.startswith("BULLET"):
            # BULLET1:, BULLET2:, etc.
            colon = line.index(":") if ":" in line else -1
            if colon != -1:
                b = line[colon + 1:].strip()
                if b:
                    bullets.append(b)
    result["bullets"] = bullets[:4]
    result["summary_text"] = "\n".join(f"• {b}" for b in result["bullets"])
    return result


def _fallback_summary() -> dict:
    return {
        "title": "Article Summary",
        "sentiment": "Neutral",
        "market_impact": "N/A",
        "key_tickers": [],
        "bullets": [],
        "summary_text": "Summary unavailable.",
    }


async def fetch_and_summarize_url(url: str) -> str:
    """Fetch URL content and return plain bullet-point summary string."""
    text = await _fetch_url_text(url)
    if not text:
        return "Could not fetch article content."
    return await summarize_text(text, source_url=url)


async def fetch_and_summarize_url_rich(url: str) -> dict:
    """Fetch URL content and return structured summary dict for infographic."""
    text = await _fetch_url_text(url)
    if not text:
        return _fallback_summary()
    return await summarize_text_rich(text, source_url=url)


async def score_news_hotness(article: dict) -> int:
    """Score an article 1-10 for market-moving impact. Returns 0 on failure."""
    title = article.get("title", "")
    source = article.get("source", "")
    if not title:
        return 0

    prompt = f"""Rate this financial news headline for market-moving importance on a scale of 1-10.
10 = breaking news (Fed decision, earnings miss/beat, M&A, major geopolitical event)
7-9 = significant (sector catalyst, large cap announcement, macro data)
4-6 = moderate (analyst upgrades, routine data, earnings previews)
1-3 = low impact (routine filings, minor updates)

Headline: {title}
Source: {source}

Respond with ONLY a single integer (1-10). No explanation."""

    client = _get_client()
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior financial news editor. "
                        "Rate headlines for market-moving importance. "
                        "Output ONLY a single integer."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=5,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        return min(10, max(1, int(raw.split()[0])))
    except Exception:
        return 0


async def _fetch_url_text(url: str) -> Optional[str]:
    """Fetch plain text from a URL using BeautifulSoup."""
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    return None
                html = await resp.text()
        soup = BeautifulSoup(html, "lxml")
        # Remove scripts, styles, nav
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        # Try article tag first, then body
        article = soup.find("article") or soup.find("main") or soup.body
        if article:
            text = article.get_text(separator=" ", strip=True)
            return text[:5000]
        return None
    except Exception as e:
        logger.error(f"_fetch_url_text({url}): {e}")
        return None
