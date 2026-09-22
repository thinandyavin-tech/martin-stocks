"""
News service — fetches articles from multiple RSS feeds.
Sources: Reuters, CNBC, MarketWatch, Seeking Alpha, Yahoo Finance
"""

import asyncio
import logging
from typing import Optional
from datetime import datetime, timezone

import aiohttp
import feedparser
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

RSS_FEEDS = {
    "Reuters": "https://feeds.reuters.com/reuters/businessNews",
    "Reuters Markets": "https://feeds.reuters.com/reuters/topNews",
    "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "CNBC Markets": "https://www.cnbc.com/id/20910258/device/rss/rss.html",
    "MarketWatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "MarketWatch Markets": "https://feeds.marketwatch.com/marketwatch/marketpulse/",
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
    "Seeking Alpha": "https://seekingalpha.com/feed.xml",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


async def fetch_feed(name: str, url: str, limit: int = 5) -> list[dict]:
    """Fetch and parse a single RSS feed, returning list of article dicts."""
    try:
        async with aiohttp.ClientSession(headers=HEADERS) as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return []
                content = await resp.read()

        feed = feedparser.parse(content)
        articles = []
        for entry in feed.entries[:limit]:
            published = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    import time
                    published = datetime.fromtimestamp(
                        time.mktime(entry.published_parsed), tz=timezone.utc
                    ).strftime("%Y-%m-%d %H:%M UTC")
                except Exception:
                    pass

            # Extract clean summary
            summary = ""
            if hasattr(entry, "summary"):
                soup = BeautifulSoup(entry.summary, "lxml")
                summary = soup.get_text(strip=True)[:300]

            articles.append({
                "source": name,
                "title": entry.get("title", "No title").strip(),
                "url": entry.get("link", ""),
                "summary": summary,
                "published": published or "Recent",
            })
        return articles
    except Exception as e:
        logger.warning(f"feed fetch failed [{name}]: {e}")
        return []


async def get_all_news(limit_per_feed: int = 5) -> list[dict]:
    """Fetch news from all RSS feeds concurrently."""
    tasks = [
        fetch_feed(name, url, limit_per_feed)
        for name, url in RSS_FEEDS.items()
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    all_articles = []
    for r in results:
        if isinstance(r, list):
            all_articles.extend(r)

    # Deduplicate by title similarity (simple substring match)
    seen_titles = set()
    unique = []
    for art in all_articles:
        title_key = art["title"][:60].lower()
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique.append(art)

    return unique


async def get_stock_news(ticker: str, company_name: str = "", limit: int = 8) -> list[dict]:
    """
    Fetch news relevant to a specific ticker from Yahoo Finance RSS + general feeds.
    """
    yahoo_url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
    google_url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"

    tasks = [
        fetch_feed(f"Yahoo Finance ({ticker})", yahoo_url, limit),
        fetch_feed(f"Google News ({ticker})", google_url, limit),
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    articles = []
    for r in results:
        if isinstance(r, list):
            articles.extend(r)

    # If we got nothing from stock-specific feeds, filter general news
    if not articles:
        all_news = await get_all_news(limit_per_feed=10)
        search_terms = [ticker.lower(), company_name.lower()] if company_name else [ticker.lower()]
        articles = [
            a for a in all_news
            if any(term in a["title"].lower() for term in search_terms if term)
        ][:limit]

    # Deduplicate
    seen = set()
    unique = []
    for art in articles:
        key = art["title"][:60].lower()
        if key not in seen:
            seen.add(key)
            unique.append(art)

    return unique[:limit]


async def fetch_article_text(url: str) -> Optional[str]:
    """Fetch the full text of an article for AI summarization."""
    try:
        async with aiohttp.ClientSession(headers=HEADERS) as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    return None
                html = await resp.text()

        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "ads"]):
            tag.decompose()

        # Try semantic article tag first
        article = soup.find("article") or soup.find("main") or soup.body
        if article:
            text = article.get_text(separator=" ", strip=True)
            # Clean up excess whitespace
            import re
            text = re.sub(r"\s+", " ", text)
            return text[:5000]
        return None
    except Exception as e:
        logger.error(f"fetch_article_text({url}): {e}")
        return None
