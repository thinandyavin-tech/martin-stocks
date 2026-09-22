"""
Twitter monitoring via Nitter RSS — no API key required.
Polls @ShayBoloor every 5 minutes using 6 Nitter fallback instances.
"""

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Optional
import time

import aiohttp
import feedparser

logger = logging.getLogger(__name__)

TWITTER_USER = "ShayBoloor"

NITTER_INSTANCES = [
    "https://nitter.net",
    "https://nitter.privacydev.net",
    "https://nitter.poast.org",
    "https://nitter.1d4.us",
    "https://nitter.kavin.rocks",
    "https://twiiit.com",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

CASHTAG_RE = re.compile(r"\$([A-Z]{1,5})\b")


def _make_rss_url(instance: str, username: str) -> str:
    return f"{instance.rstrip('/')}/{username}/rss"


async def _fetch_feed_from_instance(instance: str) -> Optional[list[dict]]:
    """Attempt to fetch RSS from a single Nitter instance. Returns list of tweet dicts or None on failure."""
    url = _make_rss_url(instance, TWITTER_USER)
    try:
        async with aiohttp.ClientSession(headers=HEADERS) as session:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=10),
                allow_redirects=True,
            ) as resp:
                if resp.status != 200:
                    logger.debug(f"Nitter {instance} returned HTTP {resp.status}")
                    return None
                content = await resp.read()

        feed = feedparser.parse(content)
        if not feed.entries:
            logger.debug(f"Nitter {instance}: empty feed")
            return None

        tweets = []
        for entry in feed.entries:
            tweet_id = _extract_tweet_id(entry.get("link", ""), entry.get("id", ""))
            if not tweet_id:
                continue

            text = _clean_tweet_text(
                entry.get("title", "") or entry.get("summary", "")
            )

            # Parse timestamp
            published_dt = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    published_dt = datetime.fromtimestamp(
                        time.mktime(entry.published_parsed), tz=timezone.utc
                    )
                except Exception:
                    pass

            cashtags = CASHTAG_RE.findall(text)
            link = entry.get("link", "")
            # Normalise link to twitter.com for display
            twitter_link = _nitter_to_twitter(link)

            tweets.append({
                "id": tweet_id,
                "text": text,
                "published": published_dt,
                "cashtags": cashtags,
                "link": twitter_link,
                "nitter_link": link,
            })

        return tweets
    except asyncio.TimeoutError:
        logger.debug(f"Nitter {instance}: timeout")
        return None
    except Exception as e:
        logger.debug(f"Nitter {instance}: {e}")
        return None


async def fetch_tweets(seed_mode: bool = False) -> list[dict]:
    """
    Try each Nitter instance in order until one succeeds.
    Returns list of tweet dicts (newest first).
    In seed_mode, returns all tweets without filtering by seen status.
    """
    for instance in NITTER_INSTANCES:
        tweets = await _fetch_feed_from_instance(instance)
        if tweets is not None:
            logger.info(f"Nitter: fetched {len(tweets)} tweets from {instance}")
            return tweets

    logger.warning("All Nitter instances failed.")
    return []


async def get_new_tweets(db) -> list[dict]:
    """
    Fetch tweets and return only the ones we haven't seen before.
    Marks new tweets as seen in the database.
    """
    all_tweets = await fetch_tweets()
    new_tweets = []
    for tweet in all_tweets:
        if not await db.has_seen_tweet(tweet["id"]):
            new_tweets.append(tweet)
            await db.mark_tweet_seen(tweet["id"])
    return new_tweets


async def seed_seen_tweets(db):
    """
    On startup, mark all current tweets as seen to prevent spam.
    """
    tweets = await fetch_tweets(seed_mode=True)
    if tweets:
        ids = [t["id"] for t in tweets]
        await db.bulk_mark_tweets_seen(ids)
        logger.info(f"Seeded {len(ids)} existing tweets as seen on startup")
    else:
        logger.warning("Seed: could not fetch tweets from any Nitter instance")


def _extract_tweet_id(link: str, entry_id: str) -> Optional[str]:
    """Extract numeric tweet ID from URL or entry id."""
    for s in [link, entry_id]:
        match = re.search(r"/status/(\d+)", s)
        if match:
            return match.group(1)
    # Fallback: use last numeric segment
    parts = link.rstrip("/").split("/")
    for part in reversed(parts):
        if part.isdigit():
            return part
    return None


def _clean_tweet_text(raw: str) -> str:
    """Strip HTML tags from Nitter RSS tweet text."""
    from bs4 import BeautifulSoup
    if "<" in raw:
        soup = BeautifulSoup(raw, "lxml")
        text = soup.get_text(separator=" ", strip=True)
    else:
        text = raw.strip()
    # Collapse whitespace
    return re.sub(r"\s+", " ", text).strip()


def _nitter_to_twitter(nitter_url: str) -> str:
    """Convert a Nitter URL to its canonical twitter.com equivalent."""
    for instance in NITTER_INSTANCES:
        host = instance.rstrip("/")
        if nitter_url.startswith(host):
            path = nitter_url[len(host):]
            return f"https://twitter.com{path}"
    # Generic: replace the domain
    try:
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(nitter_url)
        return urlunparse(parsed._replace(netloc="twitter.com", scheme="https"))
    except Exception:
        return nitter_url


def format_relative_time(dt: Optional[datetime]) -> str:
    """Return human-readable relative time string."""
    if dt is None:
        return "just now"
    now = datetime.now(tz=timezone.utc)
    diff = now - dt
    seconds = int(diff.total_seconds())
    if seconds < 60:
        return f"{seconds}s ago"
    elif seconds < 3600:
        return f"{seconds // 60}m ago"
    elif seconds < 86400:
        return f"{seconds // 3600}h ago"
    else:
        return f"{seconds // 86400}d ago"
