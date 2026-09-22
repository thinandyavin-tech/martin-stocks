"""
APScheduler background jobs:
1. check_price_alerts()  — every 5 minutes
2. poll_twitter()        — every 5 minutes
3. run_daily_report()    — daily at 8:00 AM UTC (cron)
"""

import asyncio
import logging

import discord
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from services import market_data as md
from services import twitter_service as twitter
from services import news_service as news
from services import groq_service as groq
from utils.helpers import fmt_price, change_arrow, change_color_str

logger = logging.getLogger(__name__)


def setup_scheduler(bot) -> AsyncIOScheduler:
    """Create and configure the AsyncIO scheduler. Caller is responsible for .start()."""
    scheduler = AsyncIOScheduler(timezone="UTC")

    # Price alert check — every 5 minutes
    scheduler.add_job(
        _check_price_alerts,
        trigger=IntervalTrigger(minutes=5),
        id="price_alerts",
        name="Price Alert Checker",
        args=[bot],
        max_instances=1,
        coalesce=True,
        misfire_grace_time=60,
    )

    # Twitter poll — every 5 minutes (offset by 2.5 min from alerts)
    scheduler.add_job(
        _poll_twitter,
        trigger=IntervalTrigger(minutes=5, seconds=30),
        id="twitter_poll",
        name="Twitter Poller",
        args=[bot],
        max_instances=1,
        coalesce=True,
        misfire_grace_time=60,
    )

    # Daily report — 8:00 AM UTC
    scheduler.add_job(
        _run_daily_report,
        trigger=CronTrigger(hour=8, minute=0, timezone="UTC"),
        id="daily_report",
        name="Daily AI Report",
        args=[bot],
        max_instances=1,
        coalesce=True,
        misfire_grace_time=300,
    )

    # Hot news auto-post — every 20 minutes
    scheduler.add_job(
        _check_hot_news,
        trigger=IntervalTrigger(minutes=20),
        id="hot_news",
        name="Hot News Monitor",
        args=[bot],
        max_instances=1,
        coalesce=True,
        misfire_grace_time=120,
    )

    # Pre-market briefing — 9:20 AM ET (13:20 UTC), weekdays only
    scheduler.add_job(
        _market_open_alert,
        trigger=CronTrigger(hour=13, minute=20, day_of_week="mon-fri", timezone="UTC"),
        id="market_open_alert",
        name="Market Open Alert",
        args=[bot],
        max_instances=1,
        coalesce=True,
        misfire_grace_time=300,
    )

    # Prediction evaluator — daily at 21:30 UTC (after US market close + 2h buffer)
    scheduler.add_job(
        _evaluate_predictions,
        trigger=CronTrigger(hour=21, minute=30, day_of_week="mon-fri", timezone="UTC"),
        id="prediction_eval",
        name="Prediction Accuracy Evaluator",
        args=[bot],
        max_instances=1,
        coalesce=True,
        misfire_grace_time=600,
    )

    logger.info(
        "Scheduler configured: price_alerts (5min), twitter (5min+30s), "
        "daily_report (08:00 UTC), hot_news (20min), market_open_alert (13:20 UTC Mon-Fri), "
        "prediction_eval (21:30 UTC Mon-Fri)"
    )
    return scheduler


# ─── Job: Price Alert Checker ────────────────────────────────────────────────

async def _check_price_alerts(bot):
    """
    Check all active price alerts against current market prices.
    If a target is hit, ping the user in the configured channel and mark the alert as triggered.
    """
    try:
        alerts = await bot.db.get_active_alerts()
        if not alerts:
            return

        # Group by ticker to minimize API calls
        tickers = list({a["ticker"] for a in alerts})
        if not tickers:
            return

        prices = await md.batch_get_prices(tickers)

        for alert in alerts:
            ticker = alert["ticker"]
            current_price = prices.get(ticker)
            if current_price is None:
                continue

            target = alert["target_price"]
            direction = alert["direction"]

            triggered = (
                (direction == "above" and current_price >= target)
                or (direction == "below" and current_price <= target)
            )

            if triggered:
                await _fire_alert(bot, alert, current_price)

    except Exception as e:
        logger.error(f"_check_price_alerts: {e}")


async def _fire_alert(bot, alert: dict, current_price: float):
    """Send alert notification and mark as triggered."""
    try:
        channel = bot.get_channel(int(alert["channel_id"]))
        if channel is None:
            # Try fetching it
            try:
                channel = await bot.fetch_channel(int(alert["channel_id"]))
            except Exception:
                logger.warning(f"Alert channel {alert['channel_id']} not found")
                await bot.db.mark_alert_triggered(alert["id"])
                return

        direction_text = "risen above" if alert["direction"] == "above" else "fallen below"
        cc = change_color_str(current_price - alert["target_price"])

        embed = discord.Embed(
            title=f"🔔 Price Alert Triggered — {alert['ticker']}",
            description=(
                f"<@{alert['user_id']}> Your alert fired!\n\n"
                f"**{alert['ticker']}** has {direction_text} **${alert['target_price']:.2f}**\n"
                f"Current price: **{fmt_price(current_price)}** {cc}"
            ),
            color=discord.Color.gold(),
        )
        embed.set_footer(text="Martin's Stocks • Alert system")

        await channel.send(content=f"<@{alert['user_id']}>", embed=embed)
        await bot.db.mark_alert_triggered(alert["id"])
        logger.info(f"Alert #{alert['id']} fired: {alert['ticker']} @ {current_price}")

    except discord.Forbidden:
        logger.warning(f"Cannot send alert to channel {alert['channel_id']}: Forbidden")
        await bot.db.mark_alert_triggered(alert["id"])
    except Exception as e:
        logger.error(f"_fire_alert #{alert['id']}: {e}")


# ─── Job: Twitter Poller ─────────────────────────────────────────────────────

async def _poll_twitter(bot):
    """
    Fetch new @ShayBoloor tweets from Nitter RSS.
    For each new tweet, post to all enabled channels.
    """
    try:
        new_tweets = await twitter.get_new_tweets(bot.db)
        if not new_tweets:
            return

        channels_ids = await bot.db.get_twitter_channels()
        if not channels_ids:
            return

        for tweet in new_tweets:
            embed = _build_tweet_embed(tweet)
            for cid in channels_ids:
                try:
                    channel = bot.get_channel(int(cid))
                    if channel is None:
                        channel = await bot.fetch_channel(int(cid))
                    await channel.send(embed=embed)
                except discord.Forbidden:
                    logger.warning(f"Twitter post: no permission in channel {cid}")
                except Exception as e:
                    logger.error(f"Twitter post to {cid}: {e}")

        logger.info(f"Posted {len(new_tweets)} new tweet(s) to {len(channels_ids)} channel(s)")

    except Exception as e:
        logger.error(f"_poll_twitter: {e}")


def _build_tweet_embed(tweet: dict) -> discord.Embed:
    """Build a Discord embed for a single tweet."""
    from services.twitter_service import format_relative_time, TWITTER_USER

    text = tweet.get("text", "")
    cashtags = tweet.get("cashtags", [])
    link = tweet.get("link", "")
    published = tweet.get("published")
    relative_time = format_relative_time(published)

    # Build cashtag clickable tags
    cashtag_links = ""
    if cashtags:
        tags = []
        for ct in cashtags[:5]:
            tags.append(f"[${ct}](https://finance.yahoo.com/quote/{ct})")
        cashtag_links = "\n\n📈 " + " · ".join(tags)

    embed = discord.Embed(
        description=text + cashtag_links,
        color=discord.Color.from_str("#1da1f2"),
        url=link,
    )
    embed.set_author(
        name=f"@{TWITTER_USER}",
        url=f"https://twitter.com/{TWITTER_USER}",
        icon_url="https://abs.twimg.com/icons/apple-touch-icon-192x192.png",
    )
    embed.add_field(
        name="",
        value=f"🕐 {relative_time} • [View on Twitter]({link})",
        inline=False,
    )
    embed.set_footer(text="Martin's Stocks • Twitter monitoring via Nitter RSS")
    return embed


# ─── Job: Hot News Monitor ───────────────────────────────────────────────────

# Minimum Groq hotness score to auto-post (1-10)
HOT_NEWS_THRESHOLD = 7


async def _check_hot_news(bot):
    """
    Fetch the latest headlines every 20 minutes, score each with Groq,
    and auto-post any article scoring >= HOT_NEWS_THRESHOLD to all daily-report channels.
    """
    try:
        channels = await bot.db.get_daily_report_channels()
        if not channels:
            return

        articles = await news.get_all_news(limit_per_feed=3)
        if not articles:
            return

        # Filter to unseen articles first (cheap check before calling Groq)
        unseen = []
        for article in articles:
            if article.get("url") and not await bot.db.has_seen_news(article["url"]):
                unseen.append(article)

        if not unseen:
            return

        # Score up to 8 unseen articles in parallel to avoid hammering Groq
        scores = await asyncio.gather(
            *[groq.score_news_hotness(a) for a in unseen[:8]],
            return_exceptions=True,
        )

        hot = []
        for article, score in zip(unseen[:8], scores):
            if isinstance(score, Exception):
                score = 0
            await bot.db.mark_news_seen(article["url"], article.get("title", ""))
            if score >= HOT_NEWS_THRESHOLD:
                hot.append((article, score))

        if not hot:
            return

        logger.info(f"Hot news: {len(hot)} article(s) scoring >= {HOT_NEWS_THRESHOLD}")

        for article, score in hot:
            embed = _build_hot_news_embed(article, score)
            for cid in channels:
                try:
                    channel = bot.get_channel(int(cid))
                    if channel is None:
                        channel = await bot.fetch_channel(int(cid))
                    await channel.send(embed=embed)
                except discord.Forbidden:
                    logger.warning(f"Hot news: no permission in channel {cid}")
                except Exception as e:
                    logger.error(f"Hot news post to {cid}: {e}")

    except Exception as e:
        logger.error(f"_check_hot_news: {e}")


def _build_hot_news_embed(article: dict, score: int) -> discord.Embed:
    """Build a Discord embed for a high-impact news article."""
    title = article.get("title", "Breaking News")
    url = article.get("url", "")
    source = article.get("source", "")
    published = article.get("published", "")

    # Score-based color: 10 = brightest red, 7 = orange
    color = discord.Color.from_str("#da3633") if score >= 9 else discord.Color.from_str("#d29922")

    heat = "🔥" * min(score - 6, 4)  # 1-4 flames based on score 7-10

    embed = discord.Embed(
        title=f"{heat} Breaking: {truncate_for_embed(title, 200)}",
        url=url or None,
        color=color,
    )
    embed.add_field(name="Source", value=source or "Unknown", inline=True)
    embed.add_field(name="Impact Score", value=f"**{score}/10**", inline=True)
    if published:
        embed.add_field(name="Published", value=published, inline=True)
    embed.set_footer(text="Martin's Stocks • Auto-posted by Hot News Monitor")
    return embed


def truncate_for_embed(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit - 1] + "…"


# ─── Job: Daily Report ───────────────────────────────────────────────────────

async def _run_daily_report(bot):
    """Wrapper that calls the daily report cog function."""
    try:
        from cogs.daily_report import send_daily_report
        await send_daily_report(bot)
    except Exception as e:
        logger.error(f"_run_daily_report: {e}")


# ─── Job: Prediction Accuracy Evaluator ──────────────────────────────────────

async def _evaluate_predictions(bot):
    """
    Grade AI predictions that are at least 5 trading days old.
    BUY/STRONG BUY = correct if price rose > 1%.
    SELL/STRONG SELL = correct if price fell > 1%.
    NEUTRAL = correct if price stayed within ±2%.
    Records outcomes to build the accuracy history fed back into future prompts.
    """
    try:
        pending = await bot.db.get_unevaluated_predictions(min_age_hours=120)
        if not pending:
            return

        tickers = list({p["ticker"] for p in pending})
        prices = await md.batch_get_prices(tickers)

        evaluated = 0
        for pred in pending:
            ticker = pred["ticker"]
            current = prices.get(ticker)
            if current is None:
                continue

            original = pred["price_at_call"]
            if original <= 0:
                continue

            pct_change = (current - original) / original * 100
            signal = pred["signal"].upper()

            if "BUY" in signal:
                correct = pct_change > 1.0
                outcome = f"Price moved {pct_change:+.1f}% — {'✓ BUY confirmed' if correct else '✗ BUY failed'}"
            elif "SELL" in signal:
                correct = pct_change < -1.0
                outcome = f"Price moved {pct_change:+.1f}% — {'✓ SELL confirmed' if correct else '✗ SELL failed'}"
            else:
                correct = abs(pct_change) <= 2.0
                outcome = f"Price moved {pct_change:+.1f}% — {'✓ NEUTRAL confirmed' if correct else '✗ NEUTRAL missed'}"

            await bot.db.record_prediction_outcome(pred["id"], current, outcome, correct)
            evaluated += 1

        if evaluated:
            acc = await bot.db.get_prediction_accuracy(days=30)
            rate = acc.get("overall_rate")
            logger.info(
                f"Prediction evaluator: graded {evaluated} predictions — "
                f"30-day accuracy: {rate}% ({acc.get('overall_hits')}/{acc.get('overall_total')})"
            )

    except Exception as e:
        logger.error(f"_evaluate_predictions: {e}")


# ─── Job: Market Open Alert ───────────────────────────────────────────────────

_PREMARKET_WATCH = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "META"]


async def _market_open_alert(bot):
    """
    Post a premarket briefing 10 minutes before US market opens (9:30 AM ET).
    Shows top premarket movers and any overnight headlines.
    """
    try:
        channels = await bot.db.get_daily_report_channels()
        if not channels:
            return

        async def _get_premarket(ticker: str) -> dict | None:
            try:
                info = await md.get_stock_info(ticker)
                pre_price = info.get("preMarketPrice")
                if not pre_price:
                    return None
                pre_chg = info.get("preMarketChangePercent") or 0
                return {
                    "ticker": ticker,
                    "name": info.get("shortName") or ticker,
                    "pre_price": pre_price,
                    "pre_chg": pre_chg,
                }
            except Exception:
                return None

        pm_results = await asyncio.gather(
            *[_get_premarket(t) for t in _PREMARKET_WATCH],
            return_exceptions=True,
        )
        pm_data = sorted(
            [r for r in pm_results if isinstance(r, dict)],
            key=lambda x: abs(x.get("pre_chg", 0)),
            reverse=True,
        )

        embed = discord.Embed(
            title="🔔 Market Opens in 10 Minutes!",
            description="US market opens at **9:30 AM ET**. Here's your premarket snapshot:",
            color=discord.Color.from_str("#f0883e"),
        )

        if pm_data:
            lines = []
            for item in pm_data[:6]:
                arrow = "▲" if item["pre_chg"] >= 0 else "▼"
                dot = "🟢" if item["pre_chg"] >= 0 else "🔴"
                lines.append(
                    f"{dot} **{item['ticker']}** {arrow}{abs(item['pre_chg']):.2f}%  "
                    f"@ **{fmt_price(item['pre_price'])}**"
                )
            embed.add_field(
                name="📊 Premarket Movers",
                value="\n".join(lines),
                inline=False,
            )

        try:
            articles = await news.get_all_news(limit_per_feed=2)
            unseen = [
                a for a in (articles or [])[:6]
                if a.get("url") and not await bot.db.has_seen_news(a["url"])
            ]
            if unseen:
                news_lines = [
                    f"• [{truncate_for_embed(a.get('title', ''), 80)}]({a['url']})"
                    for a in unseen[:4]
                ]
                embed.add_field(
                    name="📰 Overnight Headlines",
                    value="\n".join(news_lines),
                    inline=False,
                )
        except Exception:
            pass

        embed.add_field(
            name="💡 Quick Actions",
            value="Use `/movers` after open • `/screener` for setups • `/recommendations` for AI picks",
            inline=False,
        )
        embed.set_footer(text="Martin's Stocks • Premarket briefing • Data via Yahoo Finance")

        for cid in channels:
            try:
                channel = bot.get_channel(int(cid))
                if channel is None:
                    channel = await bot.fetch_channel(int(cid))
                await channel.send(embed=embed)
            except discord.Forbidden:
                logger.warning(f"Market open alert: no permission in channel {cid}")
            except Exception as e:
                logger.error(f"Market open alert to {cid}: {e}")

        logger.info(f"Market open alert posted to {len(channels)} channel(s)")

    except Exception as e:
        logger.error(f"_market_open_alert: {e}")


# ─── Startup Seed ────────────────────────────────────────────────────────────

async def seed_twitter_on_startup(bot):
    """
    Called from bot.setup_hook() — seeds existing tweets as seen and
    pre-warms the market context cache so the first AI analysis is fast.
    """
    try:
        await twitter.seed_seen_tweets(bot.db)
    except Exception as e:
        logger.error(f"seed_twitter_on_startup: {e}")

    # Pre-warm market context cache in background (non-blocking)
    try:
        from services import market_context as mc
        asyncio.create_task(mc.warm_cache())
        logger.info("Market context cache warm-up started.")
    except Exception as e:
        logger.error(f"market_context warm_cache: {e}")
