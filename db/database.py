"""
Async SQLite database layer for Martin's Stocks bot.
Tables: watchlists, price_alerts, seen_tweets, twitter_channels, daily_report_channels,
        ai_predictions, seen_news, premium_users, bot_config
"""

import aiosqlite
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "martin_stocks.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS ai_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    signal TEXT NOT NULL,
    score INTEGER NOT NULL,
    price_at_call REAL NOT NULL,
    market_regime TEXT,
    source TEXT DEFAULT 'analysis',
    called_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    evaluated_at TIMESTAMP NULL,
    price_at_eval REAL NULL,
    outcome TEXT NULL,
    correct INTEGER NULL
);
CREATE INDEX IF NOT EXISTS idx_pred_ticker ON ai_predictions(ticker);
CREATE INDEX IF NOT EXISTS idx_pred_called ON ai_predictions(called_at);
CREATE INDEX IF NOT EXISTS idx_pred_eval ON ai_predictions(correct, called_at);

CREATE TABLE IF NOT EXISTS seen_news (
    url_hash TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    headline TEXT,
    seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS premium_users (
    user_id TEXT PRIMARY KEY,
    granted_by TEXT NOT NULL,
    granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NULL,
    note TEXT
);

CREATE TABLE IF NOT EXISTS bot_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS watchlists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, ticker)
);

CREATE TABLE IF NOT EXISTS price_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    target_price REAL NOT NULL,
    direction TEXT NOT NULL,
    triggered INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS seen_tweets (
    tweet_id TEXT PRIMARY KEY,
    seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS twitter_channels (
    channel_id TEXT PRIMARY KEY,
    guild_id TEXT NOT NULL,
    enabled INTEGER DEFAULT 1,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS daily_report_channels (
    channel_id TEXT PRIMARY KEY,
    guild_id TEXT NOT NULL,
    enabled INTEGER DEFAULT 1,
    report_hour INTEGER DEFAULT 8,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


class Database:
    def __init__(self):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(DB_PATH)

    async def initialize(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(SCHEMA)
            await db.commit()
        logger.info(f"Database initialized at {self.db_path}")

    # ─── Watchlist ───────────────────────────────────────────────────────────

    async def add_to_watchlist(self, user_id: str, ticker: str) -> tuple[bool, str]:
        """Returns (success, message)."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    "SELECT COUNT(*) FROM watchlists WHERE user_id = ?", (user_id,)
                ) as cur:
                    count = (await cur.fetchone())[0]
                if count >= 20:
                    return False, "Watchlist is full (max 20 tickers)."
                await db.execute(
                    "INSERT OR IGNORE INTO watchlists (user_id, ticker) VALUES (?, ?)",
                    (user_id, ticker.upper()),
                )
                await db.commit()
            return True, f"Added **{ticker.upper()}** to your watchlist."
        except Exception as e:
            logger.error(f"add_to_watchlist error: {e}")
            return False, "Database error."

    async def remove_from_watchlist(self, user_id: str, ticker: str) -> tuple[bool, str]:
        try:
            async with aiosqlite.connect(self.db_path) as db:
                result = await db.execute(
                    "DELETE FROM watchlists WHERE user_id = ? AND ticker = ?",
                    (user_id, ticker.upper()),
                )
                await db.commit()
                if result.rowcount == 0:
                    return False, f"**{ticker.upper()}** was not in your watchlist."
            return True, f"Removed **{ticker.upper()}** from your watchlist."
        except Exception as e:
            logger.error(f"remove_from_watchlist error: {e}")
            return False, "Database error."

    async def get_watchlist(self, user_id: str) -> list[str]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT ticker FROM watchlists WHERE user_id = ? ORDER BY added_at",
                (user_id,),
            ) as cur:
                rows = await cur.fetchall()
        return [row[0] for row in rows]

    # ─── Price Alerts ────────────────────────────────────────────────────────

    async def add_alert(
        self,
        user_id: str,
        channel_id: str,
        ticker: str,
        target_price: float,
        direction: str,
    ) -> int:
        """direction: 'above' or 'below'. Returns new alert ID."""
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                """INSERT INTO price_alerts (user_id, channel_id, ticker, target_price, direction)
                   VALUES (?, ?, ?, ?, ?)""",
                (user_id, channel_id, ticker.upper(), target_price, direction),
            )
            await db.commit()
            return cur.lastrowid

    async def get_user_alerts(self, user_id: str) -> list[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT id, ticker, target_price, direction, created_at
                   FROM price_alerts WHERE user_id = ? AND triggered = 0
                   ORDER BY created_at DESC""",
                (user_id,),
            ) as cur:
                rows = await cur.fetchall()
        return [dict(row) for row in rows]

    async def get_active_alerts(self) -> list[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT id, user_id, channel_id, ticker, target_price, direction
                   FROM price_alerts WHERE triggered = 0"""
            ) as cur:
                rows = await cur.fetchall()
        return [dict(row) for row in rows]

    async def mark_alert_triggered(self, alert_id: int):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE price_alerts SET triggered = 1 WHERE id = ?", (alert_id,)
            )
            await db.commit()

    async def remove_alert(self, user_id: str, alert_id: int) -> bool:
        async with aiosqlite.connect(self.db_path) as db:
            result = await db.execute(
                "DELETE FROM price_alerts WHERE id = ? AND user_id = ?",
                (alert_id, user_id),
            )
            await db.commit()
        return result.rowcount > 0

    # ─── Seen Tweets ─────────────────────────────────────────────────────────

    async def has_seen_tweet(self, tweet_id: str) -> bool:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT 1 FROM seen_tweets WHERE tweet_id = ?", (tweet_id,)
            ) as cur:
                return await cur.fetchone() is not None

    async def mark_tweet_seen(self, tweet_id: str):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR IGNORE INTO seen_tweets (tweet_id) VALUES (?)", (tweet_id,)
            )
            await db.commit()

    async def bulk_mark_tweets_seen(self, tweet_ids: list[str]):
        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany(
                "INSERT OR IGNORE INTO seen_tweets (tweet_id) VALUES (?)",
                [(tid,) for tid in tweet_ids],
            )
            await db.commit()

    # ─── Twitter Channels ────────────────────────────────────────────────────

    async def enable_twitter_channel(self, channel_id: str, guild_id: str):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO twitter_channels (channel_id, guild_id, enabled) VALUES (?, ?, 1)
                   ON CONFLICT(channel_id) DO UPDATE SET enabled = 1""",
                (channel_id, guild_id),
            )
            await db.commit()

    async def disable_twitter_channel(self, channel_id: str):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE twitter_channels SET enabled = 0 WHERE channel_id = ?",
                (channel_id,),
            )
            await db.commit()

    async def get_twitter_channels(self) -> list[str]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT channel_id FROM twitter_channels WHERE enabled = 1"
            ) as cur:
                rows = await cur.fetchall()
        return [row[0] for row in rows]

    async def is_twitter_enabled(self, channel_id: str) -> bool:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT enabled FROM twitter_channels WHERE channel_id = ?",
                (channel_id,),
            ) as cur:
                row = await cur.fetchone()
        return row is not None and row[0] == 1

    # ─── Daily Report Channels ───────────────────────────────────────────────

    async def set_daily_report_channel(
        self, channel_id: str, guild_id: str, hour: int = 8
    ):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO daily_report_channels (channel_id, guild_id, enabled, report_hour)
                   VALUES (?, ?, 1, ?)
                   ON CONFLICT(channel_id) DO UPDATE SET enabled = 1, report_hour = ?""",
                (channel_id, guild_id, hour, hour),
            )
            await db.commit()

    async def remove_daily_report_channel(self, channel_id: str):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE daily_report_channels SET enabled = 0 WHERE channel_id = ?",
                (channel_id,),
            )
            await db.commit()

    async def get_daily_report_channels(self) -> list[str]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT channel_id FROM daily_report_channels WHERE enabled = 1"
            ) as cur:
                rows = await cur.fetchall()
        return [row[0] for row in rows]

    # ─── Seen News (hot news deduplication) ─────────────────────────────────

    async def has_seen_news(self, url: str) -> bool:
        """Return True if this article URL has already been posted."""
        import hashlib
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT 1 FROM seen_news WHERE url_hash = ?", (url_hash,)
            ) as cur:
                return await cur.fetchone() is not None

    async def mark_news_seen(self, url: str, headline: str = ""):
        """Mark an article URL as seen to prevent reposting."""
        import hashlib
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR IGNORE INTO seen_news (url_hash, url, headline) VALUES (?, ?, ?)",
                (url_hash, url, headline),
            )
            await db.commit()

    # ─── Premium Users ───────────────────────────────────────────────────────

    async def is_premium_user(self, user_id: str) -> bool:
        """True if user has an active premium grant (not expired)."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """SELECT 1 FROM premium_users
                   WHERE user_id = ?
                   AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)""",
                (user_id,),
            ) as cur:
                return await cur.fetchone() is not None

    async def grant_premium(self, user_id: str, granted_by: str, note: str = "") -> bool:
        """Grant premium to a user. Returns False if already granted."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """INSERT INTO premium_users (user_id, granted_by, note) VALUES (?, ?, ?)
                       ON CONFLICT(user_id) DO UPDATE SET granted_by=?, granted_at=CURRENT_TIMESTAMP, expires_at=NULL, note=?""",
                    (user_id, granted_by, note, granted_by, note),
                )
                await db.commit()
            return True
        except Exception as e:
            logger.error(f"grant_premium({user_id}): {e}")
            return False

    async def revoke_premium(self, user_id: str) -> bool:
        """Revoke premium from a user."""
        async with aiosqlite.connect(self.db_path) as db:
            result = await db.execute(
                "DELETE FROM premium_users WHERE user_id = ?", (user_id,)
            )
            await db.commit()
        return result.rowcount > 0

    async def get_premium_users(self) -> list[dict]:
        """List all users with active premium grants."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT user_id, granted_by, granted_at, expires_at, note
                   FROM premium_users
                   WHERE expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP
                   ORDER BY granted_at DESC"""
            ) as cur:
                rows = await cur.fetchall()
        return [dict(row) for row in rows]

    # ─── Bot Config (key-value store) ───────────────────────────────────────

    async def get_config(self, key: str) -> str | None:
        """Retrieve a config value by key."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT value FROM bot_config WHERE key = ?", (key,)
            ) as cur:
                row = await cur.fetchone()
        return row[0] if row else None

    async def set_config(self, key: str, value: str):
        """Upsert a config value."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO bot_config (key, value) VALUES (?, ?)
                   ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = CURRENT_TIMESTAMP""",
                (key, value, value),
            )
            await db.commit()

    # ─── AI Prediction Tracking ──────────────────────────────────────────────

    async def record_prediction(
        self,
        ticker: str,
        signal: str,
        score: int,
        price: float,
        market_regime: str = "",
        source: str = "analysis",
    ) -> int:
        """Store an AI prediction for later accuracy evaluation. Returns row ID."""
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                """INSERT INTO ai_predictions (ticker, signal, score, price_at_call, market_regime, source)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (ticker.upper(), signal, score, price, market_regime, source),
            )
            await db.commit()
            return cur.lastrowid

    async def get_unevaluated_predictions(self, min_age_hours: int = 120) -> list[dict]:
        """Return predictions older than min_age_hours that haven't been evaluated yet."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT id, ticker, signal, score, price_at_call, called_at
                   FROM ai_predictions
                   WHERE correct IS NULL
                   AND called_at <= datetime('now', ? || ' hours')
                   ORDER BY called_at ASC
                   LIMIT 100""",
                (f"-{min_age_hours}",),
            ) as cur:
                rows = await cur.fetchall()
        return [dict(row) for row in rows]

    async def record_prediction_outcome(
        self,
        prediction_id: int,
        price_at_eval: float,
        outcome: str,
        correct: bool,
    ):
        """Mark a prediction as evaluated with its actual outcome."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """UPDATE ai_predictions
                   SET evaluated_at = CURRENT_TIMESTAMP,
                       price_at_eval = ?,
                       outcome = ?,
                       correct = ?
                   WHERE id = ?""",
                (price_at_eval, outcome, 1 if correct else 0, prediction_id),
            )
            await db.commit()

    async def get_prediction_accuracy(self, days: int = 30) -> dict:
        """
        Return accuracy stats for predictions made in the last N days.
        Returns dict with overall rate, per-signal rates, and recent sample.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            async with db.execute(
                """SELECT signal, COUNT(*) as total,
                          SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as hits
                   FROM ai_predictions
                   WHERE correct IS NOT NULL
                   AND called_at >= datetime('now', ? || ' days')
                   GROUP BY signal""",
                (f"-{days}",),
            ) as cur:
                rows = await cur.fetchall()

        stats: dict[str, dict] = {}
        total_all = hits_all = 0
        for row in rows:
            sig = row["signal"]
            stats[sig] = {"total": row["total"], "hits": row["hits"],
                          "rate": round(row["hits"] / row["total"] * 100, 1) if row["total"] else 0}
            total_all += row["total"]
            hits_all  += row["hits"]

        return {
            "per_signal": stats,
            "overall_total": total_all,
            "overall_hits": hits_all,
            "overall_rate": round(hits_all / total_all * 100, 1) if total_all else None,
            "days": days,
        }

    async def bulk_mark_news_seen(self, articles: list[dict]):
        """Seed multiple articles as seen (used at startup to prevent spam)."""
        import hashlib
        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany(
                "INSERT OR IGNORE INTO seen_news (url_hash, url, headline) VALUES (?, ?, ?)",
                [
                    (hashlib.sha256(a["url"].encode()).hexdigest()[:16], a["url"], a.get("title", ""))
                    for a in articles if a.get("url")
                ],
            )
            await db.commit()
