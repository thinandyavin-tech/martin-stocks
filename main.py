"""
Martin's Stocks — Discord AI Stock Analysis Bot
Entry point. Loads all cogs, initializes DB, starts APScheduler.
"""

import asyncio
import logging
import os

import discord
from discord.ext import commands
from dotenv import load_dotenv

from db.database import Database
from scheduler.tasks import setup_scheduler, seed_twitter_on_startup

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Quiet noisy libraries
logging.getLogger("yfinance").setLevel(logging.WARNING)
logging.getLogger("peewee").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.INFO)

logger = logging.getLogger(__name__)

COGS = [
    "cogs.stock_commands",
    "cogs.market_commands",
    "cogs.personal_tools",
    "cogs.twitter_cog",
    "cogs.daily_report",
    "cogs.extras",
    "cogs.admin_cog",
    "cogs.thai_stocks",
    "cogs.pro_commands",
]


class MartinStocksBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(
            command_prefix="!ms ",
            intents=intents,
            help_command=None,
        )
        self.db = Database()
        self.scheduler = None
        self.payment_link: str = os.getenv("PAYMENT_LINK", "")

    async def setup_hook(self):
        # Initialize database
        await self.db.initialize()
        logger.info("Database initialized.")

        # Load all cogs
        for cog_path in COGS:
            try:
                await self.load_extension(cog_path)
                logger.info(f"  ✓ Loaded cog: {cog_path}")
            except Exception as e:
                logger.error(f"  ✗ Failed to load cog {cog_path}: {e}", exc_info=True)

        # Load payment link from DB (overrides env var if admin has set one)
        db_link = await self.db.get_config("payment_link")
        if db_link:
            self.payment_link = db_link

        # Sync slash commands globally
        # For faster dev iteration, replace with:
        # await self.tree.sync(guild=discord.Object(id=YOUR_GUILD_ID))
        try:
            synced = await self.tree.sync()
            logger.info(f"Synced {len(synced)} slash command(s) globally.")
        except Exception as e:
            logger.error(f"Failed to sync slash commands: {e}")

        # Seed Twitter seen tweets to prevent startup spam
        await seed_twitter_on_startup(self)

        # Start APScheduler
        self.scheduler = setup_scheduler(self)
        self.scheduler.start()
        logger.info("APScheduler started.")

    async def on_ready(self):
        logger.info("=" * 50)
        logger.info(f"Bot ready: {self.user} (ID: {self.user.id})")
        logger.info(f"Connected to {len(self.guilds)} guild(s)")
        logger.info("=" * 50)

        # Guild-specific sync for instant slash command availability
        channel_id = os.getenv("DISCORD_CHANNEL_ID")
        if channel_id:
            channel = self.get_channel(int(channel_id))
            if channel and hasattr(channel, "guild"):
                guild = channel.guild
                try:
                    synced = await self.tree.sync(guild=guild)
                    logger.info(f"Guild sync: {len(synced)} command(s) → {guild.name} (instant)")
                except Exception as e:
                    logger.error(f"Guild sync failed: {e}")

        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="the markets 📈",
            )
        )

    async def on_app_command_error(
        self,
        interaction: discord.Interaction,
        error: discord.app_commands.AppCommandError,
    ):
        logger.error(f"App command error in /{interaction.command.name}: {error}", exc_info=True)
        msg = f"❌ An error occurred: {str(error)}"

        if isinstance(error, discord.app_commands.CommandOnCooldown):
            msg = f"⏳ Command on cooldown. Try again in {error.retry_after:.1f}s."
        elif isinstance(error, discord.app_commands.MissingPermissions):
            msg = "🔒 You don't have permission to use this command."

        try:
            if not interaction.response.is_done():
                await interaction.response.send_message(msg, ephemeral=True)
            else:
                await interaction.followup.send(msg, ephemeral=True)
        except Exception:
            pass

    async def close(self):
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            logger.info("Scheduler shut down.")
        await super().close()


def main():
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        raise ValueError(
            "DISCORD_TOKEN not found. "
            "Copy .env.example to .env and fill in your bot token."
        )

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        logger.warning("GROQ_API_KEY not set — AI features will be disabled.")

    bot = MartinStocksBot()
    bot.run(token, log_handler=None)  # We handle logging ourselves


if __name__ == "__main__":
    main()
