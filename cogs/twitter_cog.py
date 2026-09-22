"""
Twitter monitoring cog.
/twitter on|off|status — enables/disables @ShayBoloor tweet monitoring per channel.
"""

import logging
import discord
from discord import app_commands
from discord.ext import commands

logger = logging.getLogger(__name__)


class TwitterCog(commands.Cog, name="Twitter"):
    def __init__(self, bot):
        self.bot = bot

    twitter_group = app_commands.Group(name="twitter", description="Monitor @ShayBoloor tweets in this channel")

    @twitter_group.command(name="on", description="Start posting @ShayBoloor tweets in this channel")
    async def twitter_on(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)

        if not interaction.guild:
            await interaction.followup.send(
                embed=discord.Embed(
                    description="❌ This command must be used in a server channel.",
                    color=discord.Color.red(),
                ),
                ephemeral=True,
            )
            return

        await self.bot.db.enable_twitter_channel(
            str(interaction.channel_id),
            str(interaction.guild_id),
        )

        embed = discord.Embed(
            title="🐦 Twitter Monitoring Enabled",
            description=(
                f"New tweets from **@ShayBoloor** will now be posted in <#{interaction.channel_id}>.\n\n"
                "• Polling every **5 minutes** via free Nitter RSS\n"
                "• Existing tweets will **not** be re-posted (seeded on startup)\n"
                "• Use `/twitter off` to disable"
            ),
            color=discord.Color.from_str("#1da1f2"),
        )
        await interaction.followup.send(embed=embed, ephemeral=True)

    @twitter_group.command(name="off", description="Stop posting @ShayBoloor tweets in this channel")
    async def twitter_off(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)

        await self.bot.db.disable_twitter_channel(str(interaction.channel_id))

        embed = discord.Embed(
            description=f"🔇 Twitter monitoring disabled in <#{interaction.channel_id}>.",
            color=discord.Color.orange(),
        )
        await interaction.followup.send(embed=embed, ephemeral=True)

    @twitter_group.command(name="status", description="Check Twitter monitoring status for this channel")
    async def twitter_status(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)

        enabled = await self.bot.db.is_twitter_enabled(str(interaction.channel_id))
        all_channels = await self.bot.db.get_twitter_channels()

        if enabled:
            description = f"✅ Twitter monitoring is **ON** in this channel.\n\n"
        else:
            description = f"❌ Twitter monitoring is **OFF** in this channel.\n\nUse `/twitter on` to enable it.\n\n"

        description += f"📡 Total enabled channels in this server: **{len([c for c in all_channels])}**"

        embed = discord.Embed(
            title="🐦 Twitter Monitoring Status",
            description=description,
            color=discord.Color.from_str("#1da1f2") if enabled else discord.Color.red(),
        )
        await interaction.followup.send(embed=embed, ephemeral=True)


async def setup(bot):
    await bot.add_cog(TwitterCog(bot))
