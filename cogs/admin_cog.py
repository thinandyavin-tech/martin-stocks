"""
Admin commands — only usable by admins (marwindsss + ADMIN_USER_IDS env var).

/admin status @user   — show user info (ID, join date, username)
/admin broadcast msg  — send a message embed to all daily report channels
/admin stats          — show bot uptime, guild count, channel count, cog count
"""

from __future__ import annotations

import datetime
import logging
import os

import discord
from discord import app_commands
from discord.ext import commands

from utils.permissions import is_admin, ADMIN_USERNAMES

logger = logging.getLogger(__name__)

# Track startup time so /admin stats can report uptime
_BOT_START_TIME = datetime.datetime.now(datetime.timezone.utc)


def _admin_only(interaction: discord.Interaction) -> bool:
    """Check decorator: raises CheckFailure if caller is not an admin."""
    if not is_admin(interaction):
        raise app_commands.CheckFailure("Admin only.")
    return True


class AdminCog(commands.Cog, name="Admin"):
    def __init__(self, bot):
        self.bot = bot

    admin_group = app_commands.Group(
        name="admin",
        description="Bot administration (admin only)",
    )

    # ─── /admin status @user ──────────────────────────────────────────────────

    @admin_group.command(name="status", description="Show user info: ID, join date, username")
    @app_commands.describe(user="User to inspect")
    @app_commands.check(_admin_only)
    async def admin_status(self, interaction: discord.Interaction, user: discord.Member):
        """Display basic profile info for any server member."""
        await interaction.response.defer(ephemeral=True)

        env_ids = {uid.strip() for uid in os.getenv("ADMIN_USER_IDS", "").split(",") if uid.strip()}
        user_is_admin = user.name in ADMIN_USERNAMES or str(user.id) in env_ids

        joined_server = user.joined_at.strftime("%Y-%m-%d %H:%M UTC") if user.joined_at else "Unknown"
        joined_discord = user.created_at.strftime("%Y-%m-%d %H:%M UTC") if user.created_at else "Unknown"
        roles = [r.name for r in user.roles if r.name != "@everyone"]

        tier = "👑 Admin" if user_is_admin else "Member"
        color = discord.Color.gold() if user_is_admin else discord.Color.from_str("#58a6ff")

        embed = discord.Embed(
            title=f"👤 User Info — {user.display_name}",
            color=color,
        )
        embed.set_thumbnail(url=user.display_avatar.url)
        embed.add_field(name="Username", value=f"{user.name}", inline=True)
        embed.add_field(name="User ID", value=f"`{user.id}`", inline=True)
        embed.add_field(name="Bot Admin", value=tier, inline=True)
        embed.add_field(name="Joined Server", value=joined_server, inline=True)
        embed.add_field(name="Account Created", value=joined_discord, inline=True)
        embed.add_field(name="Bot", value="Yes" if user.bot else "No", inline=True)
        if roles:
            embed.add_field(name=f"Roles ({len(roles)})", value=", ".join(roles[:10]), inline=False)

        await interaction.followup.send(embed=embed, ephemeral=True)

    # ─── /admin broadcast <message> ───────────────────────────────────────────

    @admin_group.command(name="broadcast", description="Send a message embed to all daily report channels")
    @app_commands.describe(message="The message to broadcast")
    @app_commands.check(_admin_only)
    async def admin_broadcast(self, interaction: discord.Interaction, message: str):
        """Broadcast a plain-text message embed to every configured daily report channel."""
        await interaction.response.defer(ephemeral=True)

        channel_ids = await self.bot.db.get_daily_report_channels()
        if not channel_ids:
            await interaction.followup.send(
                embed=discord.Embed(
                    description="No daily report channels configured — nothing to broadcast to.",
                    color=discord.Color.orange(),
                ),
                ephemeral=True,
            )
            return

        broadcast_embed = discord.Embed(
            title="📢 Broadcast",
            description=message,
            color=discord.Color.gold(),
        )
        broadcast_embed.set_footer(
            text=f"Sent by {interaction.user.display_name} • {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )

        sent, failed = 0, 0
        for ch_id in channel_ids:
            channel = self.bot.get_channel(int(ch_id))
            if channel is None:
                failed += 1
                continue
            try:
                await channel.send(embed=broadcast_embed)
                sent += 1
            except Exception as e:
                logger.warning(f"broadcast: could not send to channel {ch_id}: {e}")
                failed += 1

        summary = discord.Embed(
            title="✅ Broadcast Sent",
            description=(
                f"Delivered to **{sent}** channel(s)."
                + (f"\nFailed: **{failed}** channel(s) (bot missing access)." if failed else "")
            ),
            color=discord.Color.green() if sent else discord.Color.red(),
        )
        await interaction.followup.send(embed=summary, ephemeral=True)
        logger.info(f"Admin {interaction.user.name} broadcast to {sent} channel(s): {message[:80]}")

    # ─── /admin stats ─────────────────────────────────────────────────────────

    @admin_group.command(name="stats", description="Show bot uptime, guild count, channel count, cog count")
    @app_commands.check(_admin_only)
    async def admin_stats(self, interaction: discord.Interaction):
        """Display runtime statistics for the bot."""
        await interaction.response.defer(ephemeral=True)

        now = datetime.datetime.now(datetime.timezone.utc)
        delta = now - _BOT_START_TIME
        hours, remainder = divmod(int(delta.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{hours}h {minutes}m {seconds}s"

        guild_count = len(self.bot.guilds)
        channel_count = sum(1 for _ in self.bot.get_all_channels())
        cog_count = len(self.bot.cogs)
        cmd_count = len(self.bot.tree.get_commands())

        embed = discord.Embed(
            title="🤖 Bot Stats",
            color=discord.Color.gold(),
        )
        embed.add_field(name="Uptime", value=uptime_str, inline=True)
        embed.add_field(name="Guilds", value=str(guild_count), inline=True)
        embed.add_field(name="Channels", value=str(channel_count), inline=True)
        embed.add_field(name="Cogs Loaded", value=str(cog_count), inline=True)
        embed.add_field(name="Slash Commands", value=str(cmd_count), inline=True)
        embed.add_field(name="Latency", value=f"{round(self.bot.latency * 1000)}ms", inline=True)
        embed.set_footer(text=f"As of {now.strftime('%Y-%m-%d %H:%M UTC')}")

        await interaction.followup.send(embed=embed, ephemeral=True)

    # ─── /admin panel ─────────────────────────────────────────────────────────

    @admin_group.command(name="panel", description="Full admin control panel — health, DB stats, and scheduler status")
    @app_commands.check(_admin_only)
    async def admin_panel(self, interaction: discord.Interaction):
        """Comprehensive admin dashboard with bot metrics, DB stats, and scheduler next-run times."""
        await interaction.response.defer(ephemeral=True)

        now = datetime.datetime.now(datetime.timezone.utc)
        delta = now - _BOT_START_TIME
        hours, rem = divmod(int(delta.total_seconds()), 3600)
        minutes, seconds = divmod(rem, 60)
        uptime_str = f"{hours}h {minutes}m {seconds}s"

        try:
            report_channels = await self.bot.db.get_daily_report_channels()
            active_alerts = await self.bot.db.get_active_alerts()
            alert_count = len(active_alerts) if active_alerts else 0
        except Exception:
            report_channels = []
            alert_count = 0

        scheduler_lines = []
        if self.bot.scheduler:
            for job in self.bot.scheduler.get_jobs():
                nrt = job.next_run_time
                if nrt:
                    diff = max(0, int((nrt - now).total_seconds()))
                    h, r = divmod(diff, 3600)
                    m, s = divmod(r, 60)
                    scheduler_lines.append(f"• **{job.name}** — in {h}h {m}m {s}s")
                else:
                    scheduler_lines.append(f"• **{job.name}** — not scheduled")

        embed = discord.Embed(
            title="👑 Admin Control Panel",
            color=discord.Color.gold(),
        )

        embed.add_field(
            name="🤖 Bot Health",
            value=(
                f"Uptime:    **{uptime_str}**\n"
                f"Latency:   **{round(self.bot.latency * 1000)}ms**\n"
                f"Guilds:    **{len(self.bot.guilds)}**\n"
                f"Channels:  **{sum(1 for _ in self.bot.get_all_channels())}**\n"
                f"Cogs:      **{len(self.bot.cogs)}**\n"
                f"Commands:  **{len(self.bot.tree.get_commands())}**"
            ),
            inline=True,
        )

        embed.add_field(
            name="🗄️ Database",
            value=(
                f"Active Alerts:   **{alert_count}**\n"
                f"Report Channels: **{len(report_channels)}**"
            ),
            inline=True,
        )

        if scheduler_lines:
            embed.add_field(
                name="⏰ Scheduler — Next Run Times",
                value="\n".join(scheduler_lines[:8]),
                inline=False,
            )

        embed.add_field(
            name="⚙️ Environment",
            value=(
                f"GROQ:      {'✅ Set' if os.getenv('GROQ_API_KEY') else '❌ Missing'}\n"
                f"FINNHUB:   {'✅ Set' if os.getenv('FINNHUB_API_KEY') else '⚪ Not set'}\n"
                f"CHANNEL:   {'✅ Set' if os.getenv('DISCORD_CHANNEL_ID') else '❌ Missing'}\n"
                f"TIMEZONE:  **{os.getenv('REPORT_TIMEZONE', 'UTC')}**\n"
                f"REPORT:    **{os.getenv('REPORT_TIME', '08:00')}**"
            ),
            inline=False,
        )

        embed.set_footer(text=f"Admin Panel • {now.strftime('%Y-%m-%d %H:%M UTC')}")
        await interaction.followup.send(embed=embed, ephemeral=True)

    # ─── Error handler ────────────────────────────────────────────────────────

    async def cog_app_command_error(
        self, interaction: discord.Interaction, error: app_commands.AppCommandError
    ) -> None:
        if isinstance(error, app_commands.CheckFailure):
            try:
                await interaction.response.send_message(
                    embed=discord.Embed(
                        description="🔒 This command is for admins only.",
                        color=discord.Color.red(),
                    ),
                    ephemeral=True,
                )
            except discord.InteractionResponded:
                pass


async def setup(bot):
    await bot.add_cog(AdminCog(bot))
