"""
Permission helpers for Martin's Stocks.
No tiers — everyone gets full access.
marwindsss (and any ADMIN_USER_IDS env entries) are admins and get gold flair on their responses.
"""

import os
import discord

ADMIN_USERNAMES: set[str] = {"marwindsss"}


def is_admin(interaction: discord.Interaction) -> bool:
    env_ids = {uid.strip() for uid in os.getenv("ADMIN_USER_IDS", "").split(",") if uid.strip()}
    return (
        interaction.user.name in ADMIN_USERNAMES
        or str(interaction.user.id) in env_ids
    )


def apply_admin_flair(embed: discord.Embed, interaction: discord.Interaction) -> discord.Embed:
    """
    If the user is an admin, give their embed a gold color and a crown badge.
    Call this on the main embed before sending.
    """
    if not is_admin(interaction):
        return embed

    embed.color = discord.Color.gold()
    if embed.title and not embed.title.startswith("👑"):
        embed.title = f"👑 {embed.title}"
    existing = embed.footer.text if embed.footer and embed.footer.text else ""
    embed.set_footer(text=f"👑 Admin  •  {existing}" if existing else "👑 Admin")
    return embed
