from typing import Union
import discord
from discord.ext import commands

def get_conversation_id(context: Union[commands.Context, discord.Message]) -> str:
    """Get a consistent conversation ID from a command context or a message."""
    if isinstance(context, commands.Context):
        channel = context.channel
    else:
        channel = context.channel
    
    author_id = context.author.id

    if isinstance(channel, discord.DMChannel):
        return str(author_id)
    else:
        return f"channel_{channel.id}"