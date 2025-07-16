import logging
from datetime import datetime
import discord
from discord.ext import commands

from core.config import HELP_TEXT, OPENAI_API_KEY, OPENROUTER_API_KEY
from utils.persistence import ConversationManager, SettingsManager
from utils.formatting import system_message
from utils.context import get_conversation_id


class CommandsCog(commands.Cog):
    def __init__(self, bot: commands.Bot, conversation_manager: ConversationManager,
                 model_manager: SettingsManager, provider_manager: SettingsManager,
                 prompt_manager: SettingsManager):
        self.bot = bot
        self.conversation_manager = conversation_manager
        self.model_manager = model_manager
        self.provider_manager = provider_manager
        self.prompt_manager = prompt_manager

    @commands.command(name='prompt')
    async def set_prompt(self, ctx: commands.Context, *, new_prompt: str):
        """Set a new system prompt."""
        if len(new_prompt) > 100000:
            await ctx.send(system_message("Prompt is too long."))
            return
        if not new_prompt.strip():
            await ctx.send(system_message("Prompt cannot be empty!"))
            return
        
        conversation_id = get_conversation_id(ctx)
        self.prompt_manager.set_setting(conversation_id, new_prompt)
        self.conversation_manager.delete_conversation(conversation_id)
        await ctx.send(system_message("System prompt updated!"))

    @commands.command(name='clear')
    async def clear_history(self, ctx: commands.Context):
        """Clear conversation history."""
        conversation_id = get_conversation_id(ctx)
        self.conversation_manager.delete_conversation(conversation_id)
        is_dm = isinstance(ctx.channel, discord.DMChannel)
        await ctx.send(system_message(f"{'Personal' if is_dm else 'Channel'} conversation history cleared!"))
        logging.info(f"Cleared conversation history for ID: {conversation_id}")

    @commands.command(name='info')
    async def show_info(self, ctx: commands.Context):
        """Show conversation statistics."""
        conversation_id = get_conversation_id(ctx)
        stats = self.conversation_manager.get_stats(conversation_id)
        is_dm = isinstance(ctx.channel, discord.DMChannel)
        approx_tokens = round(stats['total_characters'] / 4)
        
        info_message = (
            f"**{'Personal' if is_dm else 'Channel'} Conversation Statistics**\n"
            f"Messages in history: {stats['message_count']}\n"
            f"Total characters: {stats['total_characters']}/{stats['max_characters']} (~{approx_tokens} tokens)\n"
            f"System prompt: {self.prompt_manager.get_setting(conversation_id)}\n"
            f"Provider: {self.provider_manager.get_setting(conversation_id)}\n"
            f"Model: {self.model_manager.get_setting(conversation_id)}\n"
        )
        if stats['last_interaction']:
            last_time = datetime.fromisoformat(stats['last_interaction'])
            info_message += f"Last interaction: {last_time.strftime('%Y-%m-%d %H:%M:%S')}"
        
        await ctx.send(info_message)

    @commands.command(name='provider')
    async def set_provider(self, ctx: commands.Context, provider: str):
        """Set AI provider (openai/openrouter)."""
        provider = provider.lower()
        if provider not in ["openai", "openrouter"]:
            await ctx.send(system_message("Invalid provider! Use 'openai' or 'openrouter'."))
            return
        if provider == "openai" and not OPENAI_API_KEY:
            await ctx.send(system_message("OPENAI_API_KEY not found in environment!"))
            return
        if provider == "openrouter" and not OPENROUTER_API_KEY:
            await ctx.send(system_message("OPENROUTER_API_KEY not found in environment!"))
            return
            
        conversation_id = get_conversation_id(ctx)
        self.provider_manager.set_setting(conversation_id, provider)
        is_dm = isinstance(ctx.channel, discord.DMChannel)
        await ctx.send(system_message(f"{'Personal' if is_dm else 'Channel'} AI provider updated to {provider}!"))

    @commands.command(name='model')
    async def set_model(self, ctx: commands.Context, model: str):
        """Set AI model."""
        conversation_id = get_conversation_id(ctx)
        self.model_manager.set_setting(conversation_id, model)
        is_dm = isinstance(ctx.channel, discord.DMChannel)
        await ctx.send(system_message(f"{'Personal' if is_dm else 'Channel'} AI model updated to {model}!"))

    @commands.command(name='setlimit')
    async def set_limit(self, ctx: commands.Context, max_chars: int):
        """Set max history characters."""
        if max_chars < 1000:
            await ctx.send(system_message("Maximum characters must be at least 1000!"))
            return
        if max_chars > 150000:
            await ctx.send(system_message("Maximum characters cannot exceed 150,000!"))
            return
        self.conversation_manager.set_max_chars(max_chars)
        await ctx.send(system_message(f"Maximum conversation history updated to {max_chars:,} characters!"))

    @set_limit.error
    async def set_limit_error(self, ctx, error):
        if isinstance(error, commands.BadArgument):
            await ctx.send(system_message("Please provide a valid number!"))

    @commands.command(name='clear-prompt')
    async def clear_prompt(self, ctx: commands.Context):
        """Reset system prompt to default."""
        conversation_id = get_conversation_id(ctx)
        self.prompt_manager.set_setting(conversation_id, self.prompt_manager.default_value)
        is_dm = isinstance(ctx.channel, discord.DMChannel)
        await ctx.send(system_message(f"{'Personal' if is_dm else 'Channel'} system prompt reset to default!"))

    @commands.command(name='help')
    async def help_command(self, ctx: commands.Context):
        """Show the help message."""
        await ctx.send(system_message(HELP_TEXT))
