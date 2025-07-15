import asyncio
import logging
import sys
import os
import discord
from discord.ext import commands

from core.config import (
    DATA_DIR,
    DEFAULT_SYSTEM_PROMPT,
    DISCORD_TOKEN,
    OPENROUTER_API_KEY,
)
from core.services import AIService
from core.image_service import ImageService
from utils.persistence import (
    ConversationManager,
    RateLimiter,
    SettingsManager,
)
from cogs.commands import CommandsCog
from cogs.events import EventsCog

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(DATA_DIR / "bot.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

# --- Bot Initialization ---
intents = discord.Intents.default()
intents.message_content = True
intents.dm_messages = True

bot = commands.Bot(command_prefix="!", intents=intents)
bot.remove_command('help')

# --- Main Application ---
async def main():
    if not DISCORD_TOKEN:
        logging.error("FATAL: DISCORD_TOKEN not found in environment.")
        return

    # --- Dependency Injection Setup ---
    # Initialize all the managers and services here
    conversation_manager = ConversationManager()
    rate_limiter = RateLimiter()

    # Determine default provider and model
    default_provider = "openrouter" if OPENROUTER_API_KEY else "openai"
    default_model = "anthropic/claude-3.5-sonnet" if default_provider == "openrouter" else "gpt-4o"

    model_manager = SettingsManager(setting_name="models", default_value=default_model)
    provider_manager = SettingsManager(setting_name="providers", default_value=default_provider)
    prompt_manager = SettingsManager(setting_name="prompts", default_value=DEFAULT_SYSTEM_PROMPT)
    
    ai_service = AIService(
        model_manager=model_manager,
        provider_manager=provider_manager
    )
    image_service = ImageService()

    # --- Load Cogs ---
    # Manually add cogs and pass dependencies
    await bot.add_cog(
        CommandsCog(
            bot,
            conversation_manager,
            model_manager,
            provider_manager,
            prompt_manager,
        )
    )
    await bot.add_cog(
        EventsCog(
            bot,
            conversation_manager,
            rate_limiter,
            ai_service,
            prompt_manager,
            image_service,
        )
    )

    async with bot:
        await bot.start(DISCORD_TOKEN)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.error(f"Fatal error in main execution: {e}")