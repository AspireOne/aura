import os
import asyncio
import discord
from discord.ext import commands
from openai import AsyncOpenAI
from dotenv import load_dotenv
from datetime import datetime, timedelta
import logging
from collections import deque
import json
from pathlib import Path

# Base directory for all persistent data
DATA_DIR = Path(".data")
DATA_DIR.mkdir(exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(DATA_DIR / 'bot.log'),
        logging.StreamHandler()
    ]
)

load_dotenv()

class ConversationManager:
    def __init__(self, expiry_hours=24, max_chars=120000):
        self.conversations = {}
        self.max_chars = max_chars
        self.expiry_hours = expiry_hours
        self.last_interaction = {}
        self.storage_path = DATA_DIR / "conversations"
        self.storage_path.mkdir(exist_ok=True)
        
        # Load existing conversations
        self.load_conversations()

    def load_conversations(self):
        for file in self.storage_path.glob("*.json"):
            user_id = file.stem
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    self.conversations[user_id] = deque(data['messages'])
                    self.last_interaction[user_id] = datetime.fromisoformat(data['last_interaction'])
                    
                    # Clean up expired conversations
                    if datetime.now() - self.last_interaction[user_id] > timedelta(hours=self.expiry_hours):
                        self.delete_conversation(user_id)
                        
            except Exception as e:
                logging.error(f"Error loading conversation for {user_id}: {e}")

    def save_conversation(self, user_id: str):
        file_path = self.storage_path / f"{user_id}.json"
        try:
            data = {
                'messages': list(self.conversations[user_id]),
                'last_interaction': self.last_interaction[user_id].isoformat()
            }
            with open(file_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logging.error(f"Error saving conversation for {user_id}: {e}")

    def delete_conversation(self, user_id: str):
        file_path = self.storage_path / f"{user_id}.json"
        try:
            file_path.unlink(missing_ok=True)
            self.conversations.pop(user_id, None)
            self.last_interaction.pop(user_id, None)
        except Exception as e:
            logging.error(f"Error deleting conversation for {user_id}: {e}")

    def add_message(self, user_id: str, message: dict):
        current_time = datetime.now()
        
        # Initialize or clear expired conversation
        if (user_id not in self.last_interaction or 
            current_time - self.last_interaction[user_id] > timedelta(hours=self.expiry_hours)):
            self.conversations[user_id] = deque()
        
        # Add new message
        self.conversations[user_id].append(message)
        self.last_interaction[user_id] = current_time
        
        # Check total character count and trim if necessary
        total_chars = sum(len(msg["content"]) for msg in self.conversations[user_id])
        while total_chars > self.max_chars and len(self.conversations[user_id]) > 1:
            removed_msg = self.conversations[user_id].popleft()
            total_chars -= len(removed_msg["content"])
        
        # Save after each message
        self.save_conversation(user_id)

    def get_conversation(self, user_id: str) -> list:
        return list(self.conversations.get(user_id, []))

    def set_max_chars(self, new_max: int):
        """Update max_chars and adjust existing conversations"""
        self.max_chars = new_max
        # Trim existing conversations if they exceed the new limit
        for user_id in self.conversations:
            total_chars = sum(len(msg["content"]) for msg in self.conversations[user_id])
            while total_chars > new_max and len(self.conversations[user_id]) > 1:
                removed_msg = self.conversations[user_id].popleft()
                total_chars -= len(removed_msg["content"])
            self.save_conversation(user_id)
        
    def get_stats(self, user_id: str) -> dict:
        conversation = self.conversations.get(user_id, deque())
        total_chars = sum(len(msg["content"]) for msg in conversation)
        message_count = len(conversation)
        last_interaction = self.last_interaction.get(user_id)
        
        return {
            "total_characters": total_chars,
            "message_count": message_count,
            "max_characters": self.max_chars,
            "last_interaction": last_interaction.isoformat() if last_interaction else None
        }

class RateLimiter:
    def __init__(self, messages_per_minute=5):
        self.rate_limits = {}
        self.messages_per_minute = messages_per_minute

    def can_send(self, user_id: str) -> bool:
        current_time = datetime.now()
        if user_id not in self.rate_limits:
            self.rate_limits[user_id] = deque(maxlen=self.messages_per_minute)
            
        while (self.rate_limits[user_id] and 
               current_time - self.rate_limits[user_id][0] > timedelta(minutes=1)):
            self.rate_limits[user_id].popleft()
            
        return len(self.rate_limits[user_id]) < self.messages_per_minute

    def add_message(self, user_id: str):
        if user_id not in self.rate_limits:
            self.rate_limits[user_id] = deque(maxlen=self.messages_per_minute)
        self.rate_limits[user_id].append(datetime.now())

class ModelManager:
    def __init__(self):
        self.storage_path = DATA_DIR / "models"
        self.storage_path.mkdir(exist_ok=True)
        self.default_model = "gpt-4o"
        self.models = {}
        
    def get_model(self, user_id: str) -> str:
        if user_id not in self.models:
            self.models[user_id] = self.load_model(user_id)
        return self.models[user_id]
        
    def load_model(self, user_id: str) -> str:
        file_path = self.storage_path / f"{user_id}.txt"
        try:
            return file_path.read_text().strip()
        except FileNotFoundError:
            self.save_model(user_id, self.default_model)
            return self.default_model
            
    def save_model(self, user_id: str, model: str):
        file_path = self.storage_path / f"{user_id}.txt"
        file_path.write_text(model)
        self.models[user_id] = model

class ProviderManager:
    def __init__(self):
        self.storage_path = DATA_DIR / "providers"
        self.storage_path.mkdir(exist_ok=True)
        self.providers = {}
        
    def get_provider(self, user_id: str) -> str:
        if user_id not in self.providers:
            self.providers[user_id] = self.load_provider(user_id)
        return self.providers[user_id]
        
    def load_provider(self, user_id: str) -> str:
        file_path = self.storage_path / f"{user_id}.txt"
        try:
            provider = file_path.read_text().strip()
            return provider if provider in ["openai", "openrouter"] else "openai"
        except FileNotFoundError:
            self.save_provider(user_id, "openai")
            return "openai"
            
    def save_provider(self, user_id: str, provider: str):
        if provider not in ["openai", "openrouter"]:
            raise ValueError("Invalid provider")
        file_path = self.storage_path / f"{user_id}.txt"
        file_path.write_text(provider)
        self.providers[user_id] = provider
        
    def get_client_config(self, user_id: str) -> dict:
        provider = self.get_provider(user_id)
        if provider == "openai":
            if not os.getenv('OPENAI_API_KEY'):
                raise ValueError("OPENAI_API_KEY not found in environment")
            return {
                "api_key": os.getenv('OPENAI_API_KEY')
            }
        else:  # openrouter
            if not os.getenv('OPENROUTER_API_KEY'):
                raise ValueError("OPENROUTER_API_KEY not found in environment")
            return {
                "api_key": os.getenv('OPENROUTER_API_KEY'),
                "base_url": "https://openrouter.ai/api/v1"
            }

class SystemPromptManager:
    def __init__(self):
        self.storage_path = DATA_DIR / "prompts"
        self.storage_path.mkdir(exist_ok=True)
        self.default_prompt = "You are a cute girl chatting with your friend on discord. For context, you play VRChat together sometimes, both like anime, and both are kind."
        self.prompts = {}

    def get_prompt(self, user_id: str) -> str:
        if user_id not in self.prompts:
            self.prompts[user_id] = self.load_prompt(user_id)
        return self.prompts[user_id]

    def load_prompt(self, user_id: str) -> str:
        file_path = self.storage_path / f"{user_id}.txt"
        try:
            return file_path.read_text().strip()
        except FileNotFoundError:
            self.save_prompt(user_id, self.default_prompt)
            return self.default_prompt

    def save_prompt(self, user_id: str, prompt: str):
        file_path = self.storage_path / f"{user_id}.txt"
        file_path.write_text(prompt)
        self.prompts[user_id] = prompt

class AICompanion(discord.Client):
    async def setup_hook(self):
        await self.tree.sync()

    async def on_ready(self):
        logging.info(f'Logged in as {self.user.name}')
        await self.change_presence(activity=discord.Activity(
            type=discord.ActivityType.listening,
            name="DMs | /info (current settings), /prompt (customize system prompt)"
        ))

    async def process_message(self, message_content: str, conversation_history: list, user_id: str) -> str:
        try:
            # Get user-specific configuration
            config = self.provider_manager.get_client_config(user_id)
            client = AsyncOpenAI(**config)
            
            messages = [{"role": "system", "content": self.prompt_manager.get_prompt(user_id)}] + conversation_history
            
            response = await client.chat.completions.create(
                model=self.model_manager.get_model(user_id),
                messages=messages,
                max_tokens=4000,
                temperature=0.9,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logging.error(f"Error in AI processing: {str(e)}")
            raise

    async def on_message(self, message):
        if message.author.bot or not isinstance(message.channel, discord.DMChannel):
            return
        
        # Don't process slash commands here
        if message.content.startswith('/'):
            return
        
        user_id = str(message.author.id)
        
        # Check rate limit
        if not self.rate_limiter.can_send(user_id):
            await message.channel.send("Please wait a moment before sending another message.")
            return
            
        self.rate_limiter.add_message(user_id)

        # Add user message to conversation
        self.conversation_manager.add_message(
            user_id,
            {"role": "user", "content": message.content}
        )

        try:
            async with message.channel.typing():
                ai_response = await self.process_message(
                    message.content,
                    self.conversation_manager.get_conversation(user_id),
                    user_id
                )

            # Split long responses
            if len(ai_response) > 2000:
                chunks = [ai_response[i:i+1999] for i in range(0, len(ai_response), 1999)]
                for chunk in chunks:
                    await message.channel.send(chunk)
            else:
                await message.channel.send(ai_response)

            # Add AI response to conversation
            self.conversation_manager.add_message(
                user_id,
                {"role": "assistant", "content": ai_response}
            )

        except Exception as e:
            logging.error(f"Error processing message: {str(e)}")
            await message.channel.send("Sorry, I encountered an error. Please try again later.")

    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.dm_messages = True
        
        super().__init__(intents=intents)
        
        self.tree = discord.app_commands.CommandTree(self)
        self.model_manager = ModelManager()
        self.provider_manager = ProviderManager()
        try:
            # No need to initialize a default client since we create per-user clients
            self.openai_client = None
        except ValueError as e:
            logging.error(f"Error initializing AI client: {str(e)}")
            self.openai_client = None
        self.conversation_manager = ConversationManager()
        self.rate_limiter = RateLimiter()
        self.prompt_manager = SystemPromptManager()

        # Register slash commands
        @self.tree.command(name="prompt", description="Set a new system prompt for the AI")
        async def set_prompt(interaction: discord.Interaction, new_prompt: str):
            if not isinstance(interaction.channel, discord.DMChannel):
                await interaction.response.send_message("This command can only be used in DMs!", ephemeral=True)
                return

            try:
                if len(new_prompt) > 1000:
                    await interaction.response.send_message("Prompt is too long! Please keep it under 1000 characters.", ephemeral=True)
                    return
                    
                if len(new_prompt.strip()) == 0:
                    await interaction.response.send_message("Prompt cannot be empty!", ephemeral=True)
                    return
                    
                user_id = str(interaction.user.id)
                self.prompt_manager.save_prompt(user_id, new_prompt)
                await interaction.response.send_message("System prompt updated! Conversation will continue with the new prompt.")
            except Exception as e:
                logging.error(f"Error setting prompt: {str(e)}")
                await interaction.response.send_message("An error occurred while setting the prompt. Please try again.", ephemeral=True)

        @self.tree.command(name="clear", description="Clear your conversation history")
        async def clear_history(interaction: discord.Interaction):
            if not isinstance(interaction.channel, discord.DMChannel):
                await interaction.response.send_message("This command can only be used in DMs!", ephemeral=True)
                return
                
            try:
                user_id = str(interaction.user.id)
                self.conversation_manager.delete_conversation(user_id)
                await interaction.response.send_message("Conversation history cleared!")
            except Exception as e:
                logging.error(f"Error clearing history: {str(e)}")
                await interaction.response.send_message("An error occurred while clearing history. Please try again.", ephemeral=True)

        @self.tree.command(name="info", description="Show current conversation statistics")
        async def show_info(interaction: discord.Interaction):
            if not isinstance(interaction.channel, discord.DMChannel):
                await interaction.response.send_message("This command can only be used in DMs!", ephemeral=True)
                return
                
            try:
                user_id = str(interaction.user.id)
                stats = self.conversation_manager.get_stats(user_id)
                
                info_message = (
                    "**Conversation Statistics**\n"
                    f"Messages in history: {stats['message_count']}\n"
                    f"Total characters: {stats['total_characters']}/{stats['max_characters']}\n"
                    f"System prompt: {self.prompt_manager.get_prompt(user_id)}\n"
                    f"Provider: {self.provider_manager.get_provider(user_id)}\n"
                    f"Model: {self.model_manager.get_model(user_id)}\n"
                )
                
                if stats['last_interaction']:
                    last_time = datetime.fromisoformat(stats['last_interaction'])
                    info_message += f"Last interaction: {last_time.strftime('%Y-%m-%d %H:%M:%S')}"
                
                await interaction.response.send_message(info_message)
            except Exception as e:
                logging.error(f"Error showing info: {str(e)}")
                await interaction.response.send_message("An error occurred while getting information. Please try again.", ephemeral=True)

        @self.tree.command(name="provider", description="Set the AI provider (openai/openrouter)")
        async def set_provider(interaction: discord.Interaction, provider: str):
            if not isinstance(interaction.channel, discord.DMChannel):
                await interaction.response.send_message("This command can only be used in DMs!", ephemeral=True)
                return
                
            try:
                provider = provider.lower()
                if provider not in ["openai", "openrouter"]:
                    await interaction.response.send_message("Invalid provider! Use 'openai' or 'openrouter'.", ephemeral=True)
                    return
                
                # Check if API key exists
                if provider == "openai" and not os.getenv('OPENAI_API_KEY'):
                    await interaction.response.send_message("OPENAI_API_KEY not found in environment!", ephemeral=True)
                    return
                elif provider == "openrouter" and not os.getenv('OPENROUTER_API_KEY'):
                    await interaction.response.send_message("OPENROUTER_API_KEY not found in environment!", ephemeral=True)
                    return
                
                user_id = str(interaction.user.id)
                self.provider_manager.save_provider(user_id, provider)
                # Update client only for this user's messages
                await interaction.response.send_message(f"AI provider updated to {provider}!")
                
            except Exception as e:
                logging.error(f"Error setting provider: {str(e)}")
                await interaction.response.send_message("An error occurred while updating the provider. Please try again.", ephemeral=True)

        @self.tree.command(name="model", description="Set the AI model to use")
        async def set_model(interaction: discord.Interaction, model: str):
            if not isinstance(interaction.channel, discord.DMChannel):
                await interaction.response.send_message("This command can only be used in DMs!", ephemeral=True)
                return
                
            try:
                if len(model.strip()) == 0:
                    await interaction.response.send_message("Model name cannot be empty!", ephemeral=True)
                    return
                    
                user_id = str(interaction.user.id)
                self.model_manager.save_model(user_id, model)
                await interaction.response.send_message(f"AI model updated to {model}!")
                
            except Exception as e:
                logging.error(f"Error setting model: {str(e)}")
                await interaction.response.send_message("An error occurred while updating the model. Please try again.", ephemeral=True)

        @self.tree.command(name="setlimit", description="Set maximum number of characters to keep in history")
        async def set_limit(interaction: discord.Interaction, max_chars: int):
            if not isinstance(interaction.channel, discord.DMChannel):
                await interaction.response.send_message("This command can only be used in DMs!", ephemeral=True)
                return
                
            try:
                if max_chars < 1000:  # Minimum reasonable limit
                    await interaction.response.send_message("Maximum characters must be at least 1000!", ephemeral=True)
                    return
                    
                if max_chars > 150000:  # Reasonable upper limit
                    await interaction.response.send_message("Maximum characters cannot exceed 150,000!", ephemeral=True)
                    return
                    
                self.conversation_manager.set_max_chars(max_chars)
                await interaction.response.send_message(f"Maximum conversation history updated to {max_chars:,} characters!")
            except Exception as e:
                logging.error(f"Error setting character limit: {str(e)}")
                await interaction.response.send_message("An error occurred while updating the character limit. Please try again.", ephemeral=True)

if __name__ == "__main__":
    client = AICompanion()
    
    async def main():
        async with client:
            await client.start(os.getenv('DISCORD_TOKEN'))

    try:
        asyncio.run(main())
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
