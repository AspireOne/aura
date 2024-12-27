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

class SystemPromptManager:
    def __init__(self):
        self.file_path = DATA_DIR / "system_prompt.txt"
        self.default_prompt = "You are a skibidi rizzler who is gooning in ohio and is talking in gen-z slang. You are talking with your friend in a discord chat. You shall not break character!"
        self.current_prompt = self.load_prompt()

    def load_prompt(self) -> str:
        try:
            return self.file_path.read_text().strip()
        except FileNotFoundError:
            self.save_prompt(self.default_prompt)
            return self.default_prompt

    def save_prompt(self, prompt: str):
        self.file_path.write_text(prompt)
        self.current_prompt = prompt

class AICompanion(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.dm_messages = True
        
        super().__init__(intents=intents)
        
        self.tree = discord.app_commands.CommandTree(self)
        self.openai_client = AsyncOpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )
        self.conversation_manager = ConversationManager()
        self.rate_limiter = RateLimiter()
        self.prompt_manager = SystemPromptManager()

    async def setup_hook(self):
        await self.tree.sync()

    async def on_ready(self):
        logging.info(f'Logged in as {self.user.name}')
        await self.change_presence(activity=discord.Activity(
            type=discord.ActivityType.listening,
            name="DMs | /prompt to customize"
        ))

    async def process_message(self, message_content: str, conversation_history: list) -> str:
        try:
            messages = [{"role": "system", "content": self.prompt_manager.current_prompt}] + conversation_history
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
                presence_penalty=0.6
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
                    self.conversation_manager.get_conversation(user_id)
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
        self.openai_client = AsyncOpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )
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
                    
                self.prompt_manager.save_prompt(new_prompt)
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
                    f"Current system prompt: {self.prompt_manager.current_prompt}\n"
                )
                
                if stats['last_interaction']:
                    last_time = datetime.fromisoformat(stats['last_interaction'])
                    info_message += f"Last interaction: {last_time.strftime('%Y-%m-%d %H:%M:%S')}"
                
                await interaction.response.send_message(info_message)
            except Exception as e:
                logging.error(f"Error showing info: {str(e)}")
                await interaction.response.send_message("An error occurred while getting information. Please try again.", ephemeral=True)

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
