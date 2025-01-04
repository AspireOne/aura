import os
import sys
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
import unicodedata

def strip_diacritics(text):
    """Remove diacritics from text, converting to ASCII equivalents"""
    return ''.join(c for c in unicodedata.normalize('NFKD', text)
                  if not unicodedata.combining(c))

# Base directory for all persistent data
DATA_DIR = Path(".data")
DATA_DIR.mkdir(exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(DATA_DIR / 'bot.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

load_dotenv()

class ConversationManager:
    def __init__(self, expiry_hours=3600, max_chars=120000):
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
        total_chars = 0
        for msg in self.conversations[user_id]:
            if isinstance(msg["content"], list):
                # Count text and images in mixed content
                for item in msg["content"]:
                    if item["type"] == "image_url":
                        total_chars += 3000  # Count each image as 3000 chars
                    else:
                        total_chars += len(item["text"])
            else:
                total_chars += len(msg["content"])
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
        total_chars = 0
        for msg in conversation:
            if isinstance(msg["content"], list):
                # Count text and images in mixed content
                for item in msg["content"]:
                    if item["type"] == "image_url":
                        total_chars += 3000  # Count each image as 3000 chars
                    else:
                        total_chars += len(item["text"])
            else:
                total_chars += len(msg["content"])
        message_count = len(conversation)
        last_interaction = self.last_interaction.get(user_id)
        
        return {
            "total_characters": total_chars,
            "message_count": message_count,
            "max_characters": self.max_chars,
            "last_interaction": last_interaction.isoformat() if last_interaction else None
        }

class RateLimiter:
    def __init__(self, messages_per_minute=7):
        self.rate_limits = {}
        self.messages_per_minute = messages_per_minute

    def can_send(self, conversation_id: str) -> bool:
        current_time = datetime.now()
        if conversation_id not in self.rate_limits:
            self.rate_limits[conversation_id] = deque(maxlen=self.messages_per_minute)
            
        while (self.rate_limits[conversation_id] and 
               current_time - self.rate_limits[conversation_id][0] > timedelta(minutes=1)):
            self.rate_limits[conversation_id].popleft()
            
        return len(self.rate_limits[conversation_id]) < self.messages_per_minute

    def add_message(self, conversation_id: str):
        if conversation_id not in self.rate_limits:
            self.rate_limits[conversation_id] = deque(maxlen=self.messages_per_minute)
        self.rate_limits[conversation_id].append(datetime.now())

class ModelManager:
    def __init__(self):
        self.storage_path = DATA_DIR / "models"
        self.storage_path.mkdir(exist_ok=True)
        self.default_model = "anthropic/claude-3.5-sonnet" if os.getenv('OPENROUTER_API_KEY') else "gpt-4o"
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
        self.default_provider = "openrouter" if os.getenv('OPENROUTER_API_KEY') else "openai"
        
    def get_provider(self, user_id: str) -> str:
        if user_id not in self.providers:
            self.providers[user_id] = self.load_provider(user_id)
        return self.providers[user_id]
        
    def load_provider(self, user_id: str) -> str:
        file_path = self.storage_path / f"{user_id}.txt"
        try:
            provider = file_path.read_text().strip()
            return provider if provider in ["openai", "openrouter"] else self.default_provider
        except FileNotFoundError:
            self.save_provider(user_id, self.default_provider)
            return self.default_provider
            
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

class MessageFormatter:
    @staticmethod
    def system_message(message: str) -> str:
        """Format a system message with [SYS] prefix"""
        return f"[SYS] {message}"

class SystemPromptManager:
    def __init__(self):
        self.storage_path = DATA_DIR / "prompts"
        self.storage_path.mkdir(exist_ok=True)
        self.default_prompt = "Tvé jméno je Aura. Jsi součástí Discord chatu a jsi tu od toho, abys pomáhala ostatním! Povaha: jsi kawaii uwu discord holka. Technické info: Neprefixuj své zprávy žádným username nebo tagem!. A taky nepiš asterisky používané v 'roleplay'. Tohle je normální chat."
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
    async def on_ready(self):
        logging.info(f'Logged in as {self.user.name}')
        await self.change_presence(activity=discord.Activity(
            type=discord.ActivityType.listening,
            name="DMs | try !help"
        ))

    async def process_message(self, message, conversation_history: list, user_id: str, conversation_id: str) -> str:
        try:
            # Get user-specific configuration
            config = self.provider_manager.get_client_config(user_id)
            client = AsyncOpenAI(**config)
            
            # Filter out system messages and create the message list
            filtered_history = [msg for msg in conversation_history 
                              if not (isinstance(msg["content"], str) and "[SYS]" in msg["content"])]
            
            messages = [{"role": "system", "content": self.prompt_manager.get_prompt(conversation_id)}] + filtered_history
            
            # Log the exact context being sent to the LLM
            logging.info("Context being sent to LLM:")
            logging.info(json.dumps(messages, indent=2))
            
            # Prepare messages for the API
            api_messages = []
            for msg in messages:
                if isinstance(msg["content"], list):
                    # Message with images/mixed content
                    api_messages.append(msg)
                else:
                    # Regular text message
                    api_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

            response = await client.chat.completions.create(
                model=self.model_manager.get_model(user_id),
                messages=api_messages,
                max_tokens=4000,
                temperature=0.9,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logging.error(f"Error in AI processing: {str(e)}")
            raise

    async def on_message(self, message):
        if message.author.bot:
            return
            
        # Handle DMs, messages starting with ? or !, messages mentioning "auro", or ending with "."
        is_dm = isinstance(message.channel, discord.DMChannel)
        is_question = not is_dm and message.content.startswith('?')
        is_command = not is_dm and message.content.startswith('!')
        mentions_aura = not is_dm and ('auro' in message.content.lower())
        ends_with_dot = not is_dm and message.content.strip().endswith('.')
        
        if not (is_dm or is_question or is_command or mentions_aura or ends_with_dot):
            return
            
        # Strip the ? prefix if it's a question
        if is_question:
            message.content = message.content[1:].strip()
            
        # Helper function to get consistent conversation ID
        def get_conversation_id(message):
            return str(message.author.id) if isinstance(message.channel, discord.DMChannel) else f"channel_{message.channel.id}"
            
        # Get the conversation ID (user_id for DMs, channel_id for text channels)
        conversation_id = get_conversation_id(message)
            
        # Handle commands
        if message.content.startswith('!'):
            cmd = message.content[1:].split()[0].lower()
            args = message.content.split()[1:]
            
            if cmd == 'prompt':
                if not args:
                    await message.channel.send(self.message_formatter.system_message("Please provide a new prompt!"))
                    return
                new_prompt = ' '.join(args)
                if len(new_prompt) > 1000:
                    await message.channel.send(self.message_formatter.system_message("Prompt is too long! Please keep it under 1000 characters."))
                    return
                if len(new_prompt.strip()) == 0:
                    await message.channel.send(self.message_formatter.system_message("Prompt cannot be empty!"))
                    return
                try:
                    # Use conversation_id for prompt storage to match channel/DM context
                    self.prompt_manager.save_prompt(conversation_id, new_prompt)
                    # Clear the relevant conversation based on context
                    self.conversation_manager.delete_conversation(conversation_id)
                    await message.channel.send(self.message_formatter.system_message(
                        f"System prompt updated!"))
                except Exception as e:
                    logging.error(f"Error setting prompt: {str(e)}")
                    await message.channel.send(self.message_formatter.system_message("An error occurred while setting the prompt. Please try again."))
                return
                
            elif cmd == 'clear':
                try:
                    # Get the correct conversation ID based on context
                    clear_id = get_conversation_id(message)
                    # Clear the conversation
                    self.conversation_manager.delete_conversation(clear_id)
                    await message.channel.send(self.message_formatter.system_message(
                        f"{'Channel' if not is_dm else 'Personal'} conversation history cleared!"))
                    logging.info(f"Cleared conversation history for ID: {clear_id}")
                except Exception as e:
                    logging.error(f"Error clearing history: {str(e)}")
                    await message.channel.send(self.message_formatter.system_message("An error occurred while clearing history. Please try again."))
                return
                
            elif cmd == 'info':
                try:
                    user_id = str(message.author.id)
                    # Get stats for the relevant conversation context
                    stats = self.conversation_manager.get_stats(conversation_id)
                    info_message = (
                        f"**{'Channel' if not is_dm else 'Personal'} Conversation Statistics**\n"
                        f"Messages in history: {stats['message_count']}\n"
                        f"Total characters: {stats['total_characters']}/{stats['max_characters']}\n"
                        f"System prompt: {self.prompt_manager.get_prompt(conversation_id)}\n"
                        f"Provider: {self.provider_manager.get_provider(conversation_id)}\n"
                        f"Model: {self.model_manager.get_model(conversation_id)}\n"
                    )
                    if stats['last_interaction']:
                        last_time = datetime.fromisoformat(stats['last_interaction'])
                        info_message += f"Last interaction: {last_time.strftime('%Y-%m-%d %H:%M:%S')}"
                    await message.channel.send(info_message)
                except Exception as e:
                    logging.error(f"Error showing info: {str(e)}")
                    await message.channel.send(self.message_formatter.system_message("An error occurred while getting information. Please try again."))
                return
                
            elif cmd == 'provider':
                if not args:
                    await message.channel.send(self.message_formatter.system_message("Please specify a provider (openai/openrouter)!"))
                    return
                provider = args[0].lower()
                if provider not in ["openai", "openrouter"]:
                    await message.channel.send(self.message_formatter.system_message("Invalid provider! Use 'openai' or 'openrouter'."))
                    return
                try:
                    if provider == "openai" and not os.getenv('OPENAI_API_KEY'):
                        await message.channel.send(self.message_formatter.system_message("OPENAI_API_KEY not found in environment!"))
                        return
                    elif provider == "openrouter" and not os.getenv('OPENROUTER_API_KEY'):
                        await message.channel.send(self.message_formatter.system_message("OPENROUTER_API_KEY not found in environment!"))
                        return
                    self.provider_manager.save_provider(conversation_id, provider)
                    await message.channel.send(self.message_formatter.system_message(f"{'Channel' if not is_dm else 'Personal'} AI provider updated to {provider}!"))
                except Exception as e:
                    logging.error(f"Error setting provider: {str(e)}")
                    await message.channel.send(self.message_formatter.system_message("An error occurred while updating the provider. Please try again."))
                return
                
            elif cmd == 'model':
                if not args:
                    await message.channel.send(self.message_formatter.system_message("Please specify a model!"))
                    return
                model = args[0]
                try:
                    self.model_manager.save_model(conversation_id, model)
                    await message.channel.send(self.message_formatter.system_message(f"{'Channel' if not is_dm else 'Personal'} AI model updated to {model}!"))
                except Exception as e:
                    logging.error(f"Error setting model: {str(e)}")
                    await message.channel.send(self.message_formatter.system_message("An error occurred while updating the model. Please try again."))
                return
                
            elif cmd == 'setlimit':
                if not args:
                    await message.channel.send(self.message_formatter.system_message("Please specify the maximum number of characters!"))
                    return
                try:
                    max_chars = int(args[0])
                    if max_chars < 1000:
                        await message.channel.send(self.message_formatter.system_message("Maximum characters must be at least 1000!"))
                        return
                    if max_chars > 150000:
                        await message.channel.send(self.message_formatter.system_message("Maximum characters cannot exceed 150,000!"))
                        return
                    self.conversation_manager.set_max_chars(max_chars)
                    await message.channel.send(self.message_formatter.system_message(f"Maximum conversation history updated to {max_chars:,} characters!"))
                except ValueError:
                    await message.channel.send(self.message_formatter.system_message("Please provide a valid number!"))
                except Exception as e:
                    logging.error(f"Error setting character limit: {str(e)}")
                    await message.channel.send(self.message_formatter.system_message("An error occurred while updating the character limit. Please try again."))
                return
                
            elif cmd == 'clear-prompt':
                try:
                    # Reset prompt to default for the conversation context
                    self.prompt_manager.save_prompt(conversation_id, self.prompt_manager.default_prompt)
                    await message.channel.send(self.message_formatter.system_message(
                        f"{'Channel' if not is_dm else 'Personal'} system prompt reset to default!"))
                except Exception as e:
                    logging.error(f"Error clearing prompt: {str(e)}")
                    await message.channel.send(self.message_formatter.system_message("An error occurred while clearing the prompt. Please try again."))
                return

            elif cmd == 'help':
                help_text = self.message_formatter.system_message("""
**Available Commands:**
!prompt <new prompt> - Set a new system prompt
!clear-prompt - Reset system prompt to default
!clear - Clear conversation history
!info - Show conversation statistics
!provider <openai/openrouter> - Set AI provider
!model <model name> - Set AI model
!setlimit <number> - Set max history characters
!help - Show this help message
""")
                await message.channel.send(help_text)
                return
                
            return
        
        # Check rate limit
        if not self.rate_limiter.can_send(conversation_id):
            await message.channel.send(self.message_formatter.system_message("Please wait a moment before sending another message."))
            return
            
        self.rate_limiter.add_message(conversation_id)

        # If in a text channel, fetch message history
        if not is_dm:
            try:
                # Get channel history up to max_chars limit
                channel_history = []
                total_chars = 0
                max_history_chars = self.conversation_manager.max_chars - len(message.content)
                
                # Find the last !clear command
                last_clear = None
                async for msg in message.channel.history(limit=30, before=message):
                    if msg.content.strip() == '!clear':
                        last_clear = msg
                        break

                # Get messages after the last clear command, or all recent messages if no !clear found
                history_query = {"limit": 100, "before": message, "oldest_first": True}
                if last_clear:
                    history_query["after"] = last_clear
                
                async for msg in message.channel.history(**history_query):
                    content = msg.content
                    if content.startswith('?'):
                        content = content[1:].strip()
                    
                    # Calculate message size
                    msg_size = len(content)
                    if total_chars + msg_size > max_history_chars:
                        break
                        
                    role = "assistant" if msg.author.id == self.user.id else "user"
                    # Include username in content
                    display_name = f"{msg.author.name}#{msg.author.discriminator}" if msg.author.discriminator != '0' else msg.author.name
                    prefixed_content = f"[{display_name}]: {content}"
                    channel_history.append({"role": role, "content": prefixed_content})
                    total_chars += msg_size
                
                # Clear existing conversation
                self.conversation_manager.delete_conversation(conversation_id)
                
                # Add channel history (already in chronological order)
                for msg in channel_history:
                    self.conversation_manager.add_message(conversation_id, msg)
                    
            except Exception as e:
                logging.error(f"Error fetching channel history: {str(e)}")

        # Handle message content
        # Remove trailing dot if that's what triggered the response
        message_content = message.content
        if ends_with_dot and message_content.strip().endswith('.'):
            message_content = message_content.rstrip('.')

        if message.attachments and any(att.content_type.startswith('image/') for att in message.attachments):
            # If there are images, create a content array
            content = []
            # Add any text message first if it exists
            if message_content.strip():
                content.append({"type": "text", "text": message_content.strip()})
            # Add each image
            for attachment in message.attachments:
                if attachment.content_type.startswith('image/'):
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "detail": "high",
                            "url": attachment.url,
                        },
                    })
        else:
            # Regular text message
            content = message_content

        # Add user message to conversation with username
        display_name = f"{message.author.name}#{message.author.discriminator}" if message.author.discriminator != '0' else message.author.name
        if isinstance(content, list):
            # For messages with images, add username to the text portion if it exists
            if content and content[0]["type"] == "text":
                content[0]["text"] = f"[{display_name}]: {content[0]['text']}"
        else:
            content = f"[{display_name}]: {content}"
            
        self.conversation_manager.add_message(
            conversation_id,
            {"role": "user", "content": content}
        )

        try:
            async with message.channel.typing():
                ai_response = await self.process_message(
                    message,
                    self.conversation_manager.get_conversation(conversation_id),
                    conversation_id,  # Use conversation_id for model/provider settings
                    conversation_id  # Pass conversation_id for prompt lookup
                )
            
                # Strip bot's username prefix if present
                ai_response = ai_response.lstrip()  # Remove leading whitespace
                if ai_response.startswith('['):
                    # Find the closing bracket and colon
                    closing_bracket = ai_response.find(']')
                    if closing_bracket != -1 and len(ai_response) > closing_bracket + 1:
                        if ai_response[closing_bracket + 1] == ':':
                            # Remove the [username]: prefix and any following whitespace
                            ai_response = ai_response[closing_bracket + 2:].lstrip()
                        
                # Strip diacritics from the response
                ai_response = strip_diacritics(ai_response)

            # Split long responses
            if len(ai_response) > 2000:
                chunks = [ai_response[i:i+1999] for i in range(0, len(ai_response), 1999)]
                for chunk in chunks:
                    await message.channel.send(chunk)
            else:
                await message.channel.send(ai_response)

            # Add AI response to conversation
            self.conversation_manager.add_message(
                conversation_id,
                {"role": "assistant", "content": ai_response}
            )

        except Exception as e:
            logging.error(f"Error processing message: {str(e)}")
            await message.channel.send(self.message_formatter.system_message("Sorry, I encountered an error. Please try again later."))

    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.dm_messages = True
        
        super().__init__(intents=intents)
        
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
        self.message_formatter = MessageFormatter()
        self.prompt_manager = SystemPromptManager()


if __name__ == "__main__":
    client = AICompanion()
    
    async def main():
        async with client:
            await client.start(os.getenv('DISCORD_TOKEN'))

    try:
        asyncio.run(main())
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
