import logging
import discord
from discord.ext import commands

from core.services import AIService
from core.image_service import ImageService
from utils.persistence import ConversationManager, RateLimiter, SettingsManager
from utils.formatting import system_message, strip_diacritics
from utils.context import get_conversation_id


class EventsCog(commands.Cog):
    def __init__(self, bot: commands.Bot, conversation_manager: ConversationManager,
                 rate_limiter: RateLimiter, ai_service: AIService,
                 prompt_manager: SettingsManager, image_service: ImageService):
        self.bot = bot
        self.conversation_manager = conversation_manager
        self.rate_limiter = rate_limiter
        self.ai_service = ai_service
        self.prompt_manager = prompt_manager
        self.image_service = image_service

    @commands.Cog.listener()
    async def on_ready(self):
        if self.bot.user:
            logging.info(f'Logged in as {self.bot.user.name}')
            # Remove presence for now.
            # await self.bot.change_presence(activity=discord.Activity(
            #     type=discord.ActivityType.listening,
            #     name="DMs | try !help"
            # ))

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot:
            return

        # Ignore messages that are commands
        # self.bot.command_prefix can be a list, so we check it carefully
        prefix = self.bot.command_prefix
        if isinstance(prefix, str) and message.content.startswith(prefix):
            return
        if isinstance(prefix, (list, tuple)) and any(message.content.startswith(p) for p in prefix):
            return

        is_dm = isinstance(message.channel, discord.DMChannel)
        is_question = not is_dm and message.content.startswith('?')
        mentions_aura = not is_dm and ('auro' in message.content.lower())
        ends_with_dot = not is_dm and message.content.strip().endswith('.')

        if not (is_dm or is_question or mentions_aura or ends_with_dot):
            return

        conversation_id = get_conversation_id(message)

        if not self.rate_limiter.can_send(conversation_id):
            await message.channel.send(system_message("Please wait a moment before sending another message."))
            return
        self.rate_limiter.add_message(conversation_id)

        try:
            async with message.channel.typing():
                # --- Prepare Context ---
                if not is_dm:
                    await self._fetch_and_rebuild_channel_history(message, conversation_id)

                # --- Prepare User Message ---
                user_content = await self._prepare_user_message(message, is_question, ends_with_dot)
                
                self.conversation_manager.add_message(
                    conversation_id,
                    {"role": "user", "content": user_content}
                )

                # --- Get AI Response ---
                conversation = self.conversation_manager.get_conversation(conversation_id)
                system_prompt = self.prompt_manager.get_setting(conversation_id)
                
                ai_response = await self.ai_service.get_ai_response(
                    conversation_id, conversation, system_prompt
                )

                # --- Process and Send Response ---
                processed_response = self._process_ai_response(ai_response)
                
                if len(processed_response) > 2000:
                    chunks = [processed_response[i:i+1999] for i in range(0, len(processed_response), 1999)]
                    for chunk in chunks:
                        await message.channel.send(chunk)
                else:
                    await message.channel.send(processed_response)

                # --- Update History ---
                self.conversation_manager.add_message(
                    conversation_id,
                    {"role": "assistant", "content": processed_response}
                )

        except Exception as e:
            logging.error(f"Error processing message: {str(e)}")
            await message.channel.send(system_message("Sorry, I encountered an error. Please try again later."))

    async def _fetch_and_rebuild_channel_history(self, message: discord.Message, conversation_id: str):
        try:
            channel_history = []
            total_chars = 0
            max_history_chars = self.conversation_manager.max_chars - len(message.content)

            last_clear = None
            async for msg in message.channel.history(limit=100, before=message):
                if msg.content.strip() == f'{self.bot.command_prefix}clear':
                    last_clear = msg
                    break

            history_query = {"limit": 100, "before": message, "oldest_first": True}
            if last_clear:
                history_query["after"] = last_clear

            async for msg in message.channel.history(**history_query):
                content = msg.content
                if content.startswith('?'):
                    content = content[1:].strip()

                msg_size = len(content)
                if total_chars + msg_size > max_history_chars:
                    break

                # Add a null check for self.bot.user
                role = "assistant" if self.bot.user and msg.author.id == self.bot.user.id else "user"
                display_name = f"{msg.author.name}#{msg.author.discriminator}" if msg.author.discriminator != '0' else msg.author.name
                prefixed_content = f"[{display_name}]: {content}"
                channel_history.append({"role": role, "content": prefixed_content})
                total_chars += msg_size

            self.conversation_manager.delete_conversation(conversation_id)
            for msg in channel_history:
                self.conversation_manager.add_message(conversation_id, msg)
        except Exception as e:
            logging.error(f"Error fetching channel history: {str(e)}")

    async def _prepare_user_message(self, message: discord.Message, is_question: bool, ends_with_dot: bool):
        message_content = message.content
        if is_question:
            message_content = message_content[1:].strip()
        if ends_with_dot and message_content.strip().endswith('.'):
            message_content = message_content.rstrip('.')

        image_descriptions = []
        if message.attachments:
            image_attachments = [
                att for att in message.attachments
                if att.content_type and att.content_type.startswith('image/')
            ]
            for attachment in image_attachments:
                try:
                    description = await self.image_service.describe_image(attachment.url)
                    image_descriptions.append(f"[Image description: {description}]")
                except Exception as e:
                    logging.error(f"Failed to describe image {attachment.url}: {e}")
                    image_descriptions.append("[Image description: Failed to analyze image.]")

        full_message = message_content
        if image_descriptions:
            full_message += " " + " ".join(image_descriptions)

        display_name = f"{message.author.name}#{message.author.discriminator}" if message.author.discriminator != '0' else message.author.name
        
        return f"[{display_name}]: {full_message.strip()}"

    def _process_ai_response(self, ai_response: str) -> str:
        response = ai_response.lstrip()
        if response.startswith('['):
            closing_bracket = response.find(']')
            if closing_bracket != -1 and len(response) > closing_bracket + 1 and response[closing_bracket + 1] == ':':
                response = response[closing_bracket + 2:].lstrip()
        
        return strip_diacritics(response)

