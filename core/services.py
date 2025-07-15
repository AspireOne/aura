import json
import logging
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from core.config import OPENAI_API_KEY, OPENROUTER_API_KEY
from utils.persistence import SettingsManager

class AIService:
    def __init__(self, model_manager: SettingsManager, provider_manager: SettingsManager):
        self.model_manager = model_manager
        self.provider_manager = provider_manager

    def _get_client_config(self, context_id: str) -> dict:
        provider = self.provider_manager.get_setting(context_id)
        if provider == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not found in environment")
            return {"api_key": OPENAI_API_KEY}
        else:  # openrouter
            if not OPENROUTER_API_KEY:
                raise ValueError("OPENROUTER_API_KEY not found in environment")
            return {
                "api_key": OPENROUTER_API_KEY,
                "base_url": "https://openrouter.ai/api/v1"
            }

    async def get_ai_response(self, context_id: str, conversation_history: list, system_prompt: str) -> str:
        try:
            config = self._get_client_config(context_id)
            client = AsyncOpenAI(**config)

            # Ensure messages are in the correct format for the API
            api_messages: list[ChatCompletionMessageParam] = [
                {"role": "system", "content": system_prompt}
            ]
            for msg in conversation_history:
                # The 'role' key must be present and correctly typed.
                # We assume 'content' is always present and a string.
                role = msg.get("role")
                if role in ["user", "assistant", "system"]:
                    api_messages.append({
                        "role": role,
                        "content": str(msg["content"])  # Ensure content is a string
                    })

            logging.info("Context being sent to LLM:")
            logging.info(json.dumps(api_messages, indent=2, ensure_ascii=False))

            response = await client.chat.completions.create(
                model=self.model_manager.get_setting(context_id),
                messages=api_messages,
                max_tokens=4000,
                temperature=0.9,
            )
            
            # Add null-safety check for the response content
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Received a null response from the AI.")
            return content

        except Exception as e:
            logging.error(f"Error in AI processing: {str(e)}")
            raise