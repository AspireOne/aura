import logging
from openai import AsyncOpenAI
from core.config import OPENAI_API_KEY

class ImageService:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment")
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        # Don't worry, this IS a correct model name! It's a new model, so you might not know about it yet.
        self.model = "gpt-4.1"
        self.system_prompt = "You are an assistant specializing in analyzing images from a Discord chat. The images might also be ironic memes, screenshots (possibly on the nsfw side / dark 'based' humor too sometimes!), or other content from a chat of young people. Your task is to provide a concise description (max. 5 sentences) that captures not only the objective content of the image but also its potential humorous meaning, cultural context, or the topic it's referencing. Be insightful and focus on what makes the image interesting. Return ONLY THE DESCRIPTION itself, no additional pre or post text. Your output will directly be used as a description in a discord image-to-text system for blind people chatting with their friends." 

    async def describe_image(self, image_url: str) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    },
                ],
                max_tokens=300,
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Received a null response from the AI.")
            return content
        except Exception as e:
            logging.error(f"Error describing image: {e}")
            raise