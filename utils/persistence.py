import json
import logging
from collections import deque
from datetime import datetime, timedelta

from core.config import DATA_DIR


class SettingsManager:
    """A generic manager for user-specific or channel-specific settings stored in text files."""
    def __init__(self, setting_name: str, default_value: str):
        self.storage_path = DATA_DIR / setting_name
        self.storage_path.mkdir(exist_ok=True)
        self.default_value = default_value
        self.settings = {}

    def get_setting(self, context_id: str) -> str:
        if context_id not in self.settings:
            self.settings[context_id] = self._load_setting(context_id)
        return self.settings[context_id]

    def _load_setting(self, context_id: str) -> str:
        file_path = self.storage_path / f"{context_id}.txt"
        try:
            return file_path.read_text().strip()
        except FileNotFoundError:
            self._save_setting(context_id, self.default_value)
            return self.default_value

    def _save_setting(self, context_id: str, value: str):
        file_path = self.storage_path / f"{context_id}.txt"
        file_path.write_text(value)
        self.settings[context_id] = value

    def set_setting(self, context_id: str, value: str):
        self._save_setting(context_id, value)


class ConversationManager:
    def __init__(self, expiry_hours=3600, max_chars=120000):
        self.conversations = {}
        self.max_chars = max_chars
        self.expiry_hours = expiry_hours
        self.last_interaction = {}
        self.storage_path = DATA_DIR / "conversations"
        self.storage_path.mkdir(exist_ok=True)
        self.load_conversations()

    def load_conversations(self):
        for file in self.storage_path.glob("*.json"):
            user_id = file.stem
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    self.conversations[user_id] = deque(data['messages'])
                    self.last_interaction[user_id] = datetime.fromisoformat(data['last_interaction'])
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
        if (user_id not in self.last_interaction or
            current_time - self.last_interaction[user_id] > timedelta(hours=self.expiry_hours)):
            self.conversations[user_id] = deque()
        
        self.conversations[user_id].append(message)
        self.last_interaction[user_id] = current_time
        
        total_chars = 0
        for msg in self.conversations[user_id]:
            if isinstance(msg["content"], list):
                for item in msg["content"]:
                    if item["type"] == "image_url":
                        total_chars += 3000
                    else:
                        total_chars += len(item.get("text", ""))
            else:
                total_chars += len(msg.get("content", ""))
        
        while total_chars > self.max_chars and len(self.conversations[user_id]) > 1:
            removed_msg = self.conversations[user_id].popleft()
            if isinstance(removed_msg["content"], list):
                 for item in removed_msg["content"]:
                    if item["type"] == "image_url":
                        total_chars -= 3000
                    else:
                        total_chars -= len(item.get("text", ""))
            else:
                total_chars -= len(removed_msg.get("content", ""))

        self.save_conversation(user_id)

    def get_conversation(self, user_id: str) -> list:
        return list(self.conversations.get(user_id, []))

    def set_max_chars(self, new_max: int):
        self.max_chars = new_max
        for user_id in self.conversations:
            # Create a dummy message to trigger the trimming logic without actually adding a message
            dummy_message = {"role": "system", "content": ""}
            self.add_message(user_id, dummy_message)
            # Remove the dummy message that was just added
            if self.conversations[user_id] and self.conversations[user_id][-1] == dummy_message:
                self.conversations[user_id].pop()
            self.save_conversation(user_id)


    def get_stats(self, user_id: str) -> dict:
        conversation = self.conversations.get(user_id, deque())
        total_chars = 0
        for msg in conversation:
            if isinstance(msg["content"], list):
                for item in msg["content"]:
                    if item["type"] == "image_url":
                        total_chars += 3000
                    else:
                        total_chars += len(item.get("text", ""))
            else:
                total_chars += len(msg.get("content", ""))
        
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