# Aura: An AI bot for discord servers

A Discord bot that uses a LLM to provide conversational AI interactions in both DMs, and specific channels in a server. The bot maintains conversation history and allows customization of system prompts, message limits etc..

| Roleplay | Play games |
|:--------:|:----------:|
| ![image](https://github.com/user-attachments/assets/cb7ffff0-ded8-49f0-bb45-7ab535c75895) | ![image](https://github.com/user-attachments/assets/52b679d1-7e78-4715-8f38-25708ce9728e) |



## Features

- Private conversations in DMs
- Conversation memory with configurable limits
- Customizable AI personality through system prompts
- Rate limiting to prevent abuse (7 messages per minute)
- Automatic conversation expiry after 3600 hours
- Character-based history limiting
- Image handling capabilities
- Simple ! commands for configuration
- Channel message triggers:
  - Messages starting with ? 
  - Messages mentioning "auro"
  - Messages ending with .

## Prerequisites

- Python 3.8 or higher
- A Discord account
- An OpenAI API key or OpenRouter API key

## Setup Instructions

### 1. Discord Developer Portal Setup

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications)
2. Click "New Application" and give your bot a name
3. Go to the "Bot" section
   - Click "Add Bot"
   - Under "Privileged Gateway Intents", enable:
     - Message Content Intent
4. Go to "OAuth2" → "URL Generator"
   - Select "bot" under Scopes
   - Select these permissions:
     - Send Messages
     - Add Reactions
     - Read Message History
5. Copy the generated URL and use it to invite the bot to your server
6. Go back to the "Bot" section and copy your bot token (you'll need this later)

### 2. OpenAI API Setup

1. Go to [OpenAI's website](https://platform.openai.com/)
2. Create an account or log in
3. Go to API keys section
4. Create a new API key and copy it (you'll need this later)

### 3. Bot Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/aspireone/aura.git
   cd aura
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Unix or MacOS:
   source venv/bin/activate
   ```
   (or use Conda `conda create -n aura python=3.11 && conda activate aura`)

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your tokens:
   ```
   DISCORD_TOKEN=your_discord_bot_token_here
   OPENAI_API_KEY=your_openai_api_key_here  # Optional if using OpenRouter
   OPENROUTER_API_KEY=your_openrouter_api_key_here  # Optional if using OpenAI
   ```

5. Run the bot:
   ```bash
   python -m core.bot
   ```

## Usage

The bot responds in DMs. Start a conversation by sending a direct message to the bot.

### Available Commands

- `!prompt <new_prompt>` - Set a new system prompt for the AI's personality
- `!clear-prompt` - Reset system prompt to default
- `!clear` - Clear your conversation history
- `!info` - Show current conversation statistics
- `!setlimit <max_chars>` - Set maximum number of characters to keep in history
- `!provider <openai/openrouter>` - Set AI provider
- `!model <model name>` - Set AI model
- `!help` - Show help message

## Configuration

The bot has several configurable parameters in `bot.py`:

- `expiry_hours`: How long until conversations expire (default: 3600 hours)
- `max_chars`: Maximum characters in conversation history (default: 120,000)
- `messages_per_minute`: Rate limit for messages (default: 7)

## Data Storage

The bot stores all persistent data in the `.data` directory:
- `.data/conversations/` - Conversation histories (JSON files)
- `.data/prompts/` - User-specific system prompts
- `.data/providers/` - User-specific AI provider settings
- `.data/models/` - User-specific AI model settings
- `.data/bot.log` - Log files

Each user's data is stored separately using their Discord ID as the filename.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
