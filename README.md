# Discord AI Companion Bot

A Discord bot that uses a LLM to provide conversational AI interactions in DMs. The bot maintains conversation history and allows customization of system prompts.

## Features

- Private conversations in DMs
- Conversation memory with configurable limits
- Customizable AI personality through system prompts
- Rate limiting to prevent abuse
- Automatic conversation expiry
- Character-based history limiting
- Slash commands for configuration
- More - this guide was written after 20 minutes of development, lol.

## Prerequisites

- Python 3.8 or higher
- A Discord account
- An OpenAI API key

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
     - Use Slash Commands
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
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. Run the bot:
   ```bash
   python bot.py
   ```

## Usage

The bot only responds in DMs. Start a conversation by sending a direct message to the bot.

### Available Commands

- `/prompt <new_prompt>` - Set a new system prompt for the AI's personality
- `/clear` - Clear your conversation history
- `/info` - Show current conversation statistics
- `/setlimit <max_chars>` - Set maximum number of characters to keep in history

## Configuration

The bot has several configurable parameters in `bot.py`:

- `expiry_hours`: How long until conversations expire (default: 24 hours)
- `max_chars`: Maximum characters in conversation history (default: 120,000)
- `messages_per_minute`: Rate limit for messages (default: 5)

## Data Storage

The bot stores all persistent data in the `.data` directory:
- `.data/conversations/` - Conversation histories
- `.data/system_prompt.txt` - Current system prompt
- `.data/bot.log` - Log files
- might change at any point in time without this guide being updates

## License

This project is licensed under the MIT License - see the LICENSE file for details.
