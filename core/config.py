import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Tokens and Keys ---
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

# --- Directories ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / ".data"
DATA_DIR.mkdir(exist_ok=True)

# --- Bot Settings ---
DEFAULT_SYSTEM_PROMPT = """
Tvé jméno je Aura. Jsi součástí Discord chatu a jsi tu od toho, abys pomáhala ostatním! 
Povaha: jsi kawaii uwu discord holka. 
Technické info: Neprefixuj své zprávy žádným username nebo tagem!. A taky nepiš asterisky používané v 'roleplay'. Tohle je normální chat.
""".strip()

# --- Help Message ---
HELP_TEXT = """
**Available Commands:**
!prompt <new prompt> - Set a new system prompt
!clear-prompt - Reset system prompt to default
!clear - Clear conversation history
!info - Show conversation statistics
!provider <openai/openrouter> - Set AI provider
!model <model name> - Set AI model
!setlimit <number> - Set max history characters
!help - Show this help message
"""