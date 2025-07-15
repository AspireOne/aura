import unicodedata

def strip_diacritics(text: str) -> str:
    """Remove diacritics from text, converting to ASCII equivalents."""
    return ''.join(c for c in unicodedata.normalize('NFKD', text)
                   if not unicodedata.combining(c))

def system_message(message: str) -> str:
    """Format a system message with [SYS] prefix."""
    return f"[SYS] {message}"