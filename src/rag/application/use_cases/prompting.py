"""
Prompting utilities.

Provides helper functions for building system prompts and clamping
conversation history.
"""
from __future__ import annotations

from typing import List


def build_system_prompt(base_prompt: str, context_chunks: List[str]) -> str:
    """Build a complete system prompt with context chunks.

    Args:
        base_prompt (str): The base system prompt text.
        context_chunks (List[str]): List of context strings to include.

    Returns:
        str: The complete prompt with optional context information.
    """
    if not context_chunks:
        return base_prompt
    ctx = "\n\n".join(context_chunks)
    extra = (
        "Veuillez utiliser les informations suivantes pour"
        "répondre à la question :\n\n"
        f"{ctx}\n\n"
        "Si les informations ne suffisent pas, dites-le."
    )
    return base_prompt + "\n\n" + extra


def clamp_dialog(messages: List[dict], max_messages: int = 5) -> List[dict]:
    """Clamp conversation to a limited number of messages.

    Keeps only the last `max_messages` exchanges between user and assistant,
    starting from a user message.

    Args:
        messages (List[dict]): List of message dictionaries.
        max_messages (int): Maximum number of messages to keep.

    Returns:
        List[dict]: Trimmed list of recent messages.
    """
    convo = [m for m in messages if m["role"] in ("user", "assistant")]
    recent = convo[-max_messages:] if len(convo) > max_messages else convo
    while recent and recent[0]["role"] != "user":
        recent.pop(0)
    return recent
