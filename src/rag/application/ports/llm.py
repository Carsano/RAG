"""
Abstract LLM interface.
Defines methods for chat completion and text embeddings.
"""
# src/rag/application/ports/llm.py
from typing import List, Protocol
from src.rag.infrastructure.config.types import LLMMessage


class LLM(Protocol):
    def chat(self, messages: List[LLMMessage]) -> str: ...
    """Generate a chat completion from a list of messages."""
