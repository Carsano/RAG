"""LLM port interface.

Defines the contract for chat completion and text generation
components.
"""
from typing import List, Protocol
from src.rag.infrastructure.config.types import LLMMessage


class LLM(Protocol):
    """Port defining the interface for a Large Language Model."""

    def chat(self, messages: List[LLMMessage]) -> str:
        """Send chat messages to the model and get a response.

        Args:
            messages (List[LLMMessage]): Ordered list of conversation messages.

        Returns:
            str: Model's text reply.
        """
        ...
