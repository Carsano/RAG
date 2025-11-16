"""LLM port interface.

Defines the contract for chat completion and text generation
components.
"""
from __future__ import annotations
from dataclasses import dataclass
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

    @property
    def model(self) -> str:
        """Return the underlying model identifier."""


@dataclass(frozen=True)
class ChatMessage:
    """Single chat message.

    Attributes:
        role: One of {"system", "user", "assistant"}.
        content: Plain text content.
    """
    role: str
    content: str
