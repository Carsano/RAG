"""
Abstract LLM interface.
Defines methods for chat completion and text embeddings.
"""
from typing import List, Protocol
from src.rag.infrastructure.config.types import LLMMessage


class LLM(Protocol):
    def chat(self, messages: List[LLMMessage]) -> str: ...
