"""
Abstract LLM interface.
Defines methods for chat completion and text embeddings.
"""
from typing import List, Protocol
from core.types import LLMMessage


class LLM(Protocol):
    def chat(self, messages: List[LLMMessage]) -> str: ...
    def embed(self, text: str) -> list[float]: ...
