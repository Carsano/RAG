"""
Abstract LLM interface.
Defines methods for chat completion and text embeddings.
"""
# src/rag/application/ports/llm.py
from typing import List, Protocol
from src.rag.infrastructure.config.types import LLMMessage


class LLM(Protocol):
    def __init__(self, model_name: str, temperature: float = 0.2,
                 max_tokens: int = 300, top_p: float = 0.22):
        """
        Initialize the LLM client.

        Args:
            model_name (str): The name of the LLM model to use.
            temperature (float): Sampling temperature for generation.
            max_tokens (int): Maximum number of tokens to generate.
            top_p (float): Nucleus sampling parameter.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

    def chat(self, messages: List[LLMMessage]) -> str: ...
    """Generate a chat completion from a list of messages."""
