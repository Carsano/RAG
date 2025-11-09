"""
Mistral client adapter module.
Implements the abstract LLM interface using the Mistral API to provide
chat completions and text embeddings.
"""
# src/rag/infrastructure/llm/mistral_client.py
import os
from typing import List
from src.rag.infrastructure.config.types import LLMMessage
from src.rag.application.ports.llm import LLM
from mistralai import Mistral


class MistralLLM(LLM):
    """
    Mistral LLM client implementing chat and embedding.

    Uses the Mistral API to perform language model operations such as
    generating chat completions and embedding text.

    Attributes:
        chat_model (str): The model name used for chat completions.
        embed_model (str): The model name used for embeddings.
        args (dict): Additional arguments for completion calls.
        client (Mistral): The injected Mistral API client instance.
    """

    def __init__(self, model_name: str,
                 temperature: float = 2,
                 max_tokens: int = 300,
                 top_p: float = 0.22,
                 client: Mistral | None = None):
        """
        Initialize the MistralLLM client with specified models and args.

        Args:
            model_name (str): Model name for chat completions.
            embed_model (str): Model name for text embeddings.
            completion_args (dict): Optional arguments for completion calls.
            client (Mistral | None): Optional prebuilt client. If None, a
                client is created from MISTRAL_API_KEY.

        Returns:
            None
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.client = client or Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

    def chat(self, messages: List[LLMMessage]) -> str:
        """
        Generate a chat completion from a list of messages.

        Args:
            messages (List[LLMMessage]): List of messages forming the chat
                history or prompt.

        Returns:
            str: The content string of the generated chat completion.

        Raises:
            RuntimeError: If the Mistral client is unavailable.
        """
        if not self.client:
            raise RuntimeError("Client Mistral indisponible.")
        resp = self.client.chat.complete(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )
        return resp.choices[0].message.content
