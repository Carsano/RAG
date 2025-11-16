"""
Mistral client adapter module.
Implements the abstract LLM interface using the Mistral API to provide
chat completions and text embeddings.
"""

import os
from typing import List
from mistralai import Mistral

from src.rag.application.ports.llm import LLM

from src.rag.infrastructure.config.types import LLMMessage


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

    def __init__(self, chat_model: str, completion_args: dict,
                 client: Mistral | None = None):
        """
        Initialize the MistralLLM client with specified models and args.

        Args:
            chat_model (str): Model name for chat completions.
            embed_model (str): Model name for text embeddings.
            completion_args (dict): Optional arguments for completion calls.
            client (Mistral | None): Optional prebuilt client. If None, a
                client is created from MISTRAL_API_KEY.

        Returns:
            None
        """
        self.chat_model = chat_model
        self.args = completion_args or {}
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
            model=self.chat_model,
            messages=messages,
            temperature=self.args.get("temperature", 0.2),
            max_tokens=self.args.get("max_tokens", 300),
            top_p=self.args.get("top_p", 0.22),
        )
        return resp.choices[0].message.content


__all__ = [
    "MistralLLM"
]
