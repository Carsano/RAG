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
        model_name (str): The model name used for chat completions.
        temperature (float): Default temperature.
        max_tokens (int): Default max tokens.
        top_p (float): Default nucleus sampling.
        client (Mistral): The injected Mistral API client instance.
    """

    def __init__(self, model_name: str,
                 temperature: float = 0.2,
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

    def chat(self, messages: List[LLMMessage], **kwargs) -> str:
        """Generate a single chat completion and return its text."""
        if not self.client:
            raise RuntimeError("Client Mistral indisponible.")
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        top_p = kwargs.get("top_p", self.top_p)
        stop = kwargs.get("stop")
        resp = self.client.chat.complete(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
        )
        return resp.choices[0].message.content

    def generate(
        self,
        prompt: str,
        n: int = 1,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: List[str] | None = None,
        **kwargs,
    ) -> list[str]:
        """Ragas-compatible text generation API returning n strings."""
        messages = [{"role": "user", "content": prompt}]
        outputs: list[str] = []
        params = {
            "temperature": self.temperature if temperature is None
            else temperature,
            "top_p": self.top_p if top_p is None else top_p,
            "max_tokens": self.max_tokens if max_tokens is None
            else max_tokens,
            "stop": stop,
        }
        for _ in range(max(1, int(n))):
            outputs.append(self.chat(messages, **params))
        return outputs
