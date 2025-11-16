"""LangChain client implementation for Mistral LLM integration.

This module provides a wrapper around the LangChain ChatMistralAI client,
enabling interaction with the Mistral language model using a standard
interface defined by the LLM base class.
"""
# src/rag/infrastructure/llm/langchain_mistral_client.py

from typing import Sequence, Mapping, Optional, List

from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    BaseMessage,
)

from src.rag.application.ports.llm import LLM, ChatMessage


def _to_lc_messages(msgs: Sequence[ChatMessage]) -> List[BaseMessage]:
    """Convert generic chat messages to LangChain message objects.

    Args:
        msgs: A sequence of ChatMessage instances representing the
        chat history.

    Returns:
        A list of BaseMessage instances compatible with LangChain clients.

    Raises:
        ValueError: If a message role is unsupported.
    """
    out: List[BaseMessage] = []
    for m in msgs:
        r = m.role.lower()
        if r == "system":
            out.append(SystemMessage(content=m.content))
        elif r == "user":
            out.append(HumanMessage(content=m.content))
        elif r == "assistant":
            out.append(AIMessage(content=m.content))
        else:
            raise ValueError(f"Unsupported role: {m.role}")
    return out


class LGMistralLLM(LLM):
    """Mistral LLM client implementation using LangChain ChatMistralAI.

    This class wraps the LangChain ChatMistralAI client and exposes a
    standardized interface for chat-based interactions.

    Args:
        model: The name of the Mistral model to use.
        api_key: API key for authenticating with the Mistral service.
        temperature: Sampling temperature for text generation.
        max_retries: Maximum number of retries on failure.

    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        temperature: float = 0.0,
        max_retries: int = 5,
    ) -> None:
        self._model = model
        self._llm = ChatMistralAI(
            model=model,
            temperature=temperature,
            max_retries=max_retries,
            api_key=api_key,
        )

    @property
    def model_name(self) -> str:
        return self._model

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        extra: Optional[Mapping[str, object]] = None,
    ) -> str:
        """Send chat messages to the Mistral model and get a response.

        Args:
            messages: A sequence of ChatMessage instances
            representing the chat.
            temperature: Sampling temperature for generation.
            max_tokens: Optional max number of tokens to generate.
            extra: Optional extra parameters to pass to the client.

        Returns:
            The generated response text from the model.
        """
        lc_msgs = _to_lc_messages(messages)
        kwargs = dict(temperature=temperature)
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if extra:
            kwargs.update(extra)
        # LangChain's .invoke returns a BaseMessage
        resp = self._llm.invoke(lc_msgs, **kwargs)
        # Ensure we always return raw text
        return getattr(resp, "content", str(resp))
