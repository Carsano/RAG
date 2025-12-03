"""
Mistral embedder implementation.
"""

import os
import time
from typing import List, Optional
from dotenv import load_dotenv
from mistralai import Mistral

from src.rag.application.ports.embedders import Embedder

load_dotenv()


class MistralEmbedder(Embedder):
    """Mistral-backed embedder with simple retry and throttling.

    Attributes:
        model (str): Mistral embedding model name.
        delay (float): Sleep seconds between calls.
        client (Mistral): Mistral client instance.
    """

    def __init__(self, model: str = "mistral-embed", delay: float = 10.0):
        """Initialize Mistral embedder.

        Args:
            model (str): Embedding model.
            delay (float): Delay between calls in seconds.

        Raises:
            RuntimeError: If MISTRAL_API_KEY is missing.
        """
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY missing from environment")
        self.client = Mistral(api_key=api_key)
        self._model = model
        self.delay = delay

    def embed(self, text: str) -> Optional[List[float]]:
        """Embed a single string with one retry on failure.

        Args:
            text (str): Input text.

        Returns:
            Optional[List[float]]: Embedding vector or None.
        """
        try:
            batch = self.client.embeddings.create(model=self.model,
                                                  inputs=text)
            return batch.data[0].embedding
        except Exception:
            time.sleep(60)
            try:
                batch = self.client.embeddings.create(
                    model=self.model, inputs=text
                )
                return batch.data[0].embedding
            except Exception:
                return None

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Embed multiple strings sequentially with throttling.

        Args:
            texts (List[str]): Input texts.

        Returns:
            List[Optional[List[float]]]: Embeddings or None per item.
        """
        out: List[Optional[List[float]]] = []
        for t in texts:
            out.append(self.embed(t))
            time.sleep(self.delay)
        return out

    def embed_query(self, text: str) -> List[float]:
        """LangChain-compatible embed_query method.

        Args:
            text (str): Input text.

        Returns:
            List[float]: Embedding vector.

        Raises:
            RuntimeError: If embedding fails.
        """
        result = self.embed(text)
        if result is None:
            raise RuntimeError("embedding failed")
        return result

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """LangChain-compatible embed_documents method.

        Args:
            texts (List[str]): Input texts.

        Returns:
            List[List[float]]: Embeddings with None replaced by empty lists.
        """
        results = self.embed_batch(texts)
        return [r if r is not None else [] for r in results]

    @property
    def model(self) -> str:
        """Return the model name (read-only)."""
        return self._model


__all__ = [
    "MistralEmbedder"
]
