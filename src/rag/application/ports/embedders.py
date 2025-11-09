"""
Embedding backends for the RAG indexer.

This module provides a simple abstraction to generate embeddings from
text. It includes a Protocol interface, a Mistral-based implementation,
and a Fake implementation for tests.
"""
from __future__ import annotations

import os
import time
from typing import List, Optional, Protocol
from dotenv import load_dotenv

from mistralai import Mistral

load_dotenv()


class Embedder(Protocol):
    """Minimal embedding interface.

    Methods accept strings or batches and return vectors of floats.
    Implementations should keep order and return None on per-item failure.
    """

    def embed(self, text: str) -> Optional[List[float]]:
        """Embed one string.

        Args:
            text (str): Input text.

        Returns:
            Optional[List[float]]: Embedding or None on failure.
        """

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Embed a batch of strings while preserving order.

        Args:
            texts (List[str]): Input texts.

        Returns:
            List[Optional[List[float]]]: Embeddings or None per item.
        """

    def embed_query(self, text: str) -> Optional[List[float]]:
        """Alias to match retriever expectations."""

    def fingerprint(self) -> dict:
        """Return a dict describing the embedder configuration.

        Returns:
            dict: Embedder configuration details.
        """


class MistralEmbedder:
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
        self.model = model
        self.delay = delay
        self.dim = 1024
        self.normalize = True
        self.pooling = "mean"

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

    def embed_query(self, text: str) -> Optional[List[float]]:
        """Alias to match retriever expectations."""
        return self.embed(text)

    def fingerprint(self) -> dict:
        """Return embedder configuration details.

        Returns:
            dict: Embedder configuration.
        """
        return {
            "name": self.model,
            "provider": "mistral",
            "dim": self.dim,
            "normalize": self.normalize,
            "pooling": self.pooling
        }

    @property
    def output_dim(self) -> int:
        """Return embedding output dimension.

        Returns:
            int: Embedding vector dimension.
        """
        return self.dim


class FakeEmbedder:
    """Deterministic, offline embedder for tests and development.

    Produces small fixed-size vectors derived from a hash of the input.
    """

    def __init__(self, dim: int = 8):
        """Initialize fake embedder.

        Args:
            dim (int): Output vector dimension.
        """
        self.dim = dim

    def embed(self, text: str) -> Optional[List[float]]:
        """Embed text deterministically into a small vector.

        Args:
            text (str): Input text.

        Returns:
            Optional[List[float]]: Vector of floats.
        """
        import hashlib

        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [b / 255.0 for b in h[: self.dim]]
        return vec

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Embed a batch using the single-text method.

        Args:
            texts (List[str]): Input texts.

        Returns:
            List[Optional[List[float]]]: Embeddings per text.
        """
        return [self.embed(t) for t in texts]
