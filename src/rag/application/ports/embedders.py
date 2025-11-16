"""
Embedding backends for the RAG indexer.

This module provides a simple abstraction to generate embeddings from
text. It includes a Protocol interface, a Mistral-based implementation,
and a Fake implementation for tests.
"""
from __future__ import annotations

from typing import List, Optional, Protocol, Sequence
Vector = List[float]


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

    def embed_documents(self, texts: Sequence[str]) -> List[Vector]:
        """Embed multiple texts.

        Args:
            texts: Sequence of strings to embed.

        Returns:
            List of vectors, one per input text, same order.
        """
        ...

    def embed_query(self, text: str) -> Vector:
        """Embed a single query string.

        Args:
            text: Input string.

        Returns:
            Vector for the input text.
        """
        ...

    @property
    def model(self) -> str:
        """Return the underlying model identifier."""
        ...

    @property
    def dimension(self) -> Optional[int]:
        """Return embedding dimensionality if known, else None."""
        ...
