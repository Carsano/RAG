"""
Fake embedder for tests and development.
"""
from __future__ import annotations

from typing import List, Optional

from src.rag.application.ports.embedders import Embedder


class FakeEmbedder(Embedder):
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


__all__ = [
    "FakeEmbedder"
]
