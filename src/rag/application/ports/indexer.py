"""Indexer protocol definition."""

from typing import List, Protocol


class Indexer(Protocol):
    """
    Protocol for an Indexer that indexes documents.
    """
    def rebuild(self, vectors: List[List[float]]) -> None:
        """Rebuild the index from a full list of vectors.

        Args:
            vectors (List[List[float]]): Embedding vectors.
        """
    def add(self, vectors: List[List[float]]) -> None:
        """Append vectors to an existing index. No-op if index is empty."""
    def save(self, path: str) -> None:
        """Persist the current index to disk."""
