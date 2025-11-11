"""Vector store port interface.

Defines the contract for components handling vector indexing,
similarity search, and retrieval.
"""
from typing import List, Tuple, Protocol


class VectorStoreManager(Protocol):
    """Port defining the interface for a vector store manager."""

    def search(self, query_embedding: list[float], k: int = 10
               ) -> Tuple[list[int], list[float]]:
        """Search for the closest vectors to a given query embedding.

        Args:
            query_embedding (list[float]): Query embedding to search for.
            k (int): Number of nearest neighbors to return.

        Returns:
            Tuple[list[int], list[float]]: Tuple of matching vector IDs and
            their similarity scores.
        """
        ...

    def get_chunks(self, ids: List[int]) -> List[str]:
        """Retrieve document chunks by their vector IDs.

        Args:
            ids (List[int]): List of vector IDs.

        Returns:
            List[str]: Corresponding document chunks.
        """
        ...
