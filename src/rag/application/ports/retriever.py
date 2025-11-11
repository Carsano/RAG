"""Retriever port definition."""
from __future__ import annotations
from typing import Protocol, List


class Retriever(Protocol):
    """Port defining a retriever interface for fetching relevant chunks."""
    def retrieve(self, query: str, k: int = 10) -> List[str]:
        """Retrieve the most relevant text chunks for a given query.

        Args:
            query (str): Input query to search for.
            k (int): Number of top chunks to return.

        Returns:
            List[str]: Retrieved text chunks sorted by relevance.
        """
        ...
