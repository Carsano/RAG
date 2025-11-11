"""
Abstract vector store interface.
Contains methods for adding, searching, and managing vectors.
"""
from typing import List, Tuple, Protocol


class VectorStoreManager(Protocol):
    def search(self, query_embedding: list[float], k: int = 10
               ) -> Tuple[list[int], list[float]]: ...

    def get_chunks(self, ids: List[int]) -> List[str]: ...
