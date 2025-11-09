"""
Retriever implementation using a vector store.
"""
# src/rag/infrastructure/retriever/vectorstore_retriever.py
from typing import List, Optional, Dict
from src.rag.application.ports.retriever import Retriever
from src.rag.application.ports.vector_store import VectorStore


class VectorStoreRetriever(Retriever):
    def __init__(self, store: VectorStore, default_k: int = 5,
                 min_score: Optional[float] = None):
        """Initialize the VectorStoreRetriever.

        Args:
            store (VectorStore): The vector store to use for retrieval.
            default_k (int): Default number of documents to retrieve.
            min_score (Optional[float]): Minimum score threshold for results.
        """
        self.store = store
        self.default_k = default_k
        self.min_score = min_score

    def search(self, query: str, k: int = None,
               filters: Optional[Dict] = None) -> List[str]:
        """Retrieve relevant document chunks for a given query.

        Args:
            query (str): The input query string.
            k (int, optional): Number of documents to retrieve.
            filters (Optional[Dict], optional): Filters to apply during search.

        Returns:
            List[str]: List of retrieved document chunks.
        """
        k = k or self.default_k
        results = self.store.search(query, k=k, filters=filters,
                                    with_scores=True)
        if self.min_score is not None:
            results = [r for r in results if r.score >= self.min_score]
        return [r.text for r in results]
