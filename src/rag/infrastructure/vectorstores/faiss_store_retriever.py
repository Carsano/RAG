"""
FAISS-based retriever implementation.
"""
from typing import List

from src.rag.application.ports.embedders import Embedder
from src.rag.application.ports.vector_store_manager import VectorStoreManager
from src.rag.application.ports.retriever import Retriever


class FaissRetriever(Retriever):
    """Retrieve relevant chunks from a FAISS-based vector store."""

    def __init__(self, embedder: Embedder, store: VectorStoreManager):
        """Initialize retriever with embedder and vector store.

        Args:
            embedder (Embedder): Component that converts text to embeddings.
            store (VectorStoreManager): Component that performs vector search.
        """
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, k: int = 10) -> List[str]:
        """Embed the query and retrieve top-k relevant chunks.

        Args:
            query (str): Query text.
            k (int): Number of top chunks to return.

        Returns:
            List[str]: Retrieved chunks.
        """
        emb = self.embedder.embed(query)
        if emb is None:
            raise RuntimeError("Embedding failed in FaissRetriever")

        ids, _ = self.store.search(emb, k=k)
        return self.store.get_chunks(ids)
