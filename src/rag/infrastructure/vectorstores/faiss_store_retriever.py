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

    def retrieve(self, query: str, k: int = 10) -> List[dict]:
        """Embed the query and retrieve top-k relevant chunks.

        Args:
            query (str): Query text.
            k (int): Number of top chunks to return.

        Returns:
            List[dict]: Retrieved chunks,
            each containing 'content' and 'source'.
        """
        emb = self.embedder.embed(query)
        if emb is None:
            raise RuntimeError("Embedding failed in FaissRetriever")

        ids, _ = self.store.search(emb, k=k)

        chunks = self.store.get_chunks(ids)

        sources = self.store.get_sources(ids)

        results = [
            {"content": chunk, "source": src}
            for chunk, src in zip(chunks, sources)
        ]

        return results


__all__ = [
    "FaissRetriever"
]
