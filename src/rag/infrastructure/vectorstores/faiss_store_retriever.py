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

        ids, distances = self.store.search(emb, k=k)
        chunks_iter = iter(self.store.get_chunks(ids))
        sources_iter = iter(self.store.get_sources(ids))

        results: List[dict] = []
        for chunk_id, distance in zip(ids, distances):
            if chunk_id < 0:
                continue

            chunk = next(chunks_iter, None)
            source = next(sources_iter, None)
            if chunk is None or source is None:
                break

            score = None
            if distance is not None:
                score = 1.0 / (1.0 + float(distance)) if distance >= 0 else 0.0

            results.append(
                {
                    "chunk_id": chunk_id,
                    "score_retriever": score,
                    "distance": float(distance)
                    if distance is not None
                    else None,
                    "content": chunk,
                    "source": source,
                }
            )

        return results


__all__ = [
    "FaissRetriever"
]
