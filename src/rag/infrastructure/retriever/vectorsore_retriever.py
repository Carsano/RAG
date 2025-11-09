"""
Retriever implementation using a vector store.
"""
# src/rag/infrastructure/retriever/vectorstore_retriever.py
from typing import List, Optional
from src.rag.application.ports.retriever import Retriever
from src.rag.application.ports.vector_store import VectorStore
from src.rag.application.ports.embedders import Embedder


class VectorStoreRetriever(Retriever):
    def __init__(self, store: VectorStore, embedder: Embedder,
                 default_k: int = 5,
                 min_score: Optional[float] = None,
                 allow_mismatch: bool = False):
        """Initialize the VectorStoreRetriever.

        Args:
            store (VectorStore): The vector store to use for retrieval.
            embedder: Component with method `embed_query(str) -> list[float]`.
            default_k (int): Default number of documents to retrieve.
            min_score (Optional[float]): Minimum score threshold for results.
            allow_mismatch (bool): Whether to allow embedder/index
            fingerprint mismatches.
        """
        self.store = store
        self.embedder = embedder
        self.default_k = default_k
        self.min_score = min_score

        idx_fp = getattr(self.store, "get_index_fingerprint", lambda: {})()
        idx_dim = getattr(self.store, "get_dim", lambda: 0)()
        emb_fp = getattr(self.embedder, "fingerprint", lambda: {})()
        emb_dim = (
            getattr(self.embedder, "output_dim", None)
            if hasattr(self.embedder, "output_dim")
            else getattr(self.embedder, "dim", None)
        )

        if idx_dim and emb_dim and idx_dim != emb_dim and not allow_mismatch:
            raise ValueError(f"Embedding dimension mismatch: "
                             f"index={idx_dim}, embedder={emb_dim}")

        critical_keys = ["name", "provider", "normalize", "pooling", "dim"]
        mismatches = {
            k: (idx_fp.get(k), emb_fp.get(k))
            for k in critical_keys
            if (idx_fp.get(k) is not None
                and emb_fp.get(k) is not None
                and idx_fp.get(k) != emb_fp.get(k))
        }
        if mismatches and not allow_mismatch:
            raise ValueError(f"Embedder fingerprint mismatch "
                             f"with index: {mismatches}")

    def search(self, query: str, k: int = None) -> List[str]:
        """Retrieve relevant document chunks for a given query.

        Pipeline: text -> embedding -> (ids, dists) -> chunks -> [str]
        """
        k = k or self.default_k
        query_emb = self.embedder.embed_query(query)
        ids, dists = self.store.search(query_emb, k=k)
        chunks = self.store.get_chunks(ids)

        if self.min_score is not None and len(dists) == len(chunks):
            filtered = []
            for text, dist in zip(chunks, dists):
                score = -float(dist)
                if score >= self.min_score:
                    filtered.append(text)
            return filtered

        return chunks
