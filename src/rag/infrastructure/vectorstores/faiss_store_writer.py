"""
Faiss vector index builder/writer.
"""


from __future__ import annotations
import numpy as np
import faiss
from typing import List


class FaissStoreWriter:
    """Builder/writer for a Faiss index.

    This class is separate from FaissStore (retrieval) and focuses on
    constructing an index from embedding vectors and saving it to disk.
    """

    def __init__(self):
        self.idx = None

    def rebuild(self, vectors: List[List[float]]) -> None:
        """Rebuild the index from a full list of vectors.

        Args:
            vectors (List[List[float]]): Embedding vectors.
        """
        if not vectors:
            self.idx = None
            return
        arr = np.array(vectors, dtype=np.float32)
        d = arr.shape[1]
        self.idx = faiss.IndexFlatL2(d)
        self.idx.add(arr)

    def add(self, vectors: List[List[float]]) -> None:
        """Append vectors to an existing index. No-op if index is empty."""
        if not vectors:
            return
        if self.idx is None:
            self.rebuild(vectors)
            return
        arr = np.array(vectors, dtype=np.float32)
        self.idx.add(arr)

    def save(self, path: str) -> None:
        """Persist the current index to disk."""
        if self.idx is None:
            raise ValueError("No index to save. Build or add vectors first.")
        faiss.write_index(self.idx, path)
