"""
Faiss vector store adapter.
Implements the abstract vector store interface using Faiss.
"""
import os
import pickle
import numpy as np
import faiss
from typing import List, Tuple
from src.rag.application.ports.vector_store import VectorStore
import streamlit as st


@st.cache_resource(show_spinner=False)
def _load_index_and_docs(index_path: str, chunks_path: str):
    """Load a Faiss index and associated document chunks from disk.

    Args:
        index_path (str): Path to the Faiss index file.
        chunks_path (str): Path to the pickled document chunks file.

    Returns:
        Tuple[faiss.Index, List[str]]: The loaded Faiss index and list of
        document chunks.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS introuvable: {index_path}")
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"Chunks introuvables: {chunks_path}")
    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        docs: List[str] = pickle.load(f)
    return index, docs


class FaissStore(VectorStore):
    """Faiss-based implementation of the VectorStore interface.

    This class manages a Faiss index and associated document chunks,
    providing search and retrieval functionality.

    Attributes:
        index (faiss.Index): The Faiss index for vector similarity search.
        documents (List[str]): List of document chunks corresponding to index.
    """

    def __init__(self, index_path: str, chunks_pickle_path: str):
        """Initialize the FaissStore by loading index and document chunks.

        Args:
            index_path (str): Path to the Faiss index file.
            chunks_pickle_path (str): Path to the pickled document chunks file.
        """
        self.index, self.documents = _load_index_and_docs(index_path,
                                                          chunks_pickle_path)

    def search(self, query_embedding: list[float],
               k: int = 10) -> Tuple[list[int], list[float]]:
        """Search the Faiss index for the nearest neighbors to a query vector.

        Args:
            query_embedding (list[float]): The query vector embedding.
            k (int, optional): Number of nearest neighbors to return.
                Defaults to 10.

        Returns:
            Tuple[list[int], list[float]]: Tuple of two lists:
                - indices of the nearest neighbors in the index.
                - distances to the nearest neighbors.
        """
        q = np.array([np.array(query_embedding, dtype=np.float32)])
        distances, indices = self.index.search(q, k)
        return indices[0].tolist(), distances[0].tolist()

    def get_chunks(self, ids: List[int]) -> List[str]:
        """Retrieve document chunks by their indices.

        Args:
            ids (List[int]): List of document chunk indices.

        Returns:
            List[str]: List of document chunks corresponding to the given ids.
        """
        return [
            self.documents[i] for i in ids if 0 <= i < len(self.documents)
            ]
