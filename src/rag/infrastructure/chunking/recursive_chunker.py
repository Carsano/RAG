"""
Chunking strategies for RAG indexing.
"""
from __future__ import annotations

from typing import List

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from src.rag.application.ports.chunkers import Chunker


class RecursiveChunker(Chunker):
    """Robust character-based chunker with overlap.

    Suitable for heterogeneous or poorly formatted documents.
    """
    VERSION = "2025-11-30_mrecursive_v1"

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150
                 ) -> None:
        """Initialize the RecursiveChunker.

        Args:
            chunk_size (int): Maximum size of each chunk.
            chunk_overlap (int): Number of overlapping characters
            between chunks.
        """
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def split(self, text: str) -> List[str]:
        """Split the input text into chunks using character-based splitting.

        Args:
            text (str): Input text.

        Returns:
            List[str]: List of text chunks.
        """
        return self._splitter.split_text(text)


__all__ = [
    "RecursiveChunker",
]
