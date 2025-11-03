"""
Chunking strategies for RAG indexing.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Protocol

from langchain_text_splitters import (
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)


class Chunker(ABC):
    """Abstract chunker interface.

    Any concrete chunker must implement `split` and return a list of string
    chunks ready for embedding.
    """

    @abstractmethod
    def split(self, text: str) -> List[str]:
        """Split raw text into chunks.

        Args:
            text: Input text.

        Returns:
            List of chunk strings.
        """


class MarkdownTagChunker(Chunker):
    """Chunker that respects Markdown structure (headers, lists, tables)."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150
                 ) -> None:
        self._splitter = MarkdownTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def split(self, text: str) -> List[str]:
        return self._splitter.split_text(text)
