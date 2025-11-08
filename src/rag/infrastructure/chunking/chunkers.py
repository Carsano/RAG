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


class RecursiveChunker(Chunker):
    """Robust character-based chunker with overlap.

    Suitable for heterogeneous or poorly formatted documents.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150
                 ) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def split(self, text: str) -> List[str]:
        return self._splitter.split_text(text)


class BoundaryModel(Protocol):
    """Protocol for models that can segment text into semantic units."""

    def segment(
        self, text: str, target_tokens: int, min_tokens: int, max_tokens: int
    ) -> List[str]:
        ...


class SemanticChunker(Chunker):
    """Chunker that delegates boundary detection to a semantic model.

    The provided `boundary_model` must implement the `BoundaryModel` protocol.
    """

    def __init__(
        self,
        boundary_model: BoundaryModel,
        *,
        target_tokens: int = 300,
        min_tokens: int = 120,
        max_tokens: int = 500,
    ) -> None:
        self._model = boundary_model
        self._target_tokens = target_tokens
        self._min_tokens = min_tokens
        self._max_tokens = max_tokens

    def split(self, text: str) -> List[str]:
        return self._model.segment(
            text,
            target_tokens=self._target_tokens,
            min_tokens=self._min_tokens,
            max_tokens=self._max_tokens,
        )


__all__ = [
    "Chunker",
    "MarkdownTagChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "BoundaryModel",
]
