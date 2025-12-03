"""
MarkdownTagChunker for RAG indexing.
"""

from typing import List

from langchain_text_splitters import (
    MarkdownTextSplitter
)

from src.rag.application.ports.chunkers import Chunker


class MarkdownTagChunker(Chunker):
    """Chunker that respects Markdown structure.

    This chunker splits Markdown text into smaller chunks while preserving
    structural elements like headers, lists, and tables.
    """
    VERSION = "2025-11-30_markdown_tag_v1"

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150
                 ) -> None:
        """Initialize the MarkdownTagChunker.

        Args:
            chunk_size (int): Maximum size of each chunk.
            chunk_overlap (int): Number of overlapping characters
            between chunks.
        """
        self._splitter = MarkdownTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def split(self, text: str) -> List[str]:
        """Split the input Markdown text into chunks.

        Args:
            text (str): Input Markdown text.

        Returns:
            List[str]: List of Markdown chunks.
        """
        return self._splitter.split_text(text)


__all__ = [
    "MarkdownTagChunker"
]
