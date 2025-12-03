"""Tests for MarkdownTagChunker.

This module validates the wiring between MarkdownTagChunker and the
MarkdownTextSplitter dependency.
"""
from __future__ import annotations

from typing import List

from src.rag.infrastructure.chunking.markdown_tag_chunker import (
    MarkdownTagChunker,
)


def test_split_uses_default_configuration(monkeypatch):
    """Ensure the chunker wires the default splitter configuration.

    Args:
        monkeypatch: Pytest helper to swap out MarkdownTextSplitter.
    """
    captured: dict[str, int | str] = {}

    class DummySplitter:
        """Minimal splitter used to capture constructor parameters."""

        def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
            captured["chunk_size"] = chunk_size
            captured["chunk_overlap"] = chunk_overlap

        def split_text(self, text: str) -> List[str]:
            captured["text"] = text
            return ["chunk"]

    monkeypatch.setattr(
        "src.rag.infrastructure.chunking.markdown_tag_chunker"
        ".MarkdownTextSplitter",
        DummySplitter,
    )

    chunker = MarkdownTagChunker()
    result = chunker.split("## Heading\n\nContent")

    assert result == ["chunk"]
    assert captured["chunk_size"] == 800
    assert captured["chunk_overlap"] == 150
    assert captured["text"] == "## Heading\n\nContent"
