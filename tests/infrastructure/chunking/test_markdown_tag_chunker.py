"""Tests for MarkdownTagChunker.

This module validates the wiring between MarkdownTagChunker and the
MarkdownTextSplitter dependency.
"""

from typing import List

from src.rag.infrastructure.chunking.markdown_tag_chunker import (
    MarkdownTagChunker,
)


#################################################
# Unit tests: internal wiring and configuration #
#################################################

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


def test_split_delegates_to_markdown_splitter(monkeypatch):
    """The chunker should return whatever the underlying splitter returns.

    Args:
        monkeypatch: Pytest helper to override MarkdownTextSplitter.
    """

    class DummySplitter:
        """Splitter that records configuration and incoming texts."""

        def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.calls: list[str] = []

        def split_text(self, text: str) -> List[str]:
            self.calls.append(text)
            return [f"chunk::{text}"]

    created_instances: list[DummySplitter] = []

    def factory(chunk_size: int, chunk_overlap: int) -> DummySplitter:
        """Factory that mimics MarkdownTextSplitter constructor.

        Args:
            chunk_size: Desired chunk size.
            chunk_overlap: Desired overlap.

        Returns:
            DummySplitter: Instrumented splitter instance.
        """
        splitter = DummySplitter(chunk_size, chunk_overlap)
        created_instances.append(splitter)
        return splitter

    # Replace MarkdownTextSplitter by factory
    monkeypatch.setattr(
        "src.rag.infrastructure.chunking.markdown_tag_chunker"
        ".MarkdownTextSplitter",
        factory,
    )

    chunker = MarkdownTagChunker(chunk_size=120, chunk_overlap=20)
    text = "# Title\n\nParagraph one.\n\n## Subtitle\n\nParagraph two."
    result = chunker.split(text)

    assert result == [f"chunk::{text}"]
    assert created_instances, "Expected a MarkdownTextSplitter instance."
    splitter = created_instances[0]
    assert splitter.chunk_size == 120
    assert splitter.chunk_overlap == 20
    assert splitter.calls == [text]


#############################################################
# Integration tests: markdown splitter end-to-end behaviour #
#############################################################

def test_split_preserves_markdown_sections_with_chunk_size():
    """Ensure Markdown sections remain grouped end-to-end."""
    chunker = MarkdownTagChunker(chunk_size=60, chunk_overlap=0)
    text = (
        "# Title\n\nIntro paragraph.\n\n"
        "## Subtitle\n\nSecond paragraph.\n\n"
        "### Sub-subtitle\n\nFinal note."
    )

    chunks = chunker.split(text)
    assert len(chunks) >= 2
    assert chunks[0].startswith("# Title")
    assert "Intro paragraph." in chunks[0]
    assert "## Subtitle" in chunks[0]
    assert "Second paragraph." in chunks[0]
    assert chunks[1].startswith("### Sub-subtitle")
    assert "Final note." in chunks[1]


def test_split_contains_all_markdown_sections():
    """Ensure every Markdown section appears in some chunk."""
    chunker = MarkdownTagChunker(chunk_size=120, chunk_overlap=0)
    text = (
        "# Title\n\nIntro paragraph.\n\n"
        "## Subtitle\n\nSecond paragraph.\n\n"
        "### Sub-subtitle\n\nFinal note."
    )

    chunks = chunker.split(text)

    def contains_section(header: str, body: str) -> bool:
        return any(header in chunk and body in chunk for chunk in chunks)

    assert contains_section("# Title", "Intro paragraph.")
    assert contains_section("## Subtitle", "Second paragraph.")
    assert contains_section("### Sub-subtitle", "Final note.")
