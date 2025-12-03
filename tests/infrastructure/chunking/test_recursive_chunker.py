"""Tests for RecursiveChunker.

Validates the integration between RecursiveChunker and
RecursiveCharacterTextSplitter.
"""

from typing import List

from src.rag.infrastructure.chunking.recursive_chunker import RecursiveChunker


#################################################
# Unit tests: internal wiring and configuration #
#################################################
def test_split_uses_default_recursive_configuration(monkeypatch):
    """Ensure default configuration is passed to Recursive splitter.

    Args:
        monkeypatch: Pytest helper to stub RecursiveCharacterTextSplitter.
    """
    captured: dict[str, int | str] = {}

    class DummyRecursive:
        """Test double capturing constructor parameters."""

        def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
            captured["chunk_size"] = chunk_size
            captured["chunk_overlap"] = chunk_overlap

        def split_text(self, text: str) -> List[str]:
            captured["text"] = text
            return ["recursive chunk"]

    monkeypatch.setattr(
        "src.rag.infrastructure.chunking.recursive_chunker"
        ".RecursiveCharacterTextSplitter",
        DummyRecursive,
    )

    chunker = RecursiveChunker()
    result = chunker.split("Text body")

    assert result == ["recursive chunk"]
    assert captured["chunk_size"] == 800
    assert captured["chunk_overlap"] == 150
    assert captured["text"] == "Text body"


def test_split_delegates_to_recursive_splitter(monkeypatch):
    """Chunker output should mirror recursive splitter output.

    Args:
        monkeypatch: Pytest helper to override recursive splitter.
    """
    created_instances: list["DummyRecursive"] = []

    class DummyRecursive:
        """Splitter double that records text payloads."""

        def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.calls: list[str] = []

        def split_text(self, text: str) -> List[str]:
            self.calls.append(text)
            return [text.upper()]

    def factory(chunk_size: int, chunk_overlap: int) -> "DummyRecursive":
        """Factory mirroring RecursiveCharacterTextSplitter signature."""
        splitter = DummyRecursive(chunk_size, chunk_overlap)
        created_instances.append(splitter)
        return splitter

    monkeypatch.setattr(
        "src.rag.infrastructure.chunking.recursive_chunker"
        ".RecursiveCharacterTextSplitter",
        factory,
    )

    chunker = RecursiveChunker(chunk_size=256, chunk_overlap=32)
    text = "Some\nlonger text\nwith multiple lines."
    result = chunker.split(text)

    assert result == [text.upper()]
    assert created_instances, "Expected recursive splitter instance."
    splitter = created_instances[0]
    assert splitter.chunk_size == 256
    assert splitter.chunk_overlap == 32
    assert splitter.calls == [text]


##############################################################
# Integration tests: recursive splitter end-to-end behaviour #
##############################################################

def test_split_returns_single_chunk_when_text_small():
    """Ensure small payloads stay in a single chunk."""
    chunker = RecursiveChunker(chunk_size=1024, chunk_overlap=0)
    text = "Tiny corpus for recursive chunker."

    chunks = chunker.split(text)

    assert chunks == [text]


def test_split_with_overlap_preserves_text_order():
    """Ensure overlapping chunks still preserve global ordering."""
    chunker = RecursiveChunker(chunk_size=30, chunk_overlap=5)
    text = " ".join(f"sentence_{i}" for i in range(8))

    chunks = chunker.split(text)

    assert len(chunks) > 1
    cursor = 0
    for chunk in chunks:
        idx = text.find(chunk, cursor)
        assert idx != -1
        cursor = idx
    assert "sentence_0" in chunks[0]
    assert "sentence_7" in chunks[-1]
