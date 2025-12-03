"""Tests for SemanticChunker and BoundaryModel integration."""

from typing import List

from src.rag.infrastructure.chunking.semantic_chunkers import (
    BoundaryModel,
    SemanticChunker,
)


############################################
# Unit tests: configuration and delegation #
############################################
class DummyBoundaryModel(BoundaryModel):
    """Simple boundary model double for unit tests."""

    def __init__(self) -> None:
        self.calls: list[dict[str, int | str]] = []

    def segment(
        self,
        text: str,
        target_tokens: int,
        min_tokens: int,
        max_tokens: int,
    ) -> List[str]:
        self.calls.append(
            {
                "text": text,
                "target_tokens": target_tokens,
                "min_tokens": min_tokens,
                "max_tokens": max_tokens,
            }
        )
        return [text.upper()]


def test_split_uses_boundary_model_with_configured_tokens():
    """Ensure SemanticChunker invokes boundary model with configured tokens."""
    model = DummyBoundaryModel()
    chunker = SemanticChunker(
        model, target_tokens=256, min_tokens=128, max_tokens=512
    )
    text = "semantic chunking is powerful"

    result = chunker.split(text)

    assert result == [text.upper()]
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["text"] == text
    assert call["target_tokens"] == 256
    assert call["min_tokens"] == 128
    assert call["max_tokens"] == 512


def test_split_uses_default_token_configuration():
    """Ensure default token configuration is used when unspecified."""
    model = DummyBoundaryModel()
    chunker = SemanticChunker(model)
    text = "defaults should be passed through"

    chunker.split(text)

    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["target_tokens"] == 300
    assert call["min_tokens"] == 120
    assert call["max_tokens"] == 500


###############################################
# Integration tests: boundary model semantics #
###############################################
class FixedBoundaryModel(BoundaryModel):
    """Boundary model double returning predefined segments."""

    def __init__(self, segments: List[str]) -> None:
        self._segments = segments

    def segment(
        self,
        text: str,
        target_tokens: int,
        min_tokens: int,
        max_tokens: int,
    ) -> List[str]:
        return self._segments


def test_split_returns_boundary_model_segments():
    """Ensure SemanticChunker returns the exact segments produced."""
    segments = ["A chunk", "Another chunk", "Last chunk"]
    model = FixedBoundaryModel(segments)
    chunker = SemanticChunker(model)

    chunks = chunker.split("ignored text")

    assert chunks == segments


def test_split_handles_empty_text():
    """Ensure empty inputs result in empty chunk lists."""
    model = FixedBoundaryModel([])
    chunker = SemanticChunker(model)

    chunks = chunker.split("")

    assert chunks == []
