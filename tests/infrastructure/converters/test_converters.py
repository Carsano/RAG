"""
Unit tests for DocumentConverter initialization behaviour.
"""

from unittest.mock import MagicMock

import pytest

import src.rag.infrastructure.converters.converters as converters_mod
from src.rag.infrastructure.converters.default_ocr_exporter import (
    DefaultOCRService,
)
from src.rag.infrastructure.converters.default_page_exporter import (
    DefaultPageExporter,
)


class DummyOCR:
    def ocr_pdf(self, _path):
        return "text"


class DummyExporter:
    def export_pages(self, _pdf_path, _md_out_path):
        return []


def test_document_converter_init_uses_injected_dependencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Ensure provided collaborators are kept as-is."""
    docling_instance = object()
    monkeypatch.setattr(
        converters_mod, "_DoclingConverter", lambda: docling_instance
    )

    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    ocr = DummyOCR()
    exporter = DummyExporter()
    logger = MagicMock()

    converter = converters_mod.DocumentConverter(
        input_root=input_root,
        output_root=output_root,
        ocr=ocr,
        exporter=exporter,
        logger=logger,
    )

    assert converter.input_root == input_root
    assert converter.output_root == output_root
    assert converter.ocr is ocr
    assert converter.exporter is exporter
    assert converter.logger is logger
    assert converter._converter is docling_instance


def test_document_converter_init_builds_default_dependencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Ensure defaults are created when collaborators are omitted."""
    monkeypatch.setattr(converters_mod, "_DoclingConverter", None)

    converter = converters_mod.DocumentConverter(
        input_root=tmp_path / "in",
        output_root=tmp_path / "out",
    )

    assert isinstance(converter.ocr, DefaultOCRService)
    assert isinstance(converter.exporter, DefaultPageExporter)
    assert converter.logger is not None
    assert converter._converter is None


def test_text_density_low_handles_empty_and_sparse_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """_text_density_low should flag empty or sparse text samples."""
    monkeypatch.setattr(converters_mod, "_DoclingConverter", None)
    converter = converters_mod.DocumentConverter(
        input_root=tmp_path / "in",
        output_root=tmp_path / "out",
    )

    assert converter._text_density_low("") is True
    assert converter._text_density_low("word") is True


def test_text_density_low_detects_dense_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """_text_density_low should accept content above thresholds."""
    monkeypatch.setattr(converters_mod, "_DoclingConverter", None)
    converter = converters_mod.DocumentConverter(
        input_root=tmp_path / "in",
        output_root=tmp_path / "out",
    )

    dense_text = " ".join(["word"] * 200)
    assert converter._text_density_low(dense_text) is False
