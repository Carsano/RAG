"""
Unit tests for DocumentConverter initialization behaviour.
"""

import pathlib
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


def test_replace_image_placeholders_inserts_links(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Placeholders should be replaced with relative markdown image links."""
    monkeypatch.setattr(converters_mod, "_DoclingConverter", None)
    converter = converters_mod.DocumentConverter(
        input_root=tmp_path / "in",
        output_root=tmp_path / "out",
    )

    md = "Intro\n<!-- image -->\nBetween\n<!-- image -->\nEnd"
    img_dir = tmp_path / "out"
    md_out_path = img_dir / "doc.md"
    img_paths = [
        img_dir / "doc_assets" / "page_001.png",
        img_dir / "doc_assets" / "page_002.png",
    ]

    replaced = converter._replace_image_placeholders(md, md_out_path,
                                                     img_paths)

    assert "![page 1](doc_assets/page_001.png)" in replaced
    assert "![page 2](doc_assets/page_002.png)" in replaced


def test_planned_md_path_preserves_relative_structure(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """_planned_md_path should mirror input tree under output root."""
    monkeypatch.setattr(converters_mod, "_DoclingConverter", None)
    converter = converters_mod.DocumentConverter(
        input_root=tmp_path / "in",
        output_root=tmp_path / "out",
    )

    nested_input = converter.input_root / "guides" / "intro.pdf"
    planned = converter._planned_md_path(nested_input)

    expected = converter.output_root / "guides" / "intro.md"
    assert planned == expected


def test_convert_with_docling_returns_markdown_for_non_pdf(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Happy-path: docling result is returned for non-PDF files."""
    docling_mock = MagicMock()
    doc_result = MagicMock()
    doc_result.document.export_to_markdown.return_value = "Converted text"
    docling_mock.convert.return_value = doc_result
    monkeypatch.setattr(
        converters_mod, "_DoclingConverter", lambda: docling_mock
    )

    converter = converters_mod.DocumentConverter(
        input_root=tmp_path / "in",
        output_root=tmp_path / "out",
    )
    sample = tmp_path / "doc.txt"

    result = converter._convert_with_docling(sample)

    assert result == "Converted text"
    docling_mock.convert.assert_called_once_with(str(sample))


def test_convert_with_docling_handles_pdf_placeholders(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """When Docling emits placeholders, exporter output should be injected."""
    docling_mock = MagicMock()
    doc_result = MagicMock()
    dense_segment = " ".join(["content"] * 200)
    doc_result.document.export_to_markdown.return_value = (
        f"{dense_segment}\n<!-- image -->\n{dense_segment}"
    )
    docling_mock.convert.return_value = doc_result
    monkeypatch.setattr(
        converters_mod, "_DoclingConverter", lambda: docling_mock
    )

    exporter = MagicMock()
    img_path = tmp_path / "out" / "asset.png"
    exporter.export_pages.return_value = [img_path]

    converter = converters_mod.DocumentConverter(
        input_root=tmp_path / "in",
        output_root=tmp_path / "out",
        exporter=exporter,
    )
    sample = tmp_path / "doc.pdf"

    result = converter._convert_with_docling(sample)

    assert "![page 1]" in result
    exporter.export_pages.assert_called_once()


def test_convert_with_docling_uses_ocr_on_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Docling errors should fall back to OCR for PDFs."""
    docling_mock = MagicMock()
    docling_mock.convert.side_effect = RuntimeError("boom")
    monkeypatch.setattr(
        converters_mod, "_DoclingConverter", lambda: docling_mock
    )
    ocr_mock = MagicMock(return_value="ocr content")
    exporter = MagicMock()
    exporter.export_pages.return_value = []

    converter = converters_mod.DocumentConverter(
        input_root=tmp_path / "in",
        output_root=tmp_path / "out",
        ocr=ocr_mock,
        exporter=exporter,
    )
    sample = tmp_path / "doc.pdf"

    result = converter._convert_with_docling(sample)

    assert result == "ocr content"
    ocr_mock.ocr_pdf.assert_called_once_with(sample)


def test_convert_file_reads_markdown_directly(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Existing Markdown files should be read without conversion."""
    monkeypatch.setattr(converters_mod, "_DoclingConverter", None)
    converter = converters_mod.DocumentConverter(
        input_root=tmp_path / "in",
        output_root=tmp_path / "out",
    )
    md_file = tmp_path / "note.md"
    md_file.write_text("hello", encoding="utf-8")

    assert converter.convert_file(md_file) == "hello"


def test_convert_file_skips_non_md_without_docling(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """When Docling is unavailable, non-Markdown files are skipped."""
    monkeypatch.setattr(converters_mod, "_DoclingConverter", None)
    converter = converters_mod.DocumentConverter(
        input_root=tmp_path / "in",
        output_root=tmp_path / "out",
    )
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_text("binary", encoding="utf-8")

    assert converter.convert_file(pdf_path) is None


def test_convert_file_delegates_to_docling_when_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Non-Markdown files should be converted via Docling path."""
    docling_mock = object()
    monkeypatch.setattr(
        converters_mod, "_DoclingConverter", lambda: docling_mock
    )
    converter = converters_mod.DocumentConverter(
        input_root=tmp_path / "in",
        output_root=tmp_path / "out",
    )
    recorded = {}

    def fake_convert(path: pathlib.Path) -> str:
        recorded["path"] = path
        return "converted"

    converter._convert_with_docling = fake_convert
    pdf_path = tmp_path / "doc.pdf"

    assert converter.convert_file(pdf_path) == "converted"
    assert recorded["path"] == pdf_path


def test_convert_all_walks_tree_and_writes_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """convert_all should visit files, convert them, and collect outputs."""
    monkeypatch.setattr(converters_mod, "_DoclingConverter", None)
    input_root = tmp_path / "in"
    input_root.mkdir()
    nested = input_root / "sub"
    nested.mkdir()
    file_a = input_root / "a.md"
    file_b = nested / "b.pdf"
    file_c = nested / "c.pdf"
    file_a.write_text("a", encoding="utf-8")
    file_b.write_text("b", encoding="utf-8")
    file_c.write_text("c", encoding="utf-8")

    converter = converters_mod.DocumentConverter(
        input_root=input_root,
        output_root=tmp_path / "out",
    )

    responses = {
        file_a: "copy",
        file_b: "converted",
        file_c: None,
    }

    def fake_convert(path: pathlib.Path) -> str | None:
        return responses[path]

    saved = []

    def fake_save(content, input_path, output_root):
        out_path = output_root / input_path.relative_to(converter.input_root)
        saved.append((content, input_path))
        return out_path

    converter.convert_file = fake_convert
    converter.save_markdown = fake_save

    outputs = converter.convert_all()

    assert len(outputs) == 2
    assert saved[0][0] == "copy"
    assert saved[1][0] == "converted"


def test_save_markdown_writes_content_and_structure(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """save_markdown should mirror relative paths under output root."""
    monkeypatch.setattr(converters_mod, "_DoclingConverter", None)
    converter = converters_mod.DocumentConverter(
        input_root=tmp_path / "in",
        output_root=tmp_path / "out",
    )
    nested_input = converter.input_root / "docs" / "ref.pdf"
    nested_input.parent.mkdir(parents=True, exist_ok=True)
    nested_input.write_text("source", encoding="utf-8")

    out_path = converter.save_markdown(
        content="converted",
        input_path=nested_input,
        output_root=converter.output_root,
    )

    assert out_path == converter.output_root / "docs" / "ref.md"
    assert out_path.read_text(encoding="utf-8") == "converted"
