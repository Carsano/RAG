"""Unit tests for DocumentConverter.__init__."""

import logging
from types import SimpleNamespace

from src.rag.infrastructure.converters import converters
from src.rag.infrastructure.converters.converters import DocumentConverter


def test_document_converter_init_builds_default_dependencies(
    monkeypatch, tmp_path, caplog
):
    """Default constructor should wire logger, OCR, exporter, and capabilities."""

    stub_logger = logging.getLogger("doc-converter-default")
    monkeypatch.setattr(converters, "get_app_logger", lambda: stub_logger)

    created = {}

    class FakeOCR:
        def __init__(self, logger):
            created["ocr_logger"] = logger

    class FakeExporter:
        def __init__(self, logger):
            created["exporter_logger"] = logger

    class FakeDocling:
        def __init__(self):
            self.tag = "docling"

    monkeypatch.setattr(converters, "DefaultOCRService", FakeOCR)
    monkeypatch.setattr(converters, "DefaultPageExporter", FakeExporter)
    monkeypatch.setattr(converters, "_DoclingConverter", FakeDocling)
    monkeypatch.setattr(converters, "_pdf2img", object())
    monkeypatch.setattr(converters, "_tesseract", object())

    caplog.set_level(logging.INFO)
    dc = DocumentConverter(tmp_path / "in", tmp_path / "out")

    assert dc.logger is stub_logger
    assert isinstance(dc.ocr, FakeOCR)
    assert isinstance(dc.exporter, FakeExporter)
    assert isinstance(dc._converter, FakeDocling)
    assert created == {
        "ocr_logger": stub_logger,
        "exporter_logger": stub_logger,
    }
    assert "Capabilities: docling, OCR fallback" in caplog.text


def test_document_converter_init_respects_injected_dependencies(
    monkeypatch, tmp_path, caplog
):
    """Caller-provided logger/OCR/exporter should stay untouched; warn when none available."""

    monkeypatch.setattr(converters, "_DoclingConverter", None)
    monkeypatch.setattr(converters, "_pdf2img", None)
    monkeypatch.setattr(converters, "_tesseract", None)

    custom_logger = logging.getLogger("doc-converter-custom")
    dummy_ocr = SimpleNamespace()
    dummy_exporter = SimpleNamespace()

    caplog.set_level(logging.WARNING)
    dc = DocumentConverter(
        input_root=tmp_path / "inputs",
        output_root=tmp_path / "outputs",
        ocr=dummy_ocr,
        exporter=dummy_exporter,
        logger=custom_logger,
    )

    assert dc.input_root == tmp_path / "inputs"
    assert dc.output_root == tmp_path / "outputs"
    assert dc.logger is custom_logger
    assert dc.ocr is dummy_ocr
    assert dc.exporter is dummy_exporter
    assert dc._converter is None
    assert "No conversion capabilities detected" in caplog.text
