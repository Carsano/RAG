"""Tests for DefaultOCRService."""

import logging

from src.rag.infrastructure.converters import default_ocr_exporter
from src.rag.infrastructure.converters.default_ocr_exporter import (
    DefaultOCRService,
)


def test_ocr_pdf_returns_none_when_pdf2image_fails(
        monkeypatch, tmp_path, caplog):
    """Should log and return None when pdf2image raises."""

    def raise_error(_path):
        raise RuntimeError("conversion failed")

    monkeypatch.setattr(
        default_ocr_exporter, "_pdf2img", raise_error,
    )
    ocr = DefaultOCRService(logger=logging.getLogger("ocr-test"))

    caplog.set_level(logging.ERROR)
    result = ocr.ocr_pdf(tmp_path / "file.pdf")

    assert result is None
    assert "pdf2image failed" in caplog.text
