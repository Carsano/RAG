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


def test_ocr_pdf_concatenates_pages_and_skips_failed_tesseract(
        monkeypatch, tmp_path):
    """Happy path: build markdown with page headers,
    skip pages where OCR fails."""

    class FakeImage:
        def __init__(self, label):
            self.label = label

    pages = [FakeImage("p1"), FakeImage("p2"), FakeImage("p3")]

    monkeypatch.setattr(
        default_ocr_exporter, "_pdf2img", lambda _path: pages
    )

    def fake_image_to_string(img):
        if img.label == "p2":
            raise RuntimeError("tesseract failed")
        return f" text from {img.label} "

    monkeypatch.setattr(
        default_ocr_exporter, "_tesseract",
        type(
            "TesseractStub",
            (),
            {"image_to_string": staticmethod(fake_image_to_string)}),
    )

    ocr = DefaultOCRService(logger=logging.getLogger("ocr-test"))
    md = ocr.ocr_pdf(tmp_path / "doc.pdf")

    assert "# Page 1" in md
    assert "text from p1" in md
    assert "# Page 2" not in md
    assert "# Page 3" in md
    assert md.strip().startswith("# Page 1")
    assert md.strip().endswith("text from p3")
