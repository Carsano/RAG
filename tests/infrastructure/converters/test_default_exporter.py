"""Unit tests for DefaultPageExporter internals."""

import logging
import pathlib

from src.rag.infrastructure.converters.default_exporter import (
    DefaultPageExporter,
)

###################################
# Unit tests: DefaultPageExporter #
###################################


def test_default_page_exporter_init_stores_logger():
    """Constructor should store the provided logger for later use."""
    custom_logger = logging.getLogger("default-exporter-init")

    exporter = DefaultPageExporter(logger=custom_logger)

    assert exporter.logger is custom_logger


def test_asset_dir_for_uses_markdown_parent_and_stem(tmp_path):
    """_asset_dir_for should append '<stem>_assets'
    under the markdown parent."""
    exporter = DefaultPageExporter(logger=logging.getLogger("unused"))
    md_out_path = tmp_path / "notes" / "meeting.md"

    asset_dir = exporter._asset_dir_for(md_out_path)

    assert asset_dir == tmp_path / "notes" / "meeting_assets"


def test_render_pdf_invokes_pdf2image(monkeypatch):
    """_render_pdf should delegate to pdf2image with the stringified path."""
    exporter = DefaultPageExporter(logger=logging.getLogger("unused"))

    called_with = {}

    def fake_pdf2img(path):
        called_with["path"] = path
        return ["page1", "page2"]

    monkeypatch.setattr(
        "src.rag.infrastructure.converters.default_exporter._pdf2img",
        fake_pdf2img,
    )

    pdf_path = pathlib.Path("/tmp/input.pdf")
    result = exporter._render_pdf(pdf_path)

    assert called_with["path"] == str(pdf_path)
    assert result == ["page1", "page2"]


def test_export_pages_returns_empty_when_pdf2image_missing(
    monkeypatch, tmp_path, caplog
):
    """export_pages should warn and return [] if pdf2image is unavailable."""
    monkeypatch.setattr(
        "src.rag.infrastructure.converters.default_exporter._pdf2img", None
    )
    exporter = DefaultPageExporter(logger=logging.getLogger("unused"))

    caplog.set_level(logging.WARNING)
    result = exporter.export_pages(
        pdf_path=tmp_path / "input.pdf",
        md_out_path=tmp_path / "doc.md",
    )

    assert result == []
    assert "pdf2image missing" in caplog.text


def test_export_pages_returns_empty_when_render_provides_no_pages(
    monkeypatch, tmp_path
):
    """When _render_pdf returns nothing the exporter should
    not attempt to save."""

    def fake_render(self, _pdf_path):
        return []

    def fake_save(self, pages, asset_dir):
        raise AssertionError("save should not be called when no pages")

    monkeypatch.setattr(
        "src.rag.infrastructure.converters.default_exporter._pdf2img",
        lambda path: ["placeholder"],
    )
    monkeypatch.setattr(DefaultPageExporter, "_render_pdf", fake_render)
    monkeypatch.setattr(DefaultPageExporter, "_save_pages", fake_save)

    exporter = DefaultPageExporter(logger=logging.getLogger("unused"))
    md_out = tmp_path / "notes" / "meeting.md"
    result = exporter.export_pages(pdf_path=tmp_path / "meeting.pdf",
                                   md_out_path=md_out)

    assert result == []
    assert (tmp_path / "notes" / "meeting_assets").exists()


def test_export_pages_creates_assets_and_returns_saved_paths(
    monkeypatch, tmp_path
):
    """Successful export should create the assets dir and return page paths."""
    saved = []

    class FakeImage:
        def __init__(self, label: str):
            self.label = label

        def save(self, path):
            path.write_text(f"png:{self.label}")
            saved.append(path.name)

    monkeypatch.setattr(
        "src.rag.infrastructure.converters.default_exporter._pdf2img",
        lambda path: [FakeImage("p1"), FakeImage("p2")],
    )

    exporter = DefaultPageExporter(logger=logging.getLogger("unused"))
    md_out = tmp_path / "exports" / "doc.md"
    result = exporter.export_pages(pdf_path=tmp_path / "doc.pdf",
                                   md_out_path=md_out)

    assets_dir = tmp_path / "exports" / "doc_assets"
    expected = [
        assets_dir / "page_001.png",
        assets_dir / "page_002.png",
    ]
    assert result == expected
    assert saved == ["page_001.png", "page_002.png"]
    for path in expected:
        assert path.exists()


def test_save_pages_persists_successful_images_and_logs_errors(
        tmp_path,
        caplog
        ):
    """_save_pages should write PNGs for successful pages
    while logging failures."""
    exporter = DefaultPageExporter(logger=logging.getLogger("unused"))
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()

    saved = []

    class FakeImage:
        def __init__(self, label: str, fail: bool = False):
            self.label = label
            self.fail = fail

        def save(self, path):
            if self.fail:
                raise RuntimeError("disk full")
            path.write_text(f"png:{self.label}")
            saved.append(path.name)

    pages = [
        FakeImage("first"),
        FakeImage("second", fail=True),
        FakeImage("third"),
    ]

    caplog.set_level(logging.ERROR)
    result_paths = exporter._save_pages(pages, asset_dir)

    expected = [
        asset_dir / "page_001.png",
        asset_dir / "page_003.png",
    ]
    assert result_paths == expected
    assert saved == ["page_001.png", "page_003.png"]
    assert "Saving page image failed" in caplog.text


##########################################
# Integration tests: DefaultPageExporter #
##########################################

def test_export_pages_handles_pdf2image_exception(
        monkeypatch, tmp_path, caplog):
    """Integration-style: ensure pdf2image errors are logged
    and result is []."""

    def _raise_error(path):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "src.rag.infrastructure.converters.default_exporter._pdf2img",
        _raise_error,
    )

    exporter = DefaultPageExporter(
        logger=logging.getLogger("default-exporter"))
    caplog.set_level(logging.ERROR)
    md_out = tmp_path / "reports" / "session.md"

    result = exporter.export_pages(pdf_path=tmp_path / "session.pdf",
                                   md_out_path=md_out)

    assert result == []
    assert "pdf2image failed" in caplog.text
