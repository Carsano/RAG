"""Unit tests for DefaultPageExporter internals."""

import logging
import pathlib

from src.rag.infrastructure.converters.default_exporter import (
    DefaultPageExporter,
)


def test_default_page_exporter_init_stores_logger():
    """Constructor should store the provided logger for later use."""
    custom_logger = logging.getLogger("default-exporter-init")

    exporter = DefaultPageExporter(logger=custom_logger)

    assert exporter.logger is custom_logger


def test_asset_dir_for_uses_markdown_parent_and_stem(tmp_path):
    """_asset_dir_for should append '<stem>_assets' under the markdown parent."""
    exporter = DefaultPageExporter(logger=logging.getLogger("unused"))
    md_out_path = tmp_path / "notes" / "meeting.md"

    asset_dir = exporter._asset_dir_for(md_out_path)

    assert asset_dir == tmp_path / "notes" / "meeting_assets"
