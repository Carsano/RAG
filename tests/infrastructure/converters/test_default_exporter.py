"""Unit test for DefaultPageExporter.__init__."""

import logging

from src.rag.infrastructure.converters.default_exporter import (
    DefaultPageExporter,
)


def test_default_page_exporter_init_stores_logger():
    """Constructor should store the provided logger for later use."""
    custom_logger = logging.getLogger("default-exporter-init")

    exporter = DefaultPageExporter(logger=custom_logger)

    assert exporter.logger is custom_logger
