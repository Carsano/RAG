"""
Default implementation of PageExporter using pdf2image.
"""

from __future__ import annotations
import logging
import pathlib
from typing import List
from pdf2image import convert_from_path as _pdf2img
from src.rag.application.ports.converters import PageExporter


class DefaultPageExporter(PageExporter):
    """Export PDF pages to PNG files using pdf2image.

    Safe to construct even if dependencies are missing. Failures return
    an empty list and are logged.
    """

    def __init__(self, logger: logging.Logger) -> None:
        """Initialize the exporter.

        Args:
          logger (logging.Logger): Application logger.
        """
        self.logger = logger

    def export_pages(
        self, pdf_path: pathlib.Path, md_out_path: pathlib.Path
    ) -> List[pathlib.Path]:
        """Export PDF pages as images.

        Args:
          pdf_path (pathlib.Path): Input PDF.
          md_out_path (pathlib.Path): Planned Markdown output path.

        Returns:
          List[pathlib.Path]: Paths to the rendered page images.
        """
        if _pdf2img is None:
            self.logger.warning("Cannot export pages: pdf2image missing")
            return []
        asset_dir = md_out_path.parent / f"{md_out_path.stem}_assets"
        asset_dir.mkdir(parents=True, exist_ok=True)
        try:
            pages = _pdf2img(str(pdf_path))
        except Exception as exc:
            self.logger.error("pdf2image failed for %s: %s", pdf_path, exc)
            return []
        out_paths: List[pathlib.Path] = []
        for i, img in enumerate(pages, start=1):
            out_path = asset_dir / f"page_{i:03d}.png"
            try:
                img.save(out_path)
                out_paths.append(out_path)
            except Exception as exc:
                self.logger.error("Saving page image failed: %s | %s",
                                  out_path, exc)
        return out_paths
