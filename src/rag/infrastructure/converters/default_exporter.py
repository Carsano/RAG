"""
Default implementation of PageExporter using pdf2image.
"""

import logging
import pathlib
from typing import List, Optional
from pdf2image import convert_from_path as _pdf2img
import pytesseract as _tesseract

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

    def _asset_dir_for(self, md_out_path: pathlib.Path) -> pathlib.Path:
        """Return the directory that will host exported page images."""
        return md_out_path.parent / f"{md_out_path.stem}_assets"

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


class DefaultOCRService:
    """OCR using pdf2image and Tesseract as a fallback backend.

    Safe to construct even if dependencies are missing. Failures return
    None and are logged.
    """

    def __init__(self, logger: logging.Logger) -> None:
        """Initialize the OCR service.

        Args:
          logger (logging.Logger): Application logger.
        """
        self.logger = logger

    def ocr_pdf(self, path: pathlib.Path) -> Optional[str]:
        """Perform OCR on a PDF file.

        Args:
          path (pathlib.Path): PDF path.

        Returns:
          Optional[str]: Extracted text or None.
        """
        try:
            pages = _pdf2img(str(path))
        except Exception as exc:
            self.logger.error("pdf2image failed for %s: %s", path, exc)
            return None
        md_pages: List[str] = []
        for idx, img in enumerate(pages, start=1):
            try:
                text = _tesseract.image_to_string(img)
            except Exception as exc:
                self.logger.error("Tesseract failed on page %s: %s", idx,
                                  exc)
                continue
            md_pages.append(f"\n\n# Page {idx}\n\n{text.strip()}\n")
        ocr_md = "".join(md_pages).strip()
        return ocr_md if ocr_md else None


__all__ = [
    "DefaultPageExporter",
    "DefaultOCRService"
]
