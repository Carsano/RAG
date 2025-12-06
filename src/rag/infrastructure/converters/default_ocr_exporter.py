"""
Default implementation of PageExporter using pdf2image.
"""

import logging
import pathlib
from typing import List, Optional
from pdf2image import convert_from_path as _pdf2img
import pytesseract as _tesseract

from src.rag.application.ports.converters import OCRService


class DefaultOCRService(OCRService):
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
    "DefaultOCRService"
]
