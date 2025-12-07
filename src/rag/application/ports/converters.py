"""
Contracts for document conversion services.
"""

from typing import List, Optional, Protocol
import pathlib


class OCRService(Protocol):
    """OCR service contract.

    Implementations turn image-based PDFs into plain text.
    """

    def ocr_pdf(self, path: pathlib.Path) -> Optional[str]:
        """Perform OCR on a PDF file.

        Args:
          path (pathlib.Path): PDF path.

        Returns:
          Optional[str]: Extracted text or None on failure.
        """
        ...


class PageExporter(Protocol):
    """Page export contract.

    Implementations render PDF pages to image files on disk.
    """

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
        ...


class DocumentConversionService(Protocol):
    """Document conversion contract for application use cases."""

    def convert_all(self) -> List[pathlib.Path]:
        """Convert every supported document
        and return written Markdown paths."""
        ...

    def convert_paths(self, paths: List[pathlib.Path]) -> List[pathlib.Path]:
        """Convert only the given files and return written Markdown paths."""
        ...
