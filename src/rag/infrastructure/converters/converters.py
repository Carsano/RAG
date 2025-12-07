"""
Document conversion utilities.

Converts heterogeneous document formats (PDF, DOCX, TXT, etc.) to
normalized Markdown for downstream indexing. The design follows SOLID
principles with small, replaceable components for OCR and page export.
"""

import logging
import os
import pathlib
from typing import List, Optional
from docling.document_converter import DocumentConverter as _DoclingConverter
from pdf2image import convert_from_path as _pdf2img
import pytesseract as _tesseract
import re

from src.rag.application.ports.converters import (
    OCRService,
    PageExporter,
    DocumentConversionService,
)

from src.rag.infrastructure.logging.logger import get_app_logger
from src.rag.infrastructure.converters.default_page_exporter import (
    DefaultPageExporter
)
from src.rag.infrastructure.converters.default_ocr_exporter import (
    DefaultOCRService
)


class DocumentConverter(DocumentConversionService):
    """Convert documents in a directory tree to Markdown.

    Uses docling for multi-format conversion when available and copies
    existing Markdown files as-is. Unsupported files are skipped silently.

    Dependencies such as OCR and page export are injected to follow SOLID.
    """

    def __init__(
        self,
        input_root: pathlib.Path,
        output_root: pathlib.Path,
        ocr: Optional[OCRService] = None,
        exporter: Optional[PageExporter] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize the converter.

        Args:
          input_root (pathlib.Path): Root directory to scan for input files.
          output_root (pathlib.Path): Root directory where converted Markdown
            files are written, preserving input_root structure.
          ocr (Optional[OCRService]): OCR service implementation.
          exporter (Optional[PageExporter]): PDF page exporter implementation.
          logger (Optional[logging.Logger]): Application logger.
        """
        self.input_root = pathlib.Path(input_root)
        self.output_root = pathlib.Path(output_root)
        self.logger = logger or get_app_logger()
        self.ocr = ocr or DefaultOCRService(self.logger)
        self.exporter = exporter or DefaultPageExporter(self.logger)

        self._converter = _DoclingConverter() if _DoclingConverter else None
        self.logger.info(
            f"DocumentConverter initialized | input_root={self.input_root} "
            f"| output_root={self.output_root}"
        )
        capabilities = []
        if self._converter:
            capabilities.append("docling")
        if _pdf2img and _tesseract:
            capabilities.append("OCR fallback")
        if capabilities:
            self.logger.info(f"Capabilities: {', '.join(capabilities)}")
        else:
            self.logger.warning("No conversion capabilities detected")

    def _text_density_low(self, text: str, min_chars: int = 600,
                          min_words: int = 120) -> bool:
        """Detect near-empty or image-only extraction results.

        Args:
          text (str): Text to analyze.
          min_chars (int): Minimum alphanumeric characters threshold.
          min_words (int): Minimum word count threshold.

        Returns:
          bool: True if text likely lacks true content.
        """
        if not text:
            return True
        alnum = len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]", text))
        words = len(re.findall(r"\b\w+\b", text))
        if alnum < min_chars or words < min_words:
            return True
        return False

    def _replace_image_placeholders(self, md: str,
                                    md_out_path: pathlib.Path,
                                    images: list[pathlib.Path]) -> str:
        """Replace '<!-- image -->' occurrences with Markdown image links.

        Args:
          md (str): Markdown text containing placeholders.
          md_out_path (pathlib.Path): Path to the Markdown output file.
          images (List[pathlib.Path]): List of page image paths.

        Returns:
          str: Markdown text with image links inserted.
        """
        if not images:
            return md
        rel_links = [
            os.path.relpath(p, md_out_path.parent)
            for p in images
        ]
        placeholder = "<!-- image -->"
        parts = md.split(placeholder)
        if len(parts) == 1:
            return md
        rebuilt = []
        for idx, part in enumerate(parts):
            rebuilt.append(part)
            if idx < len(parts) - 1:
                link_idx = min(idx, len(rel_links) - 1)
                rebuilt.append(f"![page {link_idx+1}]({rel_links[link_idx]})")
        return "".join(rebuilt)

    def _planned_md_path(self, input_path: pathlib.Path) -> pathlib.Path:
        """Compute the Markdown path that save_markdown() will write to.

        Args:
          input_path (pathlib.Path): Original input file path.

        Returns:
          pathlib.Path: Planned Markdown output path.
        """
        relative_path = os.path.relpath(input_path, self.input_root)
        return self.output_root / pathlib.Path(relative_path
                                               ).with_suffix(".md")

    def _convert_with_docling(self, path: pathlib.Path) -> Optional[str]:
        """Convert a file using Docling.

        Args:
          path (pathlib.Path): File to convert.

        Returns:
          Optional[str]: Markdown text on success, None on failure.
        """
        try:
            result = self._converter.convert(str(path))
            md = result.document.export_to_markdown()
            if path.suffix.lower() == ".pdf" and self._text_density_low(md):
                self.logger.info(f"Low text density detected in {path.name}; "
                                 "attempting OCR fallback")
                ocr_md = self.ocr.ocr_pdf(path)
                if ocr_md and not self._text_density_low(ocr_md,
                                                         min_chars=300,
                                                         min_words=60):
                    self.logger.info(f"OCR fallback succeeded for {path.name}")
                    return ocr_md
                else:
                    self.logger.warning(
                        f"OCR fallback yielded low text or failed for "
                        f"{path.name}; keeping Docling output"
                    )
            if path.suffix.lower() == ".pdf" and "<!-- image -->" in md:
                md_out_path = self._planned_md_path(path)
                page_imgs = self.exporter.export_pages(path, md_out_path)
                if page_imgs:
                    before = md
                    md = self._replace_image_placeholders(md,
                                                          md_out_path,
                                                          page_imgs)
                    if md != before:
                        self.logger.info(
                            f"Replaced image placeholders with "
                            f"{len(page_imgs)} page images for {path.name}"
                        )
                else:
                    self.logger.warning(
                        f"No page images exported; leaving placeholders in "
                        f"{path.name}"
                    )
            return md
        except Exception as exc:
            self.logger.error(f"Conversion failed for {path}: {exc}")
            if path.suffix.lower() == ".pdf":
                ocr_md = self.ocr.ocr_pdf(path)
                if ocr_md:
                    md_out_path = self._planned_md_path(path)
                    page_imgs = self.exporter.export_pages(path, md_out_path)
                    if page_imgs:
                        ocr_md = self._replace_image_placeholders(ocr_md,
                                                                  md_out_path,
                                                                  page_imgs)
                return ocr_md
            return None

    def convert_file(self, input_path: pathlib.Path) -> Optional[str]:
        """Convert or copy a single file to Markdown.

        If the file is already Markdown, the text is returned directly. If
        Docling is not available, non-Markdown files are skipped.

        Args:
          input_path (pathlib.Path): Path to the input file.

        Returns:
          Optional[str]: Markdown text if conversion or copy succeeds,
            otherwise None.
        """
        ext = input_path.suffix.lower()
        if ext == ".md":
            self.logger.info(f"Copying Markdown file: {input_path}")
            return input_path.read_text(encoding="utf-8", errors="ignore")
        if not self._converter:
            self.logger.warning(
                f"Skipping non-Markdown without docling: {input_path}"
            )
            return None
        return self._convert_with_docling(input_path)

    def convert_all(self) -> List[pathlib.Path]:
        """Convert an entire directory tree to Markdown.

        Walks input_root, converts each supported file, and writes Markdown
        under output_root preserving relative structure. Existing Markdown
        files are copied.

        Returns:
          List[pathlib.Path]: Paths to the written Markdown files.
        """
        outputs: List[pathlib.Path] = []
        copied = 0
        converted = 0
        skipped = 0
        if not self.input_root.exists():
            self.logger.error(f"Input directory not found: {self.input_root}")
            return outputs

        self.logger.info(f"Starting conversion walk: {self.input_root}")
        for root, _dirs, files in os.walk(self.input_root):
            for name in files:
                in_path = pathlib.Path(root) / name
                content = self.convert_file(in_path)
                if content is None:
                    skipped += 1
                    continue
                out_path = self.save_markdown(content, in_path,
                                              self.output_root)
                outputs.append(out_path)
                if in_path.suffix.lower() == ".md":
                    copied += 1
                else:
                    converted += 1
        self.logger.info(
            f"Conversion summary | total={copied + converted + skipped}"
            f" | converted={converted}"
            f" | copied={copied} | skipped={skipped}"
        )
        return outputs

    def save_markdown(
        self,
        content: str,
        input_path: pathlib.Path,
        output_root: pathlib.Path,
    ) -> pathlib.Path:
        """Save Markdown content while mirroring the input tree.

        The relative path from the converter's input_root to input_path is
        preserved under output_root with the extension replaced by .md.

        Args:
          content (str): Markdown content to write.
          input_path (pathlib.Path): Path to the original input file.
          output_root (pathlib.Path): Root directory to store Markdown files.

        Returns:
          pathlib.Path: The full path of the written Markdown file.
        """
        relative_path = os.path.relpath(input_path, self.input_root)
        output_path = output_root / pathlib.Path(relative_path
                                                 ).with_suffix(".md")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        return output_path
