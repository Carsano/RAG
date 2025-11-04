"""
Document conversion utilities.

This module converts heterogeneous document formats (PDF, DOCX, TXT, etc.)
to normalized Markdown for downstream indexing.
"""
from __future__ import annotations

import os
import pathlib
from typing import List, Optional

from docling.document_converter import DocumentConverter as _DoclingConverter
from utils.logger import Logger

# Optional OCR fallback for image-only PDFs

from pdf2image import convert_from_path as _pdf2img
import pytesseract as _tesseract
import re


class BaseConverter:
    """Abstract base class for file-to-Markdown converters.

    Subclasses should implement concrete conversion strategies but keep the
    public interface stable. The base class also provides a helper to persist
    converted Markdown while preserving the relative directory layout.
    """

    def convert_file(self, input_path: pathlib.Path) -> Optional[str]:
        """Convert a single file to Markdown text.

        Args:
            input_path: Absolute or relative path to the input file.

        Returns:
            The Markdown text if the format is supported and conversion
            succeeds. Returns ``None`` if the file type is unsupported or if
            conversion fails gracefully.
        """
        raise NotImplementedError

    def save_markdown(
        self,
        content: str,
        input_path: pathlib.Path,
        output_root: pathlib.Path,
    ) -> pathlib.Path:
        """Save Markdown content while mirroring the input tree.

        The relative path from the converter's ``input_root`` to ``input_path``
        is preserved under ``output_root`` and the file extension is replaced
        with ``.md``.

        Args:
            content: Markdown content to write.
            input_path: Path to the original input file.
            output_root: Root directory where Markdown artifacts are stored.

        Returns:
            The full path of the written Markdown file.
        """
        relative_path = os.path.relpath(input_path, self.input_root)
        output_path = output_root / pathlib.Path(relative_path
                                                 ).with_suffix(".md")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        return output_path


class DocumentConverter(BaseConverter):
    """Convert documents in a directory tree to Markdown.

    Uses ``docling`` for multi-format conversion when available and copies
    existing Markdown files as-is. Unsupported files are skipped silently.

    Attributes:
        input_root: Root directory containing the source files.
        output_root: Root directory where Markdown outputs will be written.
        _converter: Internal Docling converter instance.
    """

    def __init__(self, input_root: pathlib.Path,
                 output_root: pathlib.Path) -> None:
        """Initialize the converter.

        Args:
            input_root: Root directory to scan for input files.
            output_root: Root directory where converted Markdown files are
                written, preserving the relative structure of ``input_root``.
        """
        self.input_root = pathlib.Path(input_root)
        self.output_root = pathlib.Path(output_root)
        self.logger = Logger(name="converters")

        self._converter = _DoclingConverter() if _DoclingConverter else None
        self.logger.info(
            f"Initialized DocumentConverter | input_root={self.input_root}"
            f" | output_root={self.output_root}"
        )
        self.logger.info("OCR fallback available: pdf2image + "
                         "Tesseract detected")

    def _text_density_low(self, text: str, min_chars: int = 600,
                          min_words: int = 120) -> bool:
        """Heuristic: detect near-empty or image-only extraction results.

        Returns True when Markdown likely lacks true text.
        """
        if not text:
            return True
        # Count alphanumeric characters and words
        alnum = len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]", text))
        words = len(re.findall(r"\b\w+\b", text))
        # Very low signal or huge image markdown with few words
        if alnum < min_chars or words < min_words:
            return True
        return False

    def _ocr_pdf(self, path: pathlib.Path) -> Optional[str]:
        """Fallback OCR for image-only PDFs using pdf2image + Tesseract.

        Returns Markdown text or None when OCR backend is unavailable.
        """
        try:
            pages = _pdf2img(str(path))
            md_pages = []
            for idx, img in enumerate(pages, start=1):
                text = _tesseract.image_to_string(img)
                md_pages.append(f"\n\n# Page {idx}\n\n{text.strip()}\n")
            ocr_md = "".join(md_pages).strip()
            return ocr_md if ocr_md else None
        except Exception as exc:  # pragma: no cover
            self.logger.error(f"OCR fallback failed for {path}: {exc}")
            return None

    def _convert_with_docling(self, path: pathlib.Path) -> Optional[str]:
        """Convert a file using Docling.

        Args:
            path: File to convert.

        Returns:
            The Markdown text on success, otherwise ``None`` on failure.
        """
        try:
            result = self._converter.convert(str(path))
            md = result.document.export_to_markdown()
            # If PDF and text density is low, try OCR fallback
            if path.suffix.lower() == ".pdf" and self._text_density_low(md):
                self.logger.info(f"Low text density detected in {path.name}; "
                                 f"attempting OCR fallback")
                ocr_md = self._ocr_pdf(path)
                if ocr_md and not self._text_density_low(ocr_md,
                                                         min_chars=300,
                                                         min_words=60):
                    self.logger.info(f"OCR fallback succeeded for {path.name}")
                    return ocr_md
                else:
                    self.logger.warning(f"OCR fallback yielded low text or "
                                        f"failed for {path.name}; "
                                        f"keeping Docling output")
            return md
        except Exception as exc:  # pragma: no cover
            self.logger.error(f"Conversion failed for {path}: {exc}")
            if path.suffix.lower() == ".pdf":
                return self._ocr_pdf(path)
            return None

    def convert_file(self, input_path: pathlib.Path) -> Optional[str]:
        """Convert or copy a single file to Markdown.

        If the file is already Markdown, the text is returned directly. If the
        Docling backend is not available, non-Markdown files are skipped.

        Args:
            input_path: Path to the input file.

        Returns:
            The Markdown text if conversion or copy succeeds, otherwise
            ``None``.
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

        Walks ``input_root``, converts each supported file, and writes a
        Markdown artifact under ``output_root`` while preserving the relative
        structure. Existing Markdown files are copied.

        Returns:
            A list of paths to the written Markdown files.
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
            f"| copied={copied} | skipped={skipped}"
        )
        return outputs


if __name__ == "__main__":
    converter = DocumentConverter(
        input_root=pathlib.Path("./data/controlled_documentation"),
        output_root=pathlib.Path("./data/clean_md_database"),
    )
    written = converter.convert_all()
    Logger(name="converters").info(
        f"Converted {len(written)} documents to Markdown"
        f" → {pathlib.Path("./data/clean_md_database").resolve()}"
    )
