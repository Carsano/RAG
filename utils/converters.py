"""
Document conversion utilities.

This module converts heterogeneous document formats (PDF, DOCX, TXT, etc.)
to normalized Markdown for downstream indexing.
"""
from __future__ import annotations

import os
import pathlib
from typing import Optional

from docling.document_converter import DocumentConverter as _DoclingConverter


class BaseConverter:
    """Abstract base class for file-to-Markdown converters."""

    def convert_file(self, input_path: pathlib.Path) -> Optional[str]:
        raise NotImplementedError

    def save_markdown(self, content: str, input_path: pathlib.Path,
                      output_root: pathlib.Path) -> pathlib.Path:
        """Save Markdown content preserving relative structure."""
        relative_path = os.path.relpath(input_path, self.input_root)
        output_path = output_root / pathlib.Path(relative_path
                                                 ).with_suffix(".md")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        return output_path


class DocumentConverter(BaseConverter):
    """
    Convert all supported documents in an input directory to Markdown.

    If `docling` is installed, it will handle multi-format conversion.
    Markdown files are simply copied to the output directory.
    """

    def __init__(self, input_root: pathlib.Path,
                 output_root: pathlib.Path) -> None:
        self.input_root = pathlib.Path(input_root)
        self.output_root = pathlib.Path(output_root)

        self._converter = _DoclingConverter() if _DoclingConverter else None
