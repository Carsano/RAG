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

        self._converter = _DoclingConverter() if _DoclingConverter else None

    def _convert_with_docling(self, path: pathlib.Path) -> Optional[str]:
        """Convert a file using Docling.

        Args:
            path: File to convert.

        Returns:
            The Markdown text on success, otherwise ``None`` on failure.
        """
        try:
            result = self._converter.convert(str(path))
            return result.document.export_to_markdown()
        except Exception as exc:  # pragma: no cover
            print(f"Conversion failed for {path}: {exc}")
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
            return input_path.read_text(encoding="utf-8", errors="ignore")
        if not self._converter:
            return None
        return self._convert_with_docling(input_path)

    def convert_all(self) -> List[pathlib.Path]:
        """Walk input_root, convert all supported files,
        and return list of output paths."""
        outputs: List[pathlib.Path] = []
        for root, _, files in os.walk(self.input_root):
            for name in files:
                in_path = pathlib.Path(root) / name
                content = self.convert_file(in_path)
                if content is None:
                    continue
                out_path = self.save_markdown(content, in_path,
                                              self.output_root)
                outputs.append(out_path)
        return outputs
