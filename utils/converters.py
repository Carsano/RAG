"""
Document conversion utilities.

This module converts heterogeneous document formats (PDF, DOCX, TXT, etc.)
to normalized Markdown for downstream indexing.
"""
from __future__ import annotations

import os
import pathlib
from typing import Optional


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
