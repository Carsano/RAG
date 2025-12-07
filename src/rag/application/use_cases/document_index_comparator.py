"""
Use case helpers to compare database Markdown files with indexed sources.
"""

from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass
from typing import List, Set

from src.rag.infrastructure.logging.logger import get_app_logger


@dataclass(frozen=True)
class DocumentComparisonResult:
    """Comparison summary between Markdown database and indexed sources.

    Attributes:
        database_total: Number of Markdown files found under the database root.
        indexed_total: Number of unique source files referenced by the index.
        missing_from_index: Files present on disk but absent from the index.
        orphaned_index_entries: Manifest items pointing to missing files.
    """

    database_total: int
    indexed_total: int
    missing_from_index: List[pathlib.Path]
    orphaned_index_entries: List[pathlib.Path]

    @property
    def is_in_sync(self) -> bool:
        """Indicate whether database files and index entries match.

        Returns:
            bool: True when both document sets are identical, False otherwise.
        """
        return not self.missing_from_index and not self.orphaned_index_entries


class DocumentIndexComparator:
    """Compare Markdown files stored on disk against the index manifest."""

    def __init__(
        self,
        database_root: pathlib.Path | str,
        sources_manifest: pathlib.Path | str,
        logger: logging.Logger | None = None,
    ) -> None:
        self.database_root = pathlib.Path(database_root)
        self.sources_manifest = pathlib.Path(sources_manifest)
        self.logger = logger or get_app_logger()

    def compare(self) -> DocumentComparisonResult:
        """Compute differences between Markdown files and indexed entries.

        Returns:
            DocumentComparisonResult: Aggregated totals and diff lists.
        """
        database_docs = self._collect_database_docs()
        indexed_docs = self._collect_index_docs()
        missing_from_index = sorted(database_docs - indexed_docs)
        orphaned_index_entries = sorted(indexed_docs - database_docs)
        return DocumentComparisonResult(
            database_total=len(database_docs),
            indexed_total=len(indexed_docs),
            missing_from_index=missing_from_index,
            orphaned_index_entries=orphaned_index_entries,
        )

    def _collect_database_docs(self) -> Set[pathlib.Path]:
        """List Markdown files available in the database directory.

        Returns:
            Set[pathlib.Path]: Absolute file paths of on-disk Markdown files.
        """
        if not self.database_root.exists():
            self.logger.warning(
                "Database root %s does not exist", self.database_root
            )
            return set()
        return {
            path.resolve()
            for path in self.database_root.rglob("*.md")
            if path.is_file()
        }

    def _collect_index_docs(self) -> Set[pathlib.Path]:
        """List Markdown files referenced by the FAISS index manifest.

        Returns:
            Set[pathlib.Path]: Absolute source paths mentioned in the manifest.
        """
        if not self.sources_manifest.exists():
            self.logger.warning(
                "Index sources manifest %s not found", self.sources_manifest
            )
            return set()
        try:
            data = json.loads(
                self.sources_manifest.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            self.logger.error(
                "Unable to parse %s: %s", self.sources_manifest, exc
            )
            return set()

        normalized: Set[pathlib.Path] = set()
        for raw_path in data:
            if not raw_path:
                continue
            normalized_path = self._normalize_index_path(str(raw_path))
            normalized.add(normalized_path)
        return normalized

    def _normalize_index_path(self, raw: str) -> pathlib.Path:
        """Normalize manifest entries so both sources share the same form.

        Args:
            raw: Path string as stored in ``all_chunk_sources.json``.

        Returns:
            pathlib.Path: Resolved absolute path without requiring existence.
        """
        candidate = pathlib.Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = self.sources_manifest.parent / candidate
        try:
            return candidate.resolve(strict=False)
        except Exception:
            # Path resolution should not break the report.
            return candidate


__all__ = [
    "DocumentComparisonResult",
    "DocumentIndexComparator",
]
