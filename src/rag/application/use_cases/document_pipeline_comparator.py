"""
Use case orchestrating comparisons across the ingestion pipeline.

Provides visibility between the raw documentation tree, the normalized
Markdown database, and the FAISS index manifest.
"""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass
from typing import Set

from src.rag.application.use_cases.document_index_comparator import (
    DocumentComparisonResult,
    DocumentIndexComparator,
)
from src.rag.infrastructure.logging.logger import get_app_logger


@dataclass(frozen=True)
class DocumentConversionComparisonResult:
    """Comparison summary between raw documents and Markdown outputs.

    Attributes:
        raw_total: Number of raw files discovered under ``source_root``.
        markdown_total: Number of Markdown files under ``markdown_root``.
        missing_markdown_files: Expected Markdown paths without a file.
        orphaned_markdown_files: Markdown files without a source counterpart.
    """

    raw_total: int
    markdown_total: int
    missing_markdown_files: list[pathlib.Path]
    orphaned_markdown_files: list[pathlib.Path]

    @property
    def is_in_sync(self) -> bool:
        """Return True when raw documents and Markdown outputs match."""
        return (
            not self.missing_markdown_files
            and not self.orphaned_markdown_files
        )


@dataclass(frozen=True)
class DocumentPipelineComparisonResult:
    """Aggregate comparison of conversion and indexing stages.

    Attributes:
        conversion: Result of comparing raw sources to Markdown outputs.
        indexing: Result of comparing Markdown outputs to index entries.
    """

    conversion: DocumentConversionComparisonResult
    indexing: DocumentComparisonResult

    @property
    def is_in_sync(self) -> bool:
        """Return True when every stage of the pipeline is aligned."""
        return self.conversion.is_in_sync and self.indexing.is_in_sync


class DocumentPipelineComparator:
    """Compute consistency across raw -> Markdown -> index stages."""

    def __init__(
        self,
        source_root: pathlib.Path | str,
        markdown_root: pathlib.Path | str,
        sources_manifest: pathlib.Path | str,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the comparator.

        Args:
            source_root: Directory containing the original documents.
            markdown_root: Directory storing normalized Markdown files.
            sources_manifest: Json manifest listing indexed Markdown paths.
            logger: Optional application logger.
        """
        self.source_root = pathlib.Path(source_root)
        self.markdown_root = pathlib.Path(markdown_root)
        self.sources_manifest = pathlib.Path(sources_manifest)
        self.logger = logger or get_app_logger()
        self._index_comparator = DocumentIndexComparator(
            database_root=self.markdown_root,
            sources_manifest=self.sources_manifest,
            logger=self.logger,
        )

    def compare(self) -> DocumentPipelineComparisonResult:
        """Compare each stage of the ingestion pipeline.

        Returns:
            DocumentPipelineComparisonResult: Aggregated comparison data.
        """
        conversion_result = self._compare_sources_to_markdown()
        indexing_result = self._index_comparator.compare()
        return DocumentPipelineComparisonResult(
            conversion=conversion_result,
            indexing=indexing_result,
        )

    def _compare_sources_to_markdown(
        self,
    ) -> DocumentConversionComparisonResult:
        """Compare raw documents with their Markdown counterparts."""
        raw_docs = self._collect_raw_docs()
        markdown_docs = self._collect_markdown_docs()

        expected_markdown_paths = {
            self._expected_markdown_path(path) for path in raw_docs
        }
        missing_markdown = sorted(expected_markdown_paths - markdown_docs)
        orphaned_markdown = sorted(markdown_docs - expected_markdown_paths)

        return DocumentConversionComparisonResult(
            raw_total=len(raw_docs),
            markdown_total=len(markdown_docs),
            missing_markdown_files=missing_markdown,
            orphaned_markdown_files=orphaned_markdown,
        )

    def _collect_raw_docs(self) -> Set[pathlib.Path]:
        """List every raw document available under source_root."""
        if not self.source_root.exists():
            self.logger.warning(
                "Source root %s does not exist", self.source_root
            )
            return set()
        return {
            path.resolve()
            for path in self.source_root.rglob("*")
            if path.is_file()
        }

    def _collect_markdown_docs(self) -> Set[pathlib.Path]:
        """Collect Markdown files from the normalized database root."""
        if not self.markdown_root.exists():
            self.logger.warning(
                "Markdown database root %s does not exist", self.markdown_root
            )
            return set()
        return {
            path.resolve()
            for path in self.markdown_root.rglob("*.md")
            if path.is_file()
        }

    def _expected_markdown_path(self, raw_path: pathlib.Path) -> pathlib.Path:
        """Compute the Markdown path that should exist for a raw file."""
        try:
            relative = raw_path.relative_to(self.source_root)
        except ValueError:
            # Conservatively fall back to the basename when outside root.
            relative = pathlib.Path(raw_path.name)
        planned = self.markdown_root / relative
        return planned.with_suffix(".md").resolve()


__all__ = [
    "DocumentConversionComparisonResult",
    "DocumentPipelineComparisonResult",
    "DocumentPipelineComparator",
]
