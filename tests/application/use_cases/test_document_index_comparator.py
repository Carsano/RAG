"""
Tests for DocumentIndexComparator.
"""

import json
import pathlib
from unittest.mock import MagicMock

from src.rag.application.use_cases.document_index_comparator import (
    DocumentIndexComparator,
)


def test_compare_detects_missing_and_orphaned(tmp_path) -> None:
    """Files absent from the manifest should be reported as missing."""
    db_root = tmp_path / "db"
    db_root.mkdir()
    file_a = db_root / "alpha.md"
    file_a.write_text("# Alpha")
    file_b = db_root / "beta.md"
    file_b.write_text("# Beta")

    manifest = tmp_path / "indexes" / "sources.json"
    manifest.parent.mkdir()
    manifest.write_text(
        json.dumps([str(file_a), str(db_root / "orphan.md")], indent=2),
        encoding="utf-8",
    )

    comparator = DocumentIndexComparator(
        database_root=db_root,
        sources_manifest=manifest,
        logger=MagicMock(),
    )

    result = comparator.compare()

    assert result.database_total == 2
    assert result.indexed_total == 2
    assert result.missing_from_index == [file_b.resolve()]
    assert result.orphaned_index_entries == [
        (db_root / "orphan.md").resolve()
    ]


def test_compare_handles_relative_manifest_paths(tmp_path) -> None:
    """Relative manifest entries should resolve to real database files."""
    project_root = tmp_path / "project"
    db_root = project_root / "data" / "clean_md_database"
    db_root.mkdir(parents=True)
    doc_path = db_root / "notes.md"
    doc_path.write_text("notes")
    manifest = project_root / "data" / "indexes" / "all_chunk_sources.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    relative_entry = pathlib.Path("..") / "clean_md_database" / doc_path.name
    manifest.write_text(json.dumps([str(relative_entry)]), encoding="utf-8")

    comparator = DocumentIndexComparator(
        database_root=db_root,
        sources_manifest=manifest,
        logger=MagicMock(),
    )

    result = comparator.compare()

    assert result.database_total == 1
    assert result.indexed_total == 1
    assert result.missing_from_index == []
    assert result.orphaned_index_entries == []
    assert result.is_in_sync
