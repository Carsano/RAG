"""
Tests for DocumentPipelineComparator.
"""

import json
from unittest.mock import MagicMock

from src.rag.application.use_cases.document_pipeline_comparator import (
    DocumentPipelineComparator,
)


def test_compare_detects_conversion_and_indexing_gaps(tmp_path) -> None:
    """Comparator reports inconsistencies across every stage."""
    source_root = tmp_path / "controlled_documentation"
    markdown_root = tmp_path / "clean_md_database"
    index_root = tmp_path / "indexes"
    source_root.mkdir()
    markdown_root.mkdir()
    index_root.mkdir()

    raw_a = source_root / "team" / "charter.docx"
    raw_b = source_root / "policies" / "security.pdf"
    raw_a.parent.mkdir(parents=True, exist_ok=True)
    raw_b.parent.mkdir(parents=True, exist_ok=True)
    raw_a.write_text("doc")
    raw_b.write_text("pdf")

    converted_a = markdown_root / "team" / "charter.md"
    converted_extra = markdown_root / "misc" / "extra.md"
    converted_a.parent.mkdir(parents=True, exist_ok=True)
    converted_extra.parent.mkdir(parents=True, exist_ok=True)
    converted_a.write_text("alpha")
    converted_extra.write_text("orphan")

    manifest = index_root / "all_chunk_sources.json"
    manifest.write_text(
        json.dumps(
            [str(converted_a),
             str(markdown_root / "team" / "ghost.md")]),
        encoding="utf-8",
    )

    comparator = DocumentPipelineComparator(
        source_root=source_root,
        markdown_root=markdown_root,
        sources_manifest=manifest,
        logger=MagicMock(),
    )

    result = comparator.compare()

    assert result.conversion.raw_total == 2
    assert result.conversion.markdown_total == 2
    assert result.conversion.missing_markdown_files == [
        (markdown_root / "policies" / "security.md").resolve()
    ]
    assert result.conversion.orphaned_markdown_files == [
        converted_extra.resolve()
    ]

    assert result.indexing.database_total == 2
    assert result.indexing.indexed_total == 2
    assert result.indexing.missing_from_index == [
        converted_extra.resolve()
    ]
    assert result.indexing.orphaned_index_entries == [
        (markdown_root / "team" / "ghost.md").resolve()
    ]
    assert not result.is_in_sync


def test_compare_in_sync_pipeline(tmp_path) -> None:
    """Comparator reports a healthy pipeline when everything matches."""
    source_root = tmp_path / "controlled_documentation"
    markdown_root = tmp_path / "clean_md_database"
    index_root = tmp_path / "indexes"
    (source_root / "guides").mkdir(parents=True, exist_ok=True)
    (markdown_root / "guides").mkdir(parents=True, exist_ok=True)
    index_root.mkdir()

    raw_doc = source_root / "guides" / "intro.pdf"
    raw_doc.write_text("pdf")
    converted_doc = markdown_root / "guides" / "intro.md"
    converted_doc.write_text("md")

    manifest = index_root / "all_chunk_sources.json"
    manifest.write_text(json.dumps([str(converted_doc)]), encoding="utf-8")

    comparator = DocumentPipelineComparator(
        source_root=source_root,
        markdown_root=markdown_root,
        sources_manifest=manifest,
        logger=MagicMock(),
    )

    result = comparator.compare()

    assert result.conversion.is_in_sync
    assert result.indexing.is_in_sync
    assert result.is_in_sync
