"""
Tests for the compare_documents CLI helper.
"""

import pathlib

import src.rag.adapters.cli.compare_documents as cli_mod
from src.rag.application.use_cases.document_index_comparator import (
    DocumentComparisonResult,
)
from src.rag.application.use_cases.document_pipeline_comparator import (
    DocumentConversionComparisonResult,
    DocumentPipelineComparisonResult,
)


def test_main_reports_comparison(monkeypatch, tmp_path, capsys) -> None:
    """CLI should report totals and diff lists using the comparator result."""
    project_root = tmp_path / "workspace"
    (project_root / "data" / "controlled_documentation").mkdir(
        parents=True
    )
    (project_root / "data" / "clean_md_database").mkdir(parents=True)
    (project_root / "data" / "indexes").mkdir(parents=True)

    conversion_result = DocumentConversionComparisonResult(
        raw_total=3,
        markdown_total=2,
        missing_markdown_sources=[pathlib.Path("/missing/doc.md")],
        orphaned_markdown_files=[],
    )
    indexing_result = DocumentComparisonResult(
        database_total=2,
        indexed_total=2,
        missing_from_index=[],
        orphaned_index_entries=[pathlib.Path("/ghost.md")],
    )
    stub_result = DocumentPipelineComparisonResult(
        conversion=conversion_result,
        indexing=indexing_result,
    )
    captured_args: dict[str, pathlib.Path] = {}

    class StubComparator:
        def __init__(self, source_root, markdown_root, sources_manifest):
            captured_args["source"] = pathlib.Path(source_root)
            captured_args["markdown"] = pathlib.Path(markdown_root)
            captured_args["manifest"] = pathlib.Path(sources_manifest)

        def compare(self):
            return stub_result

    monkeypatch.setattr(cli_mod, "get_project_root", lambda: project_root)
    monkeypatch.setattr(cli_mod, "DocumentPipelineComparator", StubComparator)

    cli_mod.main()
    output = capsys.readouterr().out

    assert captured_args["source"] == project_root / "data" / (
        "controlled_documentation"
    )
    assert captured_args["markdown"] == project_root / "data" / (
        "clean_md_database"
    )
    assert captured_args["manifest"] == project_root / "data" / "indexes" / (
        "all_chunk_sources.json"
    )
    assert "Conversion status: OUT OF SYNC" in output
    assert "/missing/doc.md" in output
    assert "Indexing status: OUT OF SYNC" in output
    assert "/ghost.md" in output
