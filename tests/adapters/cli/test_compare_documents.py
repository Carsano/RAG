"""
Tests for the compare_documents CLI helper.
"""

import pathlib

import src.rag.adapters.cli.compare_documents as cli_mod
from src.rag.application.use_cases.document_index_comparator import (
    DocumentComparisonResult,
)


def test_main_reports_comparison(monkeypatch, tmp_path, capsys) -> None:
    """CLI should report totals and diff lists using the comparator result."""
    project_root = tmp_path / "workspace"
    (project_root / "data" / "clean_md_database").mkdir(parents=True)
    (project_root / "data" / "indexes").mkdir(parents=True)

    stub_result = DocumentComparisonResult(
        database_total=3,
        indexed_total=2,
        missing_from_index=[pathlib.Path("/missing/doc.md")],
        orphaned_index_entries=[],
    )
    captured_args: dict[str, pathlib.Path] = {}

    class StubComparator:
        def __init__(self, database_root, sources_manifest):
            captured_args["db"] = pathlib.Path(database_root)
            captured_args["manifest"] = pathlib.Path(sources_manifest)

        def compare(self):
            return stub_result

    monkeypatch.setattr(cli_mod, "get_project_root", lambda: project_root)
    monkeypatch.setattr(cli_mod, "DocumentIndexComparator", StubComparator)

    cli_mod.main()
    output = capsys.readouterr().out

    assert captured_args["db"] == project_root / "data" / "clean_md_database"
    assert captured_args["manifest"] == project_root / "data" / "indexes" / (
        "all_chunk_sources.json"
    )
    assert "Database documents: 3" in output
    assert "Indexed documents: 2" in output
    assert "Status: OUT OF SYNC" in output
    assert "/missing/doc.md" in output
