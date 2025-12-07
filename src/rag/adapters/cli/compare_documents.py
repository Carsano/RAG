"""
CLI tool that compares Markdown documents on disk versus indexed sources.
"""

import pathlib

from src.rag.application.use_cases.document_index_comparator import (
    DocumentIndexComparator,
)
from src.rag.utils.utils import get_project_root


def _print_paths(title: str, paths: list[pathlib.Path]) -> None:
    """Display a heading followed by each path on its own line.

    Args:
        title: Section heading describing the category of files.
        paths: Collection of absolute Markdown file paths.
    """
    print(f"{title}: {len(paths)}")
    if not paths:
        print("  (none)")
        return
    for path in paths:
        print(f"  - {path}")


def main() -> None:
    """CLI entry point for comparing database files to indexed sources."""
    project_root = get_project_root()
    database_root = project_root / "data" / "clean_md_database"
    manifest = project_root / "data" / "indexes" / "all_chunk_sources.json"

    comparator = DocumentIndexComparator(
        database_root=database_root, sources_manifest=manifest
    )
    result = comparator.compare()

    print("=== Document Inventory Comparison ===")
    print(f"Database root: {database_root}")
    print(f"Index manifest: {manifest}")
    print(f"Database documents: {result.database_total}")
    print(f"Indexed documents: {result.indexed_total}")
    status = "IN SYNC" if result.is_in_sync else "OUT OF SYNC"
    print(f"Status: {status}")
    print()
    _print_paths("Missing from index", result.missing_from_index)
    print()
    _print_paths("Indexed entries without a source file",
                 result.orphaned_index_entries)


if __name__ == "__main__":
    main()
