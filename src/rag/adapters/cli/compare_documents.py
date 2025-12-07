"""
CLI tool that compares the entire documentation pipeline.
"""

import pathlib

from src.rag.application.use_cases.document_pipeline_comparator import (
    DocumentPipelineComparator,
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
    """CLI entry point for comparing sources, Markdown, and indexed files."""
    project_root = get_project_root()
    source_root = project_root / "data" / "controlled_documentation"
    markdown_root = project_root / "data" / "clean_md_database"
    manifest = project_root / "data" / "indexes" / "all_chunk_sources.json"

    comparator = DocumentPipelineComparator(
        source_root=source_root,
        markdown_root=markdown_root,
        sources_manifest=manifest,
    )
    result = comparator.compare()

    print("=== Document Pipeline Comparison ===")
    print(f"Source root: {source_root}")
    print(f"Markdown root: {markdown_root}")
    print(f"Index manifest: {manifest}")
    print(f"Pipeline status: "
          f"{'IN SYNC' if result.is_in_sync else 'OUT OF SYNC'}")
    print()

    print("--- Conversion stage: sources -> Markdown ---")
    conversion = result.conversion
    print(f"Source documents: {conversion.raw_total}")
    print(f"Markdown documents: {conversion.markdown_total}")
    conv_status = "IN SYNC" if conversion.is_in_sync else "OUT OF SYNC"
    print(f"Conversion status: {conv_status}")
    print()

    _print_paths(
        "Source files without Markdown output",
        conversion.missing_markdown_sources,
    )
    print()
    _print_paths(
        "Markdown files without a source",
        conversion.orphaned_markdown_files,
    )
    print()

    print("--- Indexing stage: Markdown -> FAISS manifest ---")
    indexing = result.indexing
    print(f"Markdown documents: {indexing.database_total}")
    print(f"Indexed documents: {indexing.indexed_total}")
    idx_status = "IN SYNC" if indexing.is_in_sync else "OUT OF SYNC"
    print(f"Indexing status: {idx_status}")
    print()
    _print_paths("Missing from index", indexing.missing_from_index)
    print()
    _print_paths(
        "Indexed entries without a source file",
        indexing.orphaned_index_entries,
    )


if __name__ == "__main__":
    main()
