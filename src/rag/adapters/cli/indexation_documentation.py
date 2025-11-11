"""
CLI for manuel indexing of documentation.
"""
from __future__ import annotations

import pathlib
from src.rag.application.use_cases.documents_indexer import DocumentsIndexer
from src.rag.utils.utils import get_project_root


def main() -> None:
    """
    Entry point for a manual indexing run.

    Uses the `clean_md_database` folder as root and builds the index.
    """
    ROOT = get_project_root() / "data" / "clean_md_database"
    indexer = DocumentsIndexer(root=ROOT)
    indexer.build()
    removed = indexer.remove_file(
        pathlib.Path("data/clean_md_database/budget/budget_2024.md"))
    indexer.logger.info(f"Removed {removed}")


if __name__ == "__main__":
    main()
