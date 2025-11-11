"""
Utility functions for the RAG project.
"""
from __future__ import annotations
import pathlib


def get_project_root() -> pathlib.Path:
    """Get the root directory of the project.

    Returns:
        pathlib.Path: Path to the project root.
    """
    root = pathlib.Path(__file__).resolve()
    while root.name != "src":
        root = root.parent
    return root.parent


__all__ = [
    "get_project_root"
]
