"""
Module defining the Retriever protocol.
"""
# src/rag/application/ports/retriever.py
from typing import List, Protocol, Optional, Dict


class Retriever(Protocol):
    def search(self, query: str, k: int = 5,
               filters: Optional[Dict] = None) -> List[str]: ...
    """Retrieve relevant document chunks for a given query."""
