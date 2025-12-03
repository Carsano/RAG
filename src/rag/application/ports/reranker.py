"""Protocol definitions for reranking components."""

from typing import List, Protocol


class RankedChunk:
    """Container for a chunk and its relevance score."""

    def __init__(self, chunk: dict, score: float) -> None:
        self.chunk = chunk
        self.score = score


class Reranker(Protocol):
    """Protocol for reranking retrieved chunks based on a query."""

    def rank(self, query: str, candidates: List[dict]) -> List[RankedChunk]:
        """Rank candidates by relevance to the query.

        Args:
            query: User query string.
            candidates: List of chunk dictionaries retrieved by the retriever.

        Returns:
            A list of RankedChunk objects sorted by descending score.
        """
        ...
