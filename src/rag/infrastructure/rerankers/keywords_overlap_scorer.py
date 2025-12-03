"""Simple keyword-overlap reranker used as a baseline."""

from typing import List
from src.rag.application.ports.reranker import Reranker, RankedChunk


class KeywordsOverlapScorer(Reranker):
    """Reranker using a keywords overlap scoring mechanism.

    This is a placeholder implementation. It computes a naive score based on
    keyword overlap between the query and chunk content.
    """

    def rank(self, query: str, candidates: List[dict]) -> List[RankedChunk]:
        scored = []

        query_terms = set(query.lower().split())

        for chunk in candidates:
            text = chunk.get("content", "").lower()
            text_terms = set(text.split())
            score = len(query_terms.intersection(text_terms))
            scored.append(RankedChunk(chunk=chunk, score=float(score)))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored
