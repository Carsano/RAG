from typing import List
from sentence_transformers import CrossEncoder
from src.rag.application.ports.reranker import RankedChunk, Reranker


class CrossEncoderReranker(Reranker):
    """Rerank candidates using a true cross-encoder model.

    This implementation loads a pretrained cross-encoder and computes
    a relevance score for each (query, chunk) pair. Scores are logits
    from the model's classification head, which reflect how relevant
    the chunk is for the given query.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ) -> None:
        """Initialize the cross-encoder reranker.

        Args:
            model_name: Name of the pretrained cross-encoder model to
                load. Must be compatible with sentence-transformers.
        """
        self._model = CrossEncoder(model_name)

    def rank(self, query: str, candidates: List[dict]) -> List[RankedChunk]:
        """Rank candidates by relevance to the query.

        Args:
            query: User query string.
            candidates: List of chunk dictionaries retrieved by the
                retriever. Each dict is expected to contain a "content"
                field with the chunk text.

        Returns:
            A list of RankedChunk objects sorted by descending score.
        """
        pairs = []
        for candidate in candidates:
            text = candidate.get("content", "")
            pairs.append((query, text))

        scores = self._model.predict(pairs)

        ranked_chunks: List[RankedChunk] = []
        for candidate, score in zip(candidates, scores):
            ranked_chunks.append(
                RankedChunk(chunk=candidate, score=float(score))
            )

        ranked_chunks.sort(key=lambda rc: rc.score, reverse=True)
        return ranked_chunks
