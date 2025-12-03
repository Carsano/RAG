"""LLM-driven reranker that scores chunks via prompting."""

from typing import List

from src.rag.application.ports.llm import LLM
from src.rag.application.ports.reranker import RankedChunk, Reranker


_DEFAULT_SYSTEM_PROMPT = (
    "You are a ranking assistant. Your job is to score how relevant a "
    "given context chunk is for answering a user question.\n\n"
    "Rules:\n"
    "- Return only a numeric score between 0 and 1.\n"
    "- 0 means \"completely irrelevant\".\n"
    "- 1 means \"perfectly and directly relevant\".\n"
    "- Do not add explanations, just the number.\n"
)


class LLMReranker(Reranker):
    """Rerank candidates by asking the LLM for a relevance score.

    For each candidate chunk, the LLM receives the user question and the
    chunk content, and is asked to output a single score between 0 and 1.
    """

    def __init__(self, llm: LLM,
                 system_prompt: str = _DEFAULT_SYSTEM_PROMPT) -> None:
        """Initialize the LLM-based reranker.

        Args:
            llm: Language model used to score the candidates.
            system_prompt: Instruction given to the LLM to explain the
                scoring task.
        """
        self._llm = llm
        self._system_prompt = system_prompt

    def rank(self, query: str,
             candidates: List[dict]) -> List[RankedChunk]:
        """Rank candidates by relevance to the query.

        Args:
            query: User query string.
            candidates: List of chunk dictionaries retrieved by the
                retriever. Each dict is expected to contain a "content"
                field with the chunk text.

        Returns:
            A list of RankedChunk objects sorted by descending score.
        """
        ranked: List[RankedChunk] = []

        for candidate in candidates:
            content = candidate.get("content", "")
            score = self._score_candidate(query, content)
            ranked.append(RankedChunk(chunk=candidate, score=score))

        ranked.sort(key=lambda rc: rc.score, reverse=True)
        return ranked

    def _score_candidate(self, query: str, content: str) -> float:
        """Ask the LLM to score a single candidate.

        Args:
            query: User question string.
            content: Text content of the candidate chunk.

        Returns:
            A float score between 0 and 1. If parsing fails, returns 0.0.
        """
        messages = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": (
                    "Question:\n"
                    f"{query}\n\n"
                    "Context chunk:\n"
                    f"{content}\n\n"
                    "Score:"
                ),
            },
        ]

        raw_reply = self._llm.chat(messages)
        raw_reply = raw_reply.strip()

        try:
            score = float(raw_reply)
        except (TypeError, ValueError):
            return 0.0

        if score < 0.0:
            return 0.0
        if score > 1.0:
            return 1.0
        return score
