"""
Intent classification service.

Decides whether a user question should use RAG or plain chat. Uses an LLM
to classify, with a keyword fallback if the LLM call fails.
"""

from src.rag.infrastructure.config.types import LLMMessage
from src.rag.application.ports.llm import LLM

SYSTEM: LLMMessage = {
            "role": "system",
            "content": (
                "Classifie en RAG ou CHAT. RAG = nécessite documents. "
                "CHAT = réponse sans recherche. Réponds par un seul mot."
            ),
        }


class IntentClassifier:
    """Classify intent as 'rag' or 'chat' using an injected LLM.

    Args:
        llm (LLM): An implementation of the LLM protocol providing chat.
    """

    def __init__(self, llm: LLM):
        self.llm = llm

    def classify(self, question: str) -> str:
        """Return 'rag' or 'chat' for the given question.

        Uses an LLM with a concise system prompt. If the LLM call raises,
        falls back to a simple keyword heuristic.

        Args:
            question (str): The input question.

        Returns:
            str: 'rag' if retrieval is needed, 'chat' otherwise.
        """

        user_msg: LLMMessage = {"role": "user", "content": question}
        out = self.llm.chat([SYSTEM, user_msg]).strip().lower()
        return "rag" if "rag" in out else "chat"
