"""
RAG chat service.

Provides utilities to manage chat interactions that use
Retrieval-Augmented Generation (RAG).
"""
from __future__ import annotations
from typing import List, Optional

from src.rag.application.ports.llm import LLM
from src.rag.application.ports.retriever import Retriever
from src.rag.application.use_cases.prompting import build_system_prompt
from src.rag.application.use_cases.prompting import clamp_dialog
from src.rag.application.use_cases.intent_classifier import IntentClassifier
from src.rag.infrastructure.logging.interaction_logger import InteractionLogger


class RAGChatService:
    """
    Manage chat with Retrieval-Augmented Generation.

    Embeds questions, retrieves relevant chunks, classifies intent, and
    produces answers with a language model.
    """

    def __init__(self, llm: LLM, retriever: Retriever,
                 base_system_prompt: str, intent_classifier: IntentClassifier,
                 interaction_logger: Optional[InteractionLogger] = None):
        """Initialize the service.

        Args:
          llm (LLM): Interface to the language model used to generate replies.
          retriever (Retriever): Component that retrieves relevant chunks.
          base_system_prompt (str): Base system prompt for the model.
          intent_classifier: Component that classifies user intent.
          interaction_logger (Optional[InteractionLogger]): Logger for
          persisting question, answer, contexts, and ground_truth.
        """
        self.llm = llm
        self.retriever = retriever
        self.base_system = base_system_prompt
        self.intent_classifier = intent_classifier
        self.interaction_logger = interaction_logger

    def answer(self, history: List[dict], question: str) -> dict:
        """Generate an answer using history and optional retrieval.

        The method classifies intent. If the intent is "rag", it retrieves
        chunks. It then builds the system prompt, clamps dialog history, and
        asks the LLM for a reply.

        Args:
          history (List[dict]): Prior messages in the conversation.
          question (str): Current user question.

        Returns:
          dict: The model's answer and sources.
        """
        intent = self.intent_classifier.classify(question)
        retrievings = self.retriever.retrieve(question,
                                              k=5) if intent == "rag" else []
        chunks = retrievings.get("chunks")
        sources = retrievings.get("sources")
        system = {"role": "system",
                  "content": build_system_prompt(self.base_system, chunks)}
        convo = clamp_dialog(history + [{"role": "user",
                                         "content": question}], max_messages=5)
        reply = self.llm.chat([system] + convo)

        # Prepare contexts as plain strings for logging
        contexts_for_log = [getattr(c, "content", str(c)) for c in chunks]

        # Log the interaction if a logger is configured
        if getattr(self, "interaction_logger", None):
            self.interaction_logger.log(
                question=question,
                answer=reply,
                contexts=contexts_for_log,
                ground_truth=""
            )

        # Build structured result for UI
        final_sources = []
        for content, src in zip(chunks, sources):
            final_sources.append(
                {
                    "title": src,
                    "snippet": content,
                }
            )

        return {
            "answer": reply,
            "sources": final_sources,
        }
