"""
RAG chat service.

Provides utilities to manage chat interactions that use
Retrieval-Augmented Generation (RAG).
"""
from __future__ import annotations
import os
from typing import List, Optional

from src.rag.application.ports.llm import LLM
from src.rag.application.ports.retriever import Retriever
from src.rag.application.ports.reranker import Reranker
from src.rag.application.use_cases.prompting import build_system_prompt
from src.rag.application.use_cases.prompting import clamp_dialog
from src.rag.application.use_cases.intent_classifier import IntentClassifier
from src.rag.infrastructure.logging.interaction_logger import InteractionLogger
from src.rag.infrastructure.logging.logger import get_usage_logger


usage_logger = get_usage_logger()


class RAGChatService:
    """
    Manage chat with Retrieval-Augmented Generation.

    Embeds questions, retrieves relevant chunks, classifies intent, and
    produces answers with a language model.
    """

    def __init__(self, llm: LLM, retriever: Retriever, reranker: Reranker,
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
        self.reranker = reranker
        self.base_system = base_system_prompt
        self.intent_classifier = intent_classifier
        self.interaction_logger = interaction_logger

    def answer(self, history: List[dict], question: str,
               max_sources: int = 5) -> dict:
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
        retrievings = (
            self.retriever.retrieve(question, k=max_sources)
            if intent == "rag"
            else []
        )

        if retrievings:
            ranked = self.reranker.rank(question, retrievings)
            top_ranked = ranked[:max_sources]
            chunks = [rc.chunk["content"] for rc in top_ranked]
            sources = [rc.chunk["source"] for rc in top_ranked]
        else:
            chunks = []
            sources = []

        system = {"role": "system",
                  "content": build_system_prompt(self.base_system, chunks)}
        convo = clamp_dialog(history + [{"role": "user",
                                         "content": question}], max_messages=5)
        reply = self.llm.chat([system] + convo)

        # Log high-level usage info (intent, path, previews)
        try:
            answer_preview = (
                reply if len(reply) <= 400 else reply[:400] + "…"
            )
            question_preview = (
                question if len(question) <= 200 else question[:200] + "…"
            )
            used_retrieval = bool(retrievings)
            nb_chunks = len(chunks)
            usage_logger.info(
                "Chat interaction | intent=%s | used_retrieval=%s | "
                "nb_chunks=%d | question=\"%s\" | answer=\"%s\""
                % (
                    intent,
                    used_retrieval,
                    nb_chunks,
                    question_preview,
                    answer_preview,
                )
            )
        except Exception:
            # Never break the chat flow because of logging
            pass

        # Prepare contexts as plain strings for logging
        contexts_for_log = chunks

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
                    "title": os.path.basename(src),
                    "snippet": content,
                }
            )

        return {
            "answer": reply,
            "sources": final_sources,
        }
