"""
RAG chat service.

Provides utilities to manage chat interactions that use
Retrieval-Augmented Generation (RAG).
"""
from __future__ import annotations
import json
import os
import time
from typing import List, Optional, Dict, Any

from src.rag import PIPELINE_VERSION
from src.rag.application.ports.llm import LLM
from src.rag.application.ports.retriever import Retriever
from src.rag.application.ports.reranker import Reranker
from src.rag.application.use_cases.prompting import build_system_prompt
from src.rag.application.use_cases.prompting import clamp_dialog
from src.rag.application.use_cases.intent_classifier import IntentClassifier
from src.rag.infrastructure.logging.interaction_logger import InteractionLogger
from src.rag.infrastructure.logging.logger import get_usage_logger
from src.rag.infrastructure.logging.rag_audit_logger import RAGAuditLogger
from src.rag.utils.logging_builder import build_rag_log


usage_logger = get_usage_logger()


class RAGChatService:
    """
    Manage chat with Retrieval-Augmented Generation.

    Embeds questions, retrieves relevant chunks, classifies intent, and
    produces answers with a language model.
    """

    def __init__(
        self,
        llm: LLM,
        retriever: Retriever,
        reranker: Reranker,
        base_system_prompt: str,
        intent_classifier: IntentClassifier,
        interaction_logger: Optional[InteractionLogger] = None,
        audit_logger: Optional[RAGAuditLogger] = None,
        pipeline_version: str = PIPELINE_VERSION,
        component_versions: Optional[Dict[str, Any]] = None,
    ):
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
        self.audit_logger = audit_logger
        self.pipeline_version = pipeline_version
        self.component_versions = component_versions or {}

    def answer(
        self,
        history: List[dict],
        question: str,
        max_sources: int = 5,
        top_k: int = 5,
        user_id: Optional[str] = None,
        input_parameters: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """Generate an answer using dialog history and retrieval.

        Args:
            history: Prior conversation messages.
            question: Current user question.
            max_sources: Maximum number of sources exposed to the UI.
            top_k: Number of candidates to request from the retriever.
            user_id: Optional identifier of the active user.
            input_parameters: User-requested parameters such as temperature.

        Returns:
            dict: Final answer, sources, and optional audit payload.
        """
        user_identifier = user_id or "anonymous"
        requested_params = input_parameters or {}
        latencies: Dict[str, float | None] = {
            "retrieval_ms": None,
            "rerank_ms": None,
            "generation_ms": None,
            "total_ms": None,
        }

        total_start = time.perf_counter()
        intent = self.intent_classifier.classify(question)

        if intent == "rag":
            retrieval_start = time.perf_counter()
            retrievings = self.retriever.retrieve(question, k=top_k)
            latencies["retrieval_ms"] = (
                time.perf_counter() - retrieval_start
            ) * 1000.0
        else:
            retrievings = []

        if retrievings:
            rerank_start = time.perf_counter()
            ranked = self.reranker.rank(question, retrievings)
            latencies["rerank_ms"] = (
                time.perf_counter() - rerank_start
            ) * 1000.0
            selected_ranked = ranked[:max_sources]
            chunks = [rc.chunk["content"] for rc in selected_ranked]
            sources = [rc.chunk["source"] for rc in selected_ranked]
        else:
            ranked = []
            selected_ranked = []
            chunks = []
            sources = []

        system = {"role": "system",
                  "content": build_system_prompt(self.base_system, chunks)}
        convo = clamp_dialog(history + [{"role": "user",
                                         "content": question}], max_messages=5)
        prompt_messages = [system] + convo
        gen_start = time.perf_counter()
        reply = self.llm.chat(prompt_messages)
        latencies["generation_ms"] = (
            time.perf_counter() - gen_start
        ) * 1000.0

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

        latencies["total_ms"] = (time.perf_counter() - total_start) * 1000.0
        audit_payload = self._build_audit_payload(
            user_identifier=user_identifier,
            question=question,
            reply=reply,
            intent=intent,
            retrievings=retrievings,
            ranked=ranked,
            selected_ranked=selected_ranked,
            prompt_messages=prompt_messages,
            requested_params=requested_params,
            latencies=latencies,
        )
        if self.audit_logger and audit_payload:
            self.audit_logger.persist(audit_payload)

        return {
            "answer": reply,
            "sources": final_sources,
            "audit": audit_payload,
        }

    def _build_audit_payload(
        self,
        *,
        user_identifier: str,
        question: str,
        reply: str,
        retrievings: List[dict],
        intent: str,
        ranked: List[Any],
        selected_ranked: List[Any],
        prompt_messages: List[dict],
        requested_params: Dict[str, Any],
        latencies: Dict[str, float | None],
    ) -> Optional[dict]:
        """Assemble the JSON payload expected by the audit storage.

        Args:
            user_identifier: Identifier of the user for this interaction.
            question: Raw question asked by the user.
            reply: Final answer returned by the LLM.
            retrievings: Retriever outputs with metadata.
            intent: Intent label returned by the classifier.
            ranked: Ranked chunks emitted by the reranker.
            selected_ranked: Subset of ranked chunks shown to the user.
            prompt_messages: Prompt history sent to the LLM.
            requested_params: User-requested configuration values.
            latencies: Step timings in milliseconds.

        Returns:
            dict | None: Structured audit payload or None if logging disabled.
        """
        if not getattr(self, "audit_logger", None):
            return None

        effective_parameters = {
            "top_k_used": len(retrievings),
            "max_sources_used": len(selected_ranked),
            "model": getattr(self.llm, "model", self.llm.__class__.__name__),
            "temperature": getattr(self.llm, "args", {}).get("temperature"),
            "max_tokens": getattr(self.llm, "args", {}).get("max_tokens"),
            "top_p": getattr(self.llm, "args", {}).get("top_p"),
            "reranker": self.reranker.__class__.__name__,
            "intent": intent,
        }

        retrieval_chunks = [
            {
                "chunk_id": chunk.get("chunk_id"),
                "score_retrieval": chunk.get("score_retriever"),
                "distance": chunk.get("distance"),
                "content_snapshot": chunk.get("content"),
                "source_path": chunk.get("source"),
            }
            for chunk in retrievings
        ]

        selected_ids = {
            rc.chunk.get("chunk_id")
            for rc in selected_ranked
            if isinstance(getattr(rc, "chunk", None), dict)
            and rc.chunk.get("chunk_id") is not None
        }
        selected_contents = {
            rc.chunk.get("content")
            for rc in selected_ranked
            if isinstance(getattr(rc, "chunk", None), dict)
        }
        rerank_chunks = []
        for rc in ranked:
            chunk = getattr(rc, "chunk", None)
            if not isinstance(chunk, dict):
                continue
            content = chunk.get("content")
            chunk_id = chunk.get("chunk_id")
            was_selected = (
                chunk_id in selected_ids
                if chunk_id is not None
                else content in selected_contents
            )
            rerank_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "score_retrieval": chunk.get("score_retriever"),
                    "score_reranker": getattr(rc, "score", None),
                    "was_selected": was_selected,
                    "content_snapshot": content,
                    "source_path": chunk.get("source"),
                }
            )

        llm_metadata = getattr(self.llm, "last_call_metadata", {}) or {}
        if latencies.get("generation_ms") is not None:
            llm_metadata.setdefault("latency_ms", latencies["generation_ms"])

        metrics = {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": None,
        }

        prompt_final = json.dumps(prompt_messages, ensure_ascii=False)
        audit_payload = build_rag_log(
            user_id=user_identifier,
            raw_question=question,
            input_parameters=requested_params,
            effective_parameters=effective_parameters,
            retrieval_chunks=retrieval_chunks,
            rerank_chunks=rerank_chunks,
            prompt_final=prompt_final,
            llm_metadata=llm_metadata,
            raw_answer=reply,
            clean_answer=reply.strip(),
            metrics=metrics,
            component_versions=self.component_versions,
            pipeline_version=self.pipeline_version,
            latencies=latencies,
        )
        return audit_payload
