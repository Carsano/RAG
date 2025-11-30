"""RAG chat service wiring for UI.

This module builds and exposes a singleton instance of RAGChatService
for the Streamlit UI, reusing the same dependencies and configuration
as the CLI entry point.
"""

from typing import Optional

from src.rag.application.use_cases.rag_chat import RAGChatService
from src.rag.application.use_cases.intent_classifier import IntentClassifier

from src.rag.infrastructure.config.config import AppConfig
from src.rag.infrastructure.llm.mistral_client import MistralLLM
from src.rag.infrastructure.vectorstores.faiss_store_manager import FaissStore
from src.rag.infrastructure.vectorstores.faiss_store_retriever import (
    FaissRetriever,
)
from src.rag.infrastructure.rerankers.cross_encoder_reranker import (
    CrossEncoderReranker,
)
from src.rag.infrastructure.rerankers.llm_reranker import (
    LLMReranker
)
from src.rag.infrastructure.rerankers.keywords_overlap_scorer import (
    KeywordsOverlapScorer
)
from src.rag.infrastructure.embedders.mistral_embedder import MistralEmbedder
from src.rag.infrastructure.logging.interaction_logger import (
    InteractionLogger,
)
from src.rag.infrastructure.logging.rag_audit_logger import RAGAuditLogger


_service: Optional[RAGChatService] = None
_rerankers = {}


def _build_chat_service() -> RAGChatService:
    """Build and return a configured RAGChatService instance.

    This mirrors the wiring used in the CLI entry point:
    - load AppConfig
    - create LLM, embedder, FAISS store, retriever
    - create intent classifier and interaction logger
    - assemble RAGChatService
    """
    cfg = AppConfig.load()

    llm = MistralLLM(
        chat_model=cfg.chat_model,
        completion_args=cfg.completion_args,
    )

    embedder = MistralEmbedder(
        model=cfg.embed_model,
        delay=0.0,
    )

    store = FaissStore(
        index_path=cfg.faiss_index_path,
        chunks_pickle_path=cfg.chunks_path,
        sources_path=cfg.sources_path,
    )

    retriever = FaissRetriever(
        embedder=embedder,
        store=store,
    )

    reranker = LLMReranker(llm=llm)

    classifier = IntentClassifier(llm=llm)
    interaction_logger = InteractionLogger()
    audit_logger = RAGAuditLogger()

    service = RAGChatService(
        llm=llm,
        retriever=retriever,
        reranker=reranker,
        base_system_prompt=cfg.system_prompt,
        intent_classifier=classifier,
        interaction_logger=interaction_logger,
        audit_logger=audit_logger,
        component_versions={
            "llm_version": cfg.chat_model,
            "embedder_version": cfg.embed_model,
            "retriever_version": retriever.__class__.__name__,
            "reranker_version": reranker.__class__.__name__,
            "chunker_version": None,
        },
    )

    return service


def _get_or_create_reranker(name: str):
    if name in _rerankers:
        return _rerankers[name]

    if name == "keyword_overlap_scorer":
        _rerankers[name] = KeywordsOverlapScorer()
    elif name == "cross_encoder":
        _rerankers[name] = CrossEncoderReranker()
    elif name == "llm_reranker":
        service = get_chat_service()
        llm = service.llm
        _rerankers[name] = LLMReranker(llm=llm)
    else:
        raise ValueError(f"Unknown reranker name: {name}")

    return _rerankers[name]


def configure_reranker(name: str) -> None:
    """Configure the reranker used by the shared chat service."""
    service = get_chat_service()
    service.reranker = _get_or_create_reranker(name)


def get_chat_service() -> RAGChatService:
    """Return a singleton RAGChatService instance for the UI.

    The service is built once and reused across Streamlit reruns.
    """
    global _service
    if _service is None:
        _service = _build_chat_service()
    return _service
