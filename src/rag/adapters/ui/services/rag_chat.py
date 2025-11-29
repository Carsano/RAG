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
from src.rag.infrastructure.rerankers.llm_reranker import (
    LLMReranker
)
from src.rag.infrastructure.embedders.mistral_embedder import MistralEmbedder
from src.rag.infrastructure.logging.interaction_logger import (
    InteractionLogger,
)


_service: Optional[RAGChatService] = None


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

    service = RAGChatService(
        llm=llm,
        retriever=retriever,
        reranker=reranker,
        base_system_prompt=cfg.system_prompt,
        intent_classifier=classifier,
        interaction_logger=interaction_logger,
    )

    return service


def get_chat_service() -> RAGChatService:
    """Return a singleton RAGChatService instance for the UI.

    The service is built once and reused across Streamlit reruns.
    """
    global _service
    if _service is None:
        _service = _build_chat_service()
    return _service
