"""
app interface module.
Contains the main applicaiton logic and integration of components.
"""

from src.rag.infrastructure.config.config import AppConfig
from src.rag.infrastructure.llm.mistral_client import MistralLLM
from src.rag.infrastructure.vectorstores.faiss_store_manager import FaissStore
from src.rag.application.use_cases.rag_chat import RAGChatService
from src.rag.application.use_cases.intent_classifier import IntentClassifier
from src.rag.application.ports.retriever import Retriever
from src.rag.adapters.ui.chat_page import ChatPage

from src.rag.infrastructure.embedders.mistral_embedder import MistralEmbedder
from src.rag.infrastructure.logging.logger import get_usage_logger


def main():
    usage_logger = get_usage_logger()
    usage_logger.info("CLI started by user")

    cfg = AppConfig.load()

    llm = MistralLLM(chat_model=cfg.chat_model,
                     completion_args=cfg.completion_args)
    embedder = MistralEmbedder(model=cfg.embed_model, delay=0.0)
    store = FaissStore(index_path=cfg.faiss_index_path,
                       chunks_pickle_path=cfg.chunks_path)
    classifier = IntentClassifier(llm=llm)
    retriever = Retriever(embedder=embedder, store=store)

    svc = RAGChatService(
        llm=llm,
        retriever=retriever,
        base_system_prompt=cfg.system_prompt,
        intent_classifier=classifier,
    )

    usage_logger.info("RAGChatService initialized")

    ChatPage(service=svc, title="Assistant Virtuel de la Mairie").render()


if __name__ == "__main__":
    main()
