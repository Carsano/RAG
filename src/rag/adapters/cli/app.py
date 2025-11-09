"""
app interface module.
Contains the main applicaiton logic and integration of components.
"""

from src.rag.infrastructure.config.config import AppConfig
from src.rag.infrastructure.llm.mistral_client import MistralLLM
from src.rag.infrastructure.vectorstores.faiss_store import FaissStore
from src.rag.application.use_cases.rag_chat import RAGChatService
from src.rag.application.use_cases.intent_classifier import IntentClassifier
from src.rag.adapters.ui.chat_page import ChatPage

from src.rag.application.ports.embedders import MistralEmbedder
from src.rag.infrastructure.logging.logger import get_usage_logger


def main():
    usage_logger = get_usage_logger()
    usage_logger.info("CLI started by user")

    cfg = AppConfig.load()

    llm = MistralLLM(model_name=cfg.chat_model,
                     temperature=cfg.completion_args.get("temperature", 0.2),
                     max_tokens=cfg.completion_args.get("max_tokens", 300),
                     top_p=cfg.completion_args.get("top_p", 0.22))
    embedder = MistralEmbedder(model=cfg.embed_model, delay=0.0)
    store = FaissStore(index_path=cfg.faiss_index_path,
                       chunks_pickle_path=cfg.chunks_path)
    classifier = IntentClassifier(llm=llm)

    svc = RAGChatService(
        llm=llm,
        embedder=embedder,
        store=store,
        base_system_prompt=cfg.system_prompt,
        intent_classifier=classifier,
    )

    usage_logger.info("RAGChatService initialized")

    ChatPage(service=svc, title="Assistant Virtuel de la Mairie").render()


if __name__ == "__main__":
    main()
