"""
app interface module.
Contains the main applicaiton logic and integration of components.
"""

from infrastructure.config.config import AppConfig
from infrastructure.llm.mistral_client import MistralLLM
from infrastructure.vectorstores import FaissStore
from rag.application.use_cases.rag_chat import RAGChatService
from rag.application.use_cases.intent_classifier import IntentClassifier
from ui.chat_page import ChatPage

from src.rag.application.ports.embedders import MistralEmbedder


def main():
    cfg = AppConfig.load()

    llm = MistralLLM(chat_model=cfg.chat_model,
                     completion_args=cfg.completion_args)
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

    ChatPage(service=svc, title="Assistant Virtuel de la Mairie").render()


if __name__ == "__main__":
    main()
