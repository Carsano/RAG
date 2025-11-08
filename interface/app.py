"""
app interface module.
Contains the main applicaiton logic and integration of components.
"""

from core.config import AppConfig
from adapters.mistral_client import MistralLLM
from adapters.faiss_store import FaissStore
from services.rag_chat import RAGChatService
from ui.chat_page import ChatPage


def main():
    cfg = AppConfig.load()
    llm = MistralLLM(chat_model=cfg.chat_model, embed_model=cfg.embed_model,
                     completion_args=cfg.completion_args)
    store = FaissStore(index_path=cfg.faiss_index_path,
                       chunks_pickle_path=cfg.chunks_path)
    svc = RAGChatService(llm=llm, store=store,
                         base_system_prompt=cfg.system_prompt)
    ChatPage(service=svc, title="Assistant Virtuel de la Mairie").render()


if __name__ == "__main__":
    main()
