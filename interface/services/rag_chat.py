"""
rag chat services interface.
Contains methods for managing chat interactions with RAG.
"""
from typing import List
from ports.llm import LLM
from ports.vector_store import VectorStore
from services.prompting import build_system_prompt, clamp_dialog


class RAGChatService:
    def __init__(self, llm: LLM, store: VectorStore, base_system_prompt: str):
        self.llm = llm
        self.store = store
        self.base_system = base_system_prompt

    def _retrieve(self, question: str, k: int = 10) -> list[str]:
        emb = self.llm.embed(question)
        ids, _ = self.store.search(emb, k=k)
        return self.store.get_chunks(ids)

    def answer(self, history: List[dict], question: str) -> str:
        intent = self.llm.classify_intent(question)
        chunks = self._retrieve(question) if intent == "rag" else []
        system = {"role": "system",
                  "content": build_system_prompt(self.base_system, chunks)}
        convo = clamp_dialog(history + [{"role": "user",
                                         "content": question}], max_messages=5)
        return self.llm.chat([system] + convo)
