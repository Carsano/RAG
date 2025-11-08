"""
rag chat services interface.
Contains methods for managing chat interactions with RAG.
"""
from __future__ import annotations
from typing import List, Any
from src.rag.application.ports.llm import LLM
from src.rag.application.ports.vector_store import VectorStore
from src.rag.application.use_cases.prompting import build_system_prompt
from src.rag.application.use_cases.prompting import clamp_dialog


class RAGChatService:
    def __init__(self, llm: LLM, embedder: Any, store: VectorStore,
                 base_system_prompt: str, intent_classifier):
        self.llm = llm
        self.embedder = embedder
        self.store = store
        self.base_system = base_system_prompt
        self.intent_classifier = intent_classifier

    def _retrieve(self, question: str, k: int = 10) -> list[str]:
        emb = self.embedder.embed(question)
        if emb is None:
            raise RuntimeError("Embedding a échoué")
        ids, _ = self.store.search(emb, k=k)
        return self.store.get_chunks(ids)

    def answer(self, history: List[dict], question: str) -> str:
        intent = self.intent_classifier.classify(question)
        chunks = self._retrieve(question) if intent == "rag" else []
        system = {"role": "system",
                  "content": build_system_prompt(self.base_system, chunks)}
        convo = clamp_dialog(history + [{"role": "user",
                                         "content": question}], max_messages=5)
        return self.llm.chat([system] + convo)
