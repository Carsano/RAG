"""
rag chat services interface.
Contains methods for managing chat interactions with RAG.
"""
from __future__ import annotations
from typing import List
from logging import Logger
from src.rag.infrastructure.logging.logger import get_usage_logger
from src.rag.application.ports.llm import LLM
from src.rag.application.ports.retriever import Retriever
from src.rag.application.use_cases.prompting import build_system_prompt
from src.rag.application.use_cases.prompting import clamp_dialog


class RAGChatService:
    def __init__(self, llm: LLM, retriever: Retriever,
                 base_system_prompt: str, intent_classifier,
                 usage_logger: Logger | None = None):
        self.llm = llm
        self.retriever = retriever
        self.base_system = base_system_prompt
        self.intent_classifier = intent_classifier
        self.log = usage_logger or get_usage_logger()
        self.log.info("RAGChatService constructed")

    def _retrieve(self, question: str, k: int = 10) -> list[str]:
        chunks = self.retriever.search(question, k=k)
        self.log.debug(f"retrieve: k={k}, got={len(chunks)} chunks")
        return chunks

    def answer(self, history: List[dict], question: str) -> str:
        self.log.debug("answer: start")
        intent = self.intent_classifier.classify(question)
        self.log.debug(f"answer: intent={intent}")
        chunks = self._retrieve(question) if intent == "rag" else []
        self.log.debug(f"answer: chunks_used={len(chunks)}")
        system = {"role": "system",
                  "content": build_system_prompt(self.base_system, chunks)}
        convo = clamp_dialog(history + [{"role": "user",
                                         "content": question}], max_messages=5)
        response = self.llm.chat([system] + convo)
        self.log.debug("answer: llm_response_ready")
        return response
