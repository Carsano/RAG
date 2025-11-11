"""
RAG chat service.

Provides utilities to manage chat interactions that use
Retrieval-Augmented Generation (RAG).
"""
from __future__ import annotations
from typing import List
from src.rag.application.ports.llm import LLM
from src.rag.application.ports.vector_store_manager import VectorStoreManager
from src.rag.application.ports.embedders import Embedder
from src.rag.application.use_cases.prompting import build_system_prompt
from src.rag.application.use_cases.prompting import clamp_dialog


class RAGChatService:
    """
    Manage chat with Retrieval-Augmented Generation.

    Embeds questions, retrieves relevant chunks, classifies intent, and
    produces answers with a language model.
    """

    def __init__(self, llm: LLM, embedder: Embedder, store: VectorStoreManager,
                 base_system_prompt: str, intent_classifier):
        """Initialize the service.

        Args:
          llm (LLM): Interface to the language model used to generate replies.
          embedder (Embedder): Component that turns text into embeddings.
          store (VectorStoreManager): Vector store used to search chunks.
          base_system_prompt (str): Base system prompt for the model.
          intent_classifier: Component that classifies user intent.
        """
        self.llm = llm
        self.embedder = embedder
        self.store = store
        self.base_system = base_system_prompt
        self.intent_classifier = intent_classifier

    def _retrieve(self, question: str, k: int = 10) -> list[str]:
        """Retrieve relevant chunks for a question.

        The question is embedded then searched in the vector store.

        Args:
          question (str): User question to embed and search for.
          k (int): Number of top chunks to retrieve.

        Returns:
          list[str]: Retrieved document chunks.

        Raises:
          RuntimeError: If embedding the question fails.
        """
        emb = self.embedder.embed(question)
        if emb is None:
            raise RuntimeError("Embedding a échoué")
        ids, _ = self.store.search(emb, k=k)
        return self.store.get_chunks(ids)

    def answer(self, history: List[dict], question: str) -> str:
        """Generate an answer using history and optional retrieval.

        The method classifies intent. If the intent is "rag", it retrieves
        chunks. It then builds the system prompt, clamps dialog history, and
        asks the LLM for a reply.

        Args:
          history (List[dict]): Prior messages in the conversation.
          question (str): Current user question.

        Returns:
          str: The model's answer.
        """
        intent = self.intent_classifier.classify(question)
        chunks = self._retrieve(question) if intent == "rag" else []
        system = {"role": "system",
                  "content": build_system_prompt(self.base_system, chunks)}
        convo = clamp_dialog(history + [{"role": "user",
                                         "content": question}], max_messages=5)
        return self.llm.chat([system] + convo)
