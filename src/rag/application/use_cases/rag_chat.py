"""
RAG chat service.

Provides utilities to manage chat interactions that use
Retrieval-Augmented Generation (RAG).
"""
from __future__ import annotations
from typing import List

from src.rag.application.ports.llm import LLM
from src.rag.application.ports.retriever import Retriever
from src.rag.application.use_cases.prompting import build_system_prompt
from src.rag.application.use_cases.prompting import clamp_dialog
from src.rag.application.use_cases.intent_classifier import IntentClassifier


class RAGChatService:
    """
    Manage chat with Retrieval-Augmented Generation.

    Embeds questions, retrieves relevant chunks, classifies intent, and
    produces answers with a language model.
    """

    def __init__(self, llm: LLM, retriever: Retriever,
                 base_system_prompt: str, intent_classifier: IntentClassifier):
        """Initialize the service.

        Args:
          llm (LLM): Interface to the language model used to generate replies.
          retriever (Retriever): Component that retrieves relevant chunks.
          base_system_prompt (str): Base system prompt for the model.
          intent_classifier: Component that classifies user intent.
        """
        self.llm = llm
        self.retriever = retriever
        self.base_system = base_system_prompt
        self.intent_classifier = intent_classifier

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
        chunks = self.retriever.retrieve(question) if intent == "rag" else []
        system = {"role": "system",
                  "content": build_system_prompt(self.base_system, chunks)}
        convo = clamp_dialog(history + [{"role": "user",
                                         "content": question}], max_messages=5)
        return self.llm.chat([system] + convo)
