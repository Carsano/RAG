"""
Ragas Evaluater Module.
"""
import ragas
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from src.rag.application.ports.evaluater import Evaluater
from src.rag.application.ports.llm import LLM
from src.rag.application.ports.embedders import Embedder


class RagasEvaluater(Evaluater):
    """Ragas Evaluater implementation."""

    def __init__(self, llm_model: LLM, embedder: Embedder):
        """Initialize the Ragas Evaluater."""
        self.llm_model = llm_model
        self.embedder = embedder

    def evaluate(self, evaluation_data: dict) -> dict:
        """Evaluae the given data and return evaluation results."""
