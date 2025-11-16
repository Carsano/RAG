"""
Use case for RAG evaluation.
"""

from __future__ import annotations

from src.rag.application.ports.evaluater import Evaluater


class RAGEvaluationUseCase:
    """Use case for RAG evaluation."""

    def __init__(self, evaluater: Evaluater):
        """Initialize the RAG Evaluation Use Case."""
        self.evaluater = evaluater

    def execute(self, evaluation_data: dict) -> dict:
        """Execute the evaluation and return results.

        Args:
            evaluation_data (dict): The data to be evaluated."""

        return self.evaluater.evaluate(evaluation_data)
