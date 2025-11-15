"""
Ragas Evaluater Module.
"""

from src.rag.application.ports.evaluater import Evaluater


class RagasEvaluater(Evaluater):
    """Ragas Evaluater implementation."""

    def __init__(self):
        """Initialize the Ragas Evaluater."""

    def evaluate(self, evaluation_data: dict) -> dict:
        """Evaluae the given data and return evaluation results."""
