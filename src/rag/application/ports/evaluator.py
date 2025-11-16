"""
Defines the interface for an Evaluater in the RAG application.
"""
from typing import Protocol


class Evaluater(Protocol):
    """Interface for evaluating responses in the RAG application."""

    def evaluate(self, evaluation_data: dict) -> dict:
        """Evaluate the given data and return evaluation results.

        Args:
            evaluation_data (dict): The data to be evaluated.

        Returns:
            dict: The results of the evaluation.
        """
