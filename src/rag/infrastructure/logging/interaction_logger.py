"""Module for logging interactions including questions, answers, and contexts.

This module provides the InteractionLogger class to log interactions
to a JSONL file for later analysis or debugging.
"""
# src/rag/infrastructure/logging/interaction_logger.py
import os
import json


class InteractionLogger:
    """Logger for recording interactions to a JSONL file.

    Attributes:
        filepath (str): Path to the log file where interactions are saved.
    """

    def __init__(self, filepath="logs/interactions/interactions.jsonl"):
        """Initializes the InteractionLogger.

        Creates the directory for the log file if it does not exist.

        Args:
            filepath (str): Path to the JSONL file to log interactions.
                Defaults to 'logs/interactions.jsonl'.
        """
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def log(self, question, answer, contexts, ground_truth=""):
        """Logs an interaction entry to the JSONL file.

        Each entry contains the question, answer, related contexts, and
        an optional ground truth string.

        Args:
            question (str): The question asked.
            answer (str): The answer provided.
            contexts (list): List of context strings related
            to the interaction.
            ground_truth (str, optional): The ground truth answer if available.
                Defaults to an empty string.

        Writes a JSON object per line to the log file.
        """
        entry = {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
        }
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
