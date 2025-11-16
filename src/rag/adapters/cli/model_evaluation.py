"""
CLI for model evaluation
"""

from __future__ import annotations

from typing import Any, Dict, List
import json

from src.rag.application.use_cases.rag_evaluation import RAGEvaluationUseCase

from src.rag.infrastructure.evaluation.ragas_evaluater import RagasEvaluater
from src.rag.infrastructure.llm.mistral_client import MistralLLM
from src.rag.infrastructure.embedders.mistral_embedder import MistralEmbedder

def _load_items(path: str, limit: int | None = None) -> List[Dict[str, Any]]:
    """Load evaluation items from a given path.
    
    Args:
        path (str): Path to the evaluation data.
        limit (int | None): Optional limit on number of items to load.
        
    Returns:
        dict: Loaded evaluation data."""
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            items.appends(json.loads(line))
            if limit is not None and len(items) >= limit:
                break

    return items
def main() -> None:
    """Entry point for model evaluation CLI."""
    evaluater = RAGEvaluationUseCase(
        evaluater=RagasEvaluater(
            llm_model=MistralLLM(),
            embedder=MistralEmbedder()
        )
    )
    evaluater.execute(evaluation_data=)