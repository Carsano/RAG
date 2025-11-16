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

def __convert_items_to_ragas_format(items: List[Dict[str, Any]]
                                    ) -> Dict[str, List[Any]]:
    """Convert list of dicts items into RAGAS expected dict of lists structure.
    
    Output schema:
    {
        "question": List[str],
        "answer": List[str],
        "context": List[List[str]],
        "ground_truth": [List[str]]
    """
    questions: List[str] = [],
    answers: List[str] = [],
    contexts: List[List[str]] = [],
    ground_truths: List[str] = []

    for it in items:
        questions.append(it["question"].strip())
        answers.append(it["answer"].strip())
        ctx = it.get("contexts", [])
        if not isinstance(ctx, list):
            ctx = [str(ctx)] if ctx else []
        contexts.append([str(c) for c in ctx])
        gt = it.get("ground_truth", "")
        ground_truths.append(str(gt) if gt is not None else "")

    return {
        "question": questions,
        "answer": answers,
        "context": contexts,
        "ground_truth": ground_truths
    }

def main() -> None:
    """Entry point for model evaluation CLI."""
    evaluater = RAGEvaluationUseCase(
        evaluater=RagasEvaluater(
            llm_model=MistralLLM(),
            embedder=MistralEmbedder()
        )
    )
    evaluater.execute(evaluation_data=)