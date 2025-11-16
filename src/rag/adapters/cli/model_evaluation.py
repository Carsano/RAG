"""
CLI for model evaluation
"""

from __future__ import annotations

from typing import Any, Dict, List
import json
import pandas as pd
import datasets

from src.rag.application.use_cases.rag_evaluation import RAGEvaluationUseCase

from src.rag.infrastructure.config.config import AppConfig
from src.rag.infrastructure.evaluation.ragas_evaluater import RagasEvaluater
from src.rag.infrastructure.llm.mistral_client import MistralLLM
from src.rag.infrastructure.embedders.mistral_embedder import MistralEmbedder
from src.rag.infrastructure.logging.logger import get_evaluation_logger

from src.rag.utils.utils import get_project_root


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
            items.append(json.loads(line))
            if limit is not None and len(items) >= limit:
                break
    return items


def _convert_items_to_ragas_format(items: List[Dict[str, Any]]
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


def _prepare_evaluation_data(path: str, limit: int | None = None
                             ) -> Dict[str, List[Any]]:
    """Prepare evaluation data from file path into RAGAS format.

    Args:
        path (str): Path to the evaluation data file.
        limit (int | None): Optional limit on number of items to load.

    Returns:
        dict: Evaluation data in RAGAS format.
    """
    root = get_project_root()
    full_path = root / path
    items = _load_items(str(full_path), limit=limit)
    return _convert_items_to_ragas_format(items)


def _convert_results_to_df(results: dict) -> pd.DataFrame:
    """Convert dict results into Pandas Dataframe

    Args:
        results (dict): Evaluation results to transform
    """
    results_df = datasets.Dataset.from_dict(results).to_pandas()
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 150)
    return results_df


def _print_evaluation_results(results: dict) -> None:
    """Print evaluation results in a readable format.

    Args:
        results (dict): Evaluation results to print.
    """
    results_df = _convert_results_to_df(results)
    print("\n--- Scores Moyens (sur tout le dataset) ---")
    average_scores = results_df.mean(numeric_only=True)
    print(average_scores)
    return average_scores


def main() -> None:
    """Entry point for model evaluation CLI."""
    evaluation_logger = get_evaluation_logger()
    evaluation_logger.info("CLI started by user")
    cfg = AppConfig.load()
    evaluation_logger.info("Config loaded")
    evaluater = RAGEvaluationUseCase(
        evaluater=RagasEvaluater(
            llm_model=MistralLLM(chat_model=cfg.chat_model,
                                 completion_args=cfg.completion_args),
            embedder=MistralEmbedder(model=cfg.embed_model, delay=0.0)
        )
    )
    evaluation_logger.info("Evaluater loaded")
    evaluation_data = _prepare_evaluation_data(
        path="logs/interactions/interactions.jsonl")
    evaluation_logger.info(f"evaluation_data prepared: {evaluation_data}")
    raw_results = evaluater.execute(evaluation_data=evaluation_data)
    evaluation_logger.info(f"raw results extracted: {raw_results}")
    cleanded_results = _print_evaluation_results(raw_results)
    return cleanded_results


if __name__ == "__main__":
    main()
