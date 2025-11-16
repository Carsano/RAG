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
                                   ) -> datasets.Dataset:
    """Convert list of dicts items into a RAGAS-compatible
    Hugging Face Dataset.

    Output schema columns:
    - "question": List[str]
    - "answer": List[str]
    - "contexts": List[List[str]]
    - "ground_truth": List[str]

    Returns:
        datasets.Dataset: Dataset with required columns for ragas.evaluate.
    """
    questions: List[str] = []
    answers: List[str] = []
    contexts_list: List[List[str]] = []
    ground_truths: List[str] = []

    for it in items:
        questions.append(it["question"].strip())
        answers.append(it["answer"].strip())
        ctx = it.get("contexts", [])
        if not isinstance(ctx, list):
            ctx = [str(ctx)] if ctx else []
        contexts_list.append([str(c) for c in ctx])
        gt = it.get("ground_truth", "")
        ground_truths.append(str(gt) if gt is not None else "")

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    }
    return datasets.Dataset.from_dict(data)


def _prepare_evaluation_data(path: str, limit: int | None = None
                             ) -> datasets.Dataset:
    """Prepare evaluation data from file path into RAGAS format.

    Args:
        path (str): Path to the evaluation data file.
        limit (int | None): Optional limit on number of items to load.

    Returns:
        datasets.Dataset: Evaluation data in RAGAS format.
    """
    root = get_project_root()
    full_path = root / path
    items = _load_items(str(full_path), limit=limit)
    return _convert_items_to_ragas_format(items)


def _convert_results_to_df(results: Any) -> pd.DataFrame:
    """Convert evaluation results into a Pandas DataFrame.

    Supports ragas `EvaluationResult` via `.to_pandas()` and falls back
    to a plain dict of column->values.
    """
    if hasattr(results, "to_pandas") and callable(getattr(results,
                                                          "to_pandas")):
        return results.to_pandas()
    if isinstance(results, dict):
        return datasets.Dataset.from_dict(results).to_pandas()
    raise TypeError(
        f"Unsupported results type: {type(results)}. "
        f"Expected EvaluationResult or dict."
    )


def _print_evaluation_results(results: Any) -> pd.Series:
    """Print evaluation results in a readable format and return averages."""
    results_df = _convert_results_to_df(results)
    print("\n--- Mean scores (dataset-wide) ---")
    avg = results_df.mean(numeric_only=True)
    print(avg)
    return avg


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
