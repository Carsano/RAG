"""EvaluatorPort: abstraction for evaluation backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Protocol

import pandas as pd


@dataclass(frozen=True)
class EvaluatorOptions:
    """Options controlling evaluator behavior.

    Attributes:
        raise_exceptions: If True, evaluator bubbles exceptions. If False,
            evaluator should trap transient errors and return best-effort
            results.
    """
    raise_exceptions: bool = False


class Evaluator(Protocol):
    """Contract for a RAG evaluator implementation.

    The evaluator computes per-sample metrics for a dataset, given a set
    of metric objects and model clients. It must return a pandas DataFrame
    where each row corresponds to one input sample.
    """

    def evaluate(
        self,
        ds: Any,
        metrics: List[Any],
        llm: Any,
        embeddings: Any,
        options: EvaluatorOptions | None = None,
    ) -> pd.DataFrame:
        """Run evaluation and return a per-sample DataFrame.

        Args:
            ds: Dataset to evaluate (e.g., a Hugging Face Dataset).
            metrics: Metric objects understood by the implementation.
            llm: Chat/completions client understood by the implementation.
            embeddings: Embedding client understood by the implementation.
            options: Optional evaluator options.

        Returns:
            A DataFrame with one row per sample and one column per metric.
        """
        ...
