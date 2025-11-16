"""RagasEvaluator: Evaluator implementation using RAGAS."""

from __future__ import annotations

from typing import Any, List

import pandas as pd

from ragas import evaluate as ragas_evaluate
# Metrics are passed from the caller; no direct import required here.


from src.rag.application.ports.evaluator import (
    Evaluator,
    EvaluatorOptions,
)


class RagasEvaluator(Evaluator):
    """Evaluator backed by RAGAS.

    This implementation delegates metric computation to `ragas.evaluate`
    and converts the result to a per-sample DataFrame. It does not perform
    retries or pass orchestration; keep that in the use case.
    """

    def evaluate(
        self,
        ds: Any,
        metrics: List[Any],
        llm: Any,
        embeddings: Any,
        options: EvaluatorOptions | None = None,
    ) -> pd.DataFrame:
        """Run RAGAS evaluation and return a per-sample DataFrame.

        Args:
            ds: Dataset to evaluate (e.g., HF Dataset).
            metrics: RAGAS metric objects.
            llm: LLM client compatible with RAGAS' LangChain bridge.
            embeddings: Embedding client compatible with RAGAS.
            options: Evaluator options (e.g., raise_exceptions).

        Returns:
            DataFrame with one row per sample and metric columns.

        Raises:
            Any exception from RAGAS if `options.raise_exceptions` is True.
        """
        opts = options or EvaluatorOptions()
        scores = ragas_evaluate(
            ds,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=opts.raise_exceptions,
        )
        return scores.to_pandas()
