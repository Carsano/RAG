"""
Ragas-based EvaluationEngine implementation.
"""
from __future__ import annotations
from typing import Any, Dict, List, Mapping, Sequence, Tuple
from rag.application.ports.evaluation import EvaluationEngine
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision
from ragas.metrics import context_relevancy, faithfulness


class RagasEvaluationEngine(EvaluationEngine):
    """Compute RAG metrics with Ragas.

    Args:
        judge_llm (Any): LLM compatible with the installed Ragas version.
        embeddings (Any | None): Embedding model for metrics that need it.
        metrics (Sequence[str]): Metric names: "faithfulness",
            "answer_relevancy", "context_relevancy", "context_precision".
    """

    SUPPORTED_METRICS = (
        "faithfulness",
        "answer_relevancy",
        "context_relevancy",
        "context_precision",
    )

    def __init__(
        self,
        judge_llm: Any,
        embeddings: Any | None = None,
        metrics: Sequence[str] = (
            "faithfulness",
            "answer_relevancy",
            "context_relevancy",
            "context_precision",
        ),
    ) -> None:
        """Initialize RagasEvaluationEngine.

        Args:
            judge_llm (Any): LLM compatible with the installed Ragas version.
            embeddings (Any | None): Embedding model for metrics that need it.
            metrics (Sequence[str]): Metric names to compute.
        """
        self.judge_llm = judge_llm
        self.embeddings = embeddings
        self.metrics = list(metrics)

    def _load_ragas(self):
        """Load Ragas and related dependencies.

        Returns:
            Tuple containing Dataset class, evaluate function, and metric map.

        Raises:
            RuntimeError: If Ragas or datasets are not installed.
        """
        metric_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_relevancy": context_relevancy,
            "context_precision": context_precision,
        }
        return Dataset, evaluate, metric_map

    def _validate_metrics(self, metric_map: Dict[str, Any]) -> List[Any]:
        """Validate selected metrics against supported metrics.

        Args:
            metric_map (Dict[str, Any]): Mapping of metric names.

        Returns:
            List of selected metric functions.

        Raises:
            ValueError: If any selected metric is unsupported.
        """
        unknown = [m for m in self.metrics if m not in metric_map]
        if unknown:
            supported = ", ".join(sorted(metric_map))
            raise ValueError(
                f"Unsupported metrics: {unknown}. Supported: {supported}"
            )
        return [metric_map[m] for m in self.metrics]

    def _build_columns(
        self, samples: Sequence[Mapping]
    ) -> Tuple[List[str], Dict[str, List[Any]]]:
        """Build columns from samples for Ragas evaluation.

        Args:
            samples (Sequence[Mapping]): List of sample mappings.

        Returns:
            Tuple containing list of ids and dictionary of columns.
        """
        ids: List[str] = []
        questions: List[str] = []
        answers: List[str] = []
        contexts: List[List[str]] = []
        gts: List[str | None] = []

        for s in samples:
            ids.append(s.get("id"))  # type: ignore[arg-type]
            questions.append(str(s.get("question", "")))
            answers.append(str(s.get("predicted_answer", "")))
            ctx = s.get("contexts") or []
            contexts.append(list(ctx))
            gts.append(s.get("ground_truth"))

        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": gts,
        }
        return ids, data

    def _to_port_output(
        self, ids: Sequence[str], result: Any, metric_names: Sequence[str]
    ) -> List[Dict[str, Any]]:
        """Convert Ragas result to port output format.

        Args:
            ids (Sequence[str]): List of sample ids.
            result (Any): Ragas evaluation result.
            metric_names (Sequence[str]): Names of metrics computed.

        Returns:
            List of dictionaries with 'id' and 'metrics' keys.
        """
        try:
            df = result.to_pandas()
        except AttributeError:
            df = result

        out: List[Dict[str, Any]] = []
        for i, sid in enumerate(ids):
            row = df.iloc[i]
            row_dict = getattr(row, "to_dict", lambda: dict(row))()
            metrics: Dict[str, Any] = {}
            for name in metric_names:
                val = row_dict.get(name)
                if val != val:
                    val = None
                metrics[name] = float(val) if val is not None else None
            out.append({"id": sid, "metrics": metrics})
        return out

    def evaluate(self, samples: Sequence[Mapping]):
        """Evaluate samples using selected RAG metrics.

        Args:
            samples (Sequence[Mapping]): List of samples to evaluate.

        Returns:
            List of dictionaries with sample ids and their metric scores.

        Raises:
            RuntimeError: If Ragas or datasets are not installed.
            ValueError: If unsupported metrics are selected.
        """
        Dataset, evaluate_fn, metric_map = self._load_ragas()
        selected = self._validate_metrics(metric_map)

        ids, cols = self._build_columns(samples)
        ds = Dataset.from_dict(cols)

        result = evaluate_fn(
            ds,
            metrics=selected,
            llm=self.judge_llm,
            embeddings=self.embeddings,
        )
        return self._to_port_output(ids, result, self.metrics)
