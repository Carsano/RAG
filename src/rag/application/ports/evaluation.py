"""
Module for evaluation ports.
"""
from typing import TypedDict, Sequence, Protocol


class EvaluationSample(TypedDict):
    id: str
    question: str
    predicted_answer: str
    contexts: list[str]
    ground_truth: str | None


class EvaluationScores(TypedDict):
    id: str
    metrics: dict[str, float]


class EvaluationEngine(Protocol):
    def evaluate(self, samples: Sequence[EvaluationSample]
                 ) -> list[EvaluationScores]: ...


class EvaluationResultsSink(Protocol):
    def save(self, run_meta: dict, scores: Sequence[EvaluationScores]
             ) -> None: ...
