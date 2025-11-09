"""
Module for RAG evaluation.
"""

from typing import Sequence
from rag.application.ports.evaluation import EvaluationEngine
from rag.application.ports.evaluation import EvaluationResultsSink
from rag.application.ports.evaluation import EvaluationSample


class RAGEvaluation:
    def __init__(self, retriever, generator, engine: EvaluationEngine,
                 sink: EvaluationResultsSink):
        self.retriever = retriever
        self.generator = generator
        self.engine = engine
        self.sink = sink

    def run(self, items: Sequence[dict], run_meta: dict) -> list[dict]:
        samples: list[EvaluationSample] = []
        for it in items:
            ctx = self.retriever.retrieve(it["question"])
            answer = self.generator.generate(it["question"], ctx)
            samples.append({
                "id": it["id"],
                "question": it["question"],
                "predicted_answer": answer,
                "contexts": ctx,
                "ground_truth": it.get("ground_truth")
            })

        scores = self.engine.evaluate(samples)
        self.sink.save(run_meta, scores)
        return scores
