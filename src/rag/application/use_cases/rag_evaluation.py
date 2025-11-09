"""
Module for RAG evaluation.
"""
# src/rag/application/use_cases/rag_evaluation.py
from typing import Sequence
from src.rag.application.ports.evaluation import (
    EvaluationEngine,
    EvaluationResultsSink,
    EvaluationSample
)
from src.rag.application.ports.retriever import Retriever
from src.rag.application.ports.llm import LLM
from src.rag.infrastructure.config.config import SYSTEM_PROMPT


class RAGEvaluation:
    def __init__(self, retriever: Retriever, generator: LLM,
                 engine: EvaluationEngine, sink: EvaluationResultsSink):
        self.retriever = retriever
        self.generator = generator
        self.engine = engine
        self.sink = sink

    def run(self, items: Sequence[dict], run_meta: dict) -> list[dict]:
        samples: list[EvaluationSample] = []
        system_prompt = run_meta.get("system_prompt", SYSTEM_PROMPT)
        for it in items:
            ctx = self.retriever.search(it["question"])
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: "
                 f"{it['question']}\nContext:\n" + "\n".join(ctx)}
            ]
            answer = self.generator.chat(messages)
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
