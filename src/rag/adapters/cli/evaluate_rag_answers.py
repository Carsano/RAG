"""Evaluate RAG answers — minimal, maintainable CLI.

Responsibilities:
- Parse args
- Validate I/O
- Wire concrete implementations
- Run the RAGEvaluation use case
- Print a concise summary

All domain logic stays in application/infrastructure layers.
"""
# src/rag/adapters/cli/evaluate_rag_answers.py
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path
from typing import List, Mapping, MutableMapping, Sequence

from src.rag.application.use_cases.rag_evaluation import RAGEvaluation
from src.rag.application.ports.embedders import MistralEmbedder

from src.rag.infrastructure.evaluation.csv_sink import CSVSink
from src.rag.infrastructure.evaluation.ragas_engine import (
    RagasEvaluationEngine
)
from src.rag.infrastructure.vectorstores.faiss_store import FaissStore
from src.rag.infrastructure.llm.mistral_client import MistralLLM

from src.rag.infrastructure.config.config import SYSTEM_PROMPT
from src.rag.infrastructure.retriever.vectorsore_retriever import (
    VectorStoreRetriever
)

logger = logging.getLogger(__name__)

# -----------------------------
# Parsing
# -----------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate RAG answers using Ragas with inline wiring.",
    )
    p.add_argument(
        "--questions",
        required=True,
        type=Path,
        help="Path to questions JSONL. "
        "Each line: {id, question, ground_truth?}.",
    )
    p.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory for scores and run metadata.",
    )
    p.add_argument(
        "--model",
        default="ministral-8b-latest",
        help="Generator/judge model name.",
    )
    p.add_argument(
        "--index-hash",
        required=True,
        help="Identifier of the retrieval index used (for traceability).",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        default=["faithfulness", "answer_relevancy",
                 "context_recall", "context_precision"],
        help="Ragas metrics to compute.",
    )
    p.add_argument(
        "--system-prompt",
        default=SYSTEM_PROMPT,
        help="Override system prompt. Defaults to config SYSTEM_PROMPT.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for components that support it (metadata only).",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity. -v for INFO, -vv for DEBUG.",
    )
    return p.parse_args(argv)


# -----------------------------
# I/O helpers
# -----------------------------

def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level,
                        format="%(asctime)s %(levelname)s "
                        "%(name)s - %(message)s")


def _validate_io(args: argparse.Namespace) -> None:
    if not args.questions.exists():
        raise FileNotFoundError(f"Questions file not found: {args.questions}")
    args.out.mkdir(parents=True, exist_ok=True)


def _read_jsonl(path: Path) -> List[MutableMapping[str, object]]:
    items: List[MutableMapping[str, object]] = []
    with io.open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Invalid JSON at line {line_no}: {e.msg}", e.doc, e.pos
                ) from e
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} must be a JSON object.")
            items.append(obj)
    return items


# -----------------------------
# Wiring helpers
# -----------------------------

_ALLOWED_METRICS = {"faithfulness", "answer_relevancy",
                    "context_recall", "context_precision"}
_ALIAS_METRICS = {"context_relevancy": "context_recall"}


def _normalize_metrics(metrics: Sequence[str]) -> list[str]:
    out: list[str] = []
    for m in metrics:
        m2 = _ALIAS_METRICS.get(m, m)
        if m2 not in _ALLOWED_METRICS:
            raise ValueError(f"Unknown metric: {m}")
        out.append(m2)
    return out


def _wire_components(args: argparse.Namespace):
    embedder = MistralEmbedder()
    store = FaissStore(
        index_path="data/indexes/faiss_index.idx",
        chunks_pickle_path="data/indexes/all_chunks.pkl",
    )
    retriever = VectorStoreRetriever(store=store, embedder=embedder)
    generator = MistralLLM(model_name=args.model)
    judge_llm = MistralLLM(model_name=args.model)
    engine = RagasEvaluationEngine(
        judge_llm=judge_llm,
        embeddings=embedder,
        metrics=tuple(_normalize_metrics(args.metrics)),
    )
    sink = CSVSink(str(args.out))
    return retriever, generator, engine, sink


def _build_run_meta(args: argparse.Namespace) -> Mapping[str, object]:
    return {
        "model": args.model,
        "index_hash": args.index_hash,
        "seed": args.seed,
        "metrics": _normalize_metrics(args.metrics),
        "system_prompt": args.system_prompt,
    }


# -----------------------------
# Use case execution
# -----------------------------

def _execute_use_case(
    retriever: object,
    generator: object,
    engine: RagasEvaluationEngine,
    sink: CSVSink,
    items: list[MutableMapping[str, object]],
    run_meta: Mapping[str, object],
) -> list[MutableMapping[str, object]]:
    use_case = RAGEvaluation(retriever=retriever, generator=generator,
                             engine=engine, sink=sink)
    return use_case.run(items, dict(run_meta))


def _print_summary(scores: Sequence[MutableMapping[str, object]],
                   out_dir: Path) -> None:
    total = len(scores)
    print(f"Evaluated {total} items.")
    if total == 0:
        print(f"Written artifacts to: {out_dir}")
        return

    first = scores[0]
    metric_keys = list(first.get("metrics", {}).keys())
    agg: dict[str, float] = {k: 0.0 for k in metric_keys}
    for s in scores:
        metrics = s.get("metrics", {}) or {}
        for k in metric_keys:
            v = metrics.get(k)
            if v is None:
                continue
            try:
                agg[k] += float(v)
            except (TypeError, ValueError):
                continue
    for k in agg:
        agg[k] /= total

    print("Averages:")
    for k in sorted(agg):
        print(f"  {k}: {agg[k]:.3f}")
    print(f"Written artifacts to: {out_dir}")


# -----------------------------
# Entry point
# -----------------------------

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.verbose)
    logger.debug("Arguments: %s", vars(args))

    try:
        _validate_io(args)
        retriever, generator, engine, sink = _wire_components(args)
        run_meta = _build_run_meta(args)
        items = _read_jsonl(args.questions)
        scores = _execute_use_case(retriever, generator, engine,
                                   sink, items, run_meta)
        _print_summary(scores, args.out)
        return 0
    except FileNotFoundError as e:
        logger.error(str(e))
        return 2
    except Exception as e:
        logger.exception("Evaluation failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
