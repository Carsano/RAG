"""CLI to evaluate RAG answers with a pluggable evaluation engine.

This adapter is responsible only for:
- Parsing CLI arguments.
- Loading dependency factories from dotted paths.
- Wiring ports and implementations.
- Orchestrating the evaluation use case and handling I/O.

All domain logic remains in application and infrastructure layers.

Usage example:
    uv run python -m src.rag.adapters.cli.evaluate_rag_answers \
        --questions data/eval/questions.jsonl \
        --out logs/eval/2025-11-09 \
        --model mistral-7b \
        --index-hash 1f2c9a \
        --retriever-factory src.rag.infrastructure.builders:build_retriever \
        --generator-factory src.rag.infrastructure.builders:build_generator \
        --judge-llm-factory src.rag.infrastructure.builders:build_judge_llm
"""
from __future__ import annotations

import argparse
import importlib
import io
import json
import logging
import sys
from pathlib import Path
from typing import Callable, List, Mapping, MutableMapping, Sequence

from src.rag.application.use_cases.rag_evaluation import RAGEvaluation
from src.rag.infrastructure.evaluation.csv_sink import CSVSink
from src.rag.infrastructure.evaluation.ragas_engine import (
    RagasEvaluationEngine
)

logger = logging.getLogger(__name__)


def _setup_logging(verbosity: int) -> None:
    """Configure logging for the CLI.

    Args:
        verbosity: Verbosity level. 0=WARNING, 1=INFO, 2=DEBUG.
    """
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _load_object(dotted_path: str) -> Callable:
    """Load a callable from a dotted path like 'pkg.module:factory'.

    Args:
        dotted_path: Dotted path in the format 'module.sub:attr'.

    Returns:
        Loaded callable.
    """
    if ":" not in dotted_path:
        raise ValueError("Invalid path. Use 'module.sub:attr' format.")
    module_path, attr_name = dotted_path.split(":", 1)
    module = importlib.import_module(module_path)
    obj = getattr(module, attr_name)
    if not callable(obj):
        raise ValueError(f"Object at '{dotted_path}' is not callable.")
    return obj


def _read_jsonl(path: Path) -> List[MutableMapping[str, object]]:
    """Read a JSON Lines file into a list of dicts.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of items read from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If a line is not valid JSON.
    """
    items: List[MutableMapping[str, object]] = []
    with io.open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Invalid JSON at line {line_no}: {e.msg}",
                    e.doc,
                    e.pos,
                ) from e
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} must be a JSON object.")
            items.append(obj)  # type: ignore[arg-type]
    return items


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument vector. Defaults to sys.argv[1:].

    Returns:
        Parsed arguments namespace.
    """
    p = argparse.ArgumentParser(
        description="Evaluate RAG answers using Ragas "
        "via clean architecture ports.",
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
        required=True,
        help="Generator model name to record in run metadata.",
    )
    p.add_argument(
        "--index-hash",
        required=True,
        help="Identifier of the retrieval index used. "
        "Recorded in run metadata.",
    )
    p.add_argument(
        "--retriever-factory",
        required=True,
        help="Dotted path to a callable that builds the retriever. "
        "Format: module.sub:func",
    )
    p.add_argument(
        "--generator-factory",
        required=True,
        help="Dotted path to a callable that builds the generator. "
        "Format: module.sub:func",
    )
    p.add_argument(
        "--judge-llm-factory",
        required=True,
        help="Dotted path to a callable that builds the judge LLM for Ragas. "
        "Format: module.sub:func",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        default=["faithfulness", "answer_relevancy", "context_relevancy",
                 "context_precision"],
        help="List of Ragas metrics to compute.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for components that support it. "
        "Recorded in run metadata.",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity. -v for INFO, -vv for DEBUG.",
    )
    return p.parse_args(argv)


def _validate_io(args: argparse.Namespace) -> None:
    """Validate input paths and prepare output directory.

    Args:
        args: Parsed CLI arguments.

    Raises:
        FileNotFoundError: If questions file does not exist.
    """
    if not args.questions.exists():
        raise FileNotFoundError(f"Questions file not found: {args.questions}")
    args.out.mkdir(parents=True, exist_ok=True)


def _load_factories(args: argparse.Namespace
                    ) -> tuple[Callable, Callable, Callable]:
    """Load dependency factories from dotted paths.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Tuple of callables (build_retriever, build_generator, build_judge_llm).

    Raises:
        Exception: If loading fails or paths are invalid.
    """
    build_retriever = _load_object(args.retriever_factory)
    build_generator = _load_object(args.generator_factory)
    build_judge_llm = _load_object(args.judge_llm_factory)
    return build_retriever, build_generator, build_judge_llm


def _instantiate_components(
    args: argparse.Namespace,
    build_retriever: Callable,
    build_generator: Callable,
    build_judge_llm: Callable,
) -> tuple[object, object, RagasEvaluationEngine, CSVSink]:
    """Instantiate retriever, generator, evaluation engine, and sink.

    Args:
        args: Parsed CLI arguments.
        build_retriever: Factory for retriever.
        build_generator: Factory for generator.
        build_judge_llm: Factory for judge LLM.

    Returns:
        Tuple of (retriever, generator, engine, sink).
    """
    retriever = build_retriever()
    generator = build_generator(args.model)
    judge_llm = build_judge_llm()
    engine = RagasEvaluationEngine(judge_llm=judge_llm,
                                   metrics=tuple(args.metrics))
    sink = CSVSink(str(args.out))
    return retriever, generator, engine, sink


def _build_run_meta(args: argparse.Namespace) -> Mapping[str, object]:
    """Build run metadata dictionary from arguments.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Mapping with metadata about the run.
    """
    return {
        "model": args.model,
        "index_hash": args.index_hash,
        "seed": args.seed,
        "metrics": args.metrics,
    }


def _read_items(args: argparse.Namespace
                ) -> list[MutableMapping[str, object]]:
    """Read evaluation items from the questions JSONL file.

    Args:
        args: Parsed CLI arguments.

    Returns:
        List of items.
    """
    return _read_jsonl(args.questions)


def _execute_use_case(
    retriever: object,
    generator: object,
    engine: RagasEvaluationEngine,
    sink: CSVSink,
    items: list[MutableMapping[str, object]],
    run_meta: Mapping[str, object],
) -> list[MutableMapping[str, object]]:
    """Run the evaluation use case and return scores.

    Args:
        retriever: Retriever implementation.
        generator: Generator implementation.
        engine: Evaluation engine implementation.
        sink: Results sink implementation.
        items: Evaluation items.
        run_meta: Run metadata.

    Returns:
        List of score dicts.
    """
    use_case = RAGEvaluation(
        retriever=retriever,
        generator=generator,
        engine=engine,
        sink=sink,
    )
    return use_case.run(items, dict(run_meta))


def _print_summary(scores: Sequence[MutableMapping[str, object]],
                   out_dir: Path) -> None:
    """Print a concise summary of evaluation results.

    Args:
        scores: Sequence of score dicts.
        out_dir: Output directory where artifacts were written.
    """
    total = len(scores)
    print(f"Evaluated {total} items.")
    if total == 0:
        print(f"Written artifacts to: {out_dir}")
        return

    first = scores[0]
    metric_keys = list(first.get("metrics", {}).keys())
    agg: dict[str, float] = {k: 0.0 for k in metric_keys}
    for s in scores:
        metrics = s.get("metrics", {})
        for k in metric_keys:
            agg[k] += float(metrics.get(k, 0.0))
    for k in agg:
        agg[k] /= total
    print("Averages:")
    for k in sorted(agg):
        print(f"  {k}: {agg[k]:.3f}")
    print(f"Written artifacts to: {out_dir}")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the evaluation CLI.

    Args:
        argv: Optional argument vector. Defaults to sys.argv[1:].

    Returns:
        Process exit code. 0 on success, non-zero on failure.
    """
    args = parse_args(argv)
    _setup_logging(args.verbose)
    logger.debug("Arguments: %s", vars(args))

    try:
        _validate_io(args)
        factories = _load_factories(args)
        (retriever, generator,
         engine, sink) = _instantiate_components(args, *factories)
        run_meta = _build_run_meta(args)
        items = _read_items(args)
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
