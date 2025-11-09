"""
CSV sink for evaluation results.

This module implements a concrete `EvaluationResultsSink` that persists
artifacts to a directory using JSON and CSV files. The class is designed to be
simple, testable, and to comply with SOLID principles:
"""
from __future__ import annotations

from pathlib import Path
import csv
import json
import os
from typing import Any, Mapping, Sequence

from rag.application.ports.evaluation import (
    EvaluationResultsSink,
    EvaluationScores,
)


class CSVSink(EvaluationResultsSink):
    """Persist evaluation outputs as CSV and JSON.

    Files written into ``out_dir``:
      * ``run_meta.json``: metadata about the evaluation run.
      * ``scores.csv``: long-format table with columns ``id,metric,value``.

    The sink creates ``out_dir`` if it does not exist and writes files
    atomically to reduce the risk of corruption.

    Args:
        out_dir: Target directory where artifacts will be written.
        csv_filename: Filename for the scores CSV file.
        meta_filename: Filename for the run metadata JSON file.
        csv_delimiter: Delimiter used for the CSV writer.
        newline: Value passed to ``open(..., newline=newline)`` for the CSV
            file. Keep default for RFC-compliant CSV.
    """

    def __init__(
        self,
        out_dir: str | Path,
        *,
        csv_filename: str = "scores.csv",
        meta_filename: str = "run_meta.json",
        csv_delimiter: str = ",",
        newline: str = "",
    ) -> None:
        """Initialize a CSV sink.

        Args:
            out_dir: Target directory where artifacts will be written.
            csv_filename: Filename for the scores CSV file.
            meta_filename: Filename for the run metadata JSON file.
            csv_delimiter: Delimiter used for the CSV writer.
            newline: Value forwarded to ``open(..., newline=newline)`` when
                writing the CSV file.
        """
        self._out_dir = Path(out_dir)
        self._csv_path = self._out_dir / csv_filename
        self._meta_path = self._out_dir / meta_filename
        self._csv_delimiter = csv_delimiter
        self._newline = newline

    def save(self, run_meta: Mapping[str, Any],
             scores: Sequence[EvaluationScores]) -> None:
        """Save evaluation artifacts to disk.

        Args:
            run_meta: Mapping with metadata such as model, prompt hash, index
                hash, seed, and timestamps. Must be JSON serializable.
            scores: Sequence of per-sample scores. Each item contains an ``id``
                and a ``metrics`` mapping from metric name to numeric value.

        Raises:
            ValueError: If ``scores`` contains entries without an ``id`` or
                ``metrics``.
            OSError: If the filesystem operations fail.
            TypeError: If ``run_meta`` is not JSON serializable.
        """
        self._ensure_dir()
        self._write_meta(run_meta)
        self._write_scores(scores)

    def _ensure_dir(self) -> None:
        """Create the output directory if it does not exist.

        Raises:
            OSError: If the directory cannot be created.
        """
        self._out_dir.mkdir(parents=True, exist_ok=True)

    def _write_meta(self, run_meta: Mapping[str, Any]) -> None:
        """Write run metadata to ``run_meta.json`` atomically.

        Args:
            run_meta: Mapping containing run-level metadata. Must be JSON
                serializable.

        Raises:
            OSError: If the file cannot be written.
            TypeError: If ``run_meta`` is not JSON serializable.
        """
        payload = json.dumps(dict(run_meta), indent=2, ensure_ascii=False)
        self._atomic_write_text(self._meta_path, payload)

    def _write_scores(self, scores: Sequence[EvaluationScores]) -> None:
        """Write per-sample metric rows to the CSV file atomically.

        Args:
            scores: Sequence of score objects containing ``id``
            and ``metrics``.

        Raises:
            ValueError: If an item is missing ``id`` or ``metrics``.
            OSError: If the file cannot be written.
        """
        rows: list[tuple[str, str, float | int | str]] = []
        for item in scores:
            sid: Any = item.get("id")
            metrics: Mapping[str, Any] | None = item.get("metrics")

            if sid is None:
                raise ValueError("score item missing 'id'")
            if not isinstance(metrics, Mapping):
                raise ValueError("score item missing 'metrics' mapping")

            for metric_name, value in metrics.items():
                rows.append((str(sid), str(metric_name),
                             self._coerce_value(value)))

        tmp_path = self._csv_path.with_suffix(self._csv_path.suffix + ".tmp")
        with open(tmp_path, "w", newline=self._newline, encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=self._csv_delimiter)
            writer.writerow(["id", "metric", "value"])
            writer.writerows(rows)
        os.replace(tmp_path, self._csv_path)

    @staticmethod
    def _coerce_value(value: Any) -> float | int | str:
        """Coerce metric values to a CSV-safe primitive.

        Prefers ``float`` or ``int`` when possible. Falls back to ``str``.

        Args:
            value: Any metric value emitted by the evaluation engine.

        Returns:
            A primitive suitable for CSV serialization.
        """
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, int):
            return value
        try:
            return float(value)
        except Exception:
            return str(value)

    def _atomic_write_text(self, path: Path, content: str) -> None:
        """Atomically write a text payload to a path.

        The content is first written to ``<path>.tmp`` then moved in place with
        ``os.replace`` which is atomic on POSIX systems.

        Args:
            path: Destination path.
            content: UTF-8 text content to write.
        """
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, path)
