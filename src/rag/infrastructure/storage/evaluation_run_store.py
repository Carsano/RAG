from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd

from src.rag.application.ports.evaluation_storage import (
    Storage,
    RunHandle,
    RunModels
)


def _compute_summary(
    df_samples: pd.DataFrame,
    metric_cols: List[str],
) -> Dict[str, float]:
    """Compute mean per metric, ignoring non-numeric/NaN."""
    out: Dict[str, float] = {}
    for col in metric_cols:
        if col in df_samples.columns:
            mean = pd.to_numeric(df_samples[col], errors="coerce").mean()
            if pd.notna(mean):
                out[col] = float(mean)
    return out


class _FsRunHandle(RunHandle):
    """Filesystem-backed run handle."""

    def __init__(self, root: pathlib.Path) -> None:
        self._root = root

    def root(self) -> pathlib.Path:
        return self._root

    def _pass_dir(self, pass_idx: int) -> pathlib.Path:
        d = self._root / f"pass{pass_idx}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_pass(
        self,
        pass_idx: int,
        df_samples: pd.DataFrame,
        metric_cols: List[str],
        models: RunModels,
        llm_max_retries: int,
        retry_passes_done: int,
        retry_sleep_seconds: int,
    ) -> pathlib.Path:
        """Persist one pass (CSV + summary + manifest JSON)."""
        pdir = self._pass_dir(pass_idx)

        # per-sample
        per_sample_csv = pdir / "per_sample.csv"
        df_samples.to_csv(per_sample_csv, index=False)

        # summary
        summary = _compute_summary(df_samples, metric_cols)
        (pdir / "summary.csv").write_text(
            pd.Series(summary).to_csv(), encoding="utf-8"
        )

        # manifest
        failed = int(df_samples[metric_cols].isna().any(axis=1).sum())
        total = int(len(df_samples))
        manifest = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "models": {"llm": models.llm, "embeddings": models.embeddings},
            "llm_max_retries": int(llm_max_retries),
            "retry_passes": int(retry_passes_done),
            "retry_sleep": int(retry_sleep_seconds),
            "items_total": total,
            "items_success": total - failed,
            "items_failed": failed,
            "metrics": summary,
        }
        with open(
            pdir / f"run_manifest_pass_{pass_idx}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        return pdir


class FileSystemStorage(Storage):
    """Filesystem implementation of the Storage port.

    Layout:
      <root>/<date_str>/pass_1/...
      <root>/<date_str>/pass_2/...
    """

    def __init__(self, root: pathlib.Path | str) -> None:
        self._root = pathlib.Path(root)

    def create_run(self, date_str: str) -> RunHandle:
        """Create or open a new nested attempt directory for the given date."""
        day_dir = self._root / date_str
        day_dir.mkdir(parents=True, exist_ok=True)

        # Find max NNN among subdirs named <date_str>-NNN
        max_idx = 0
        prefix = f"{date_str}-"
        for child in day_dir.iterdir():
            if child.is_dir() and child.name.startswith(prefix):
                suffix = child.name[len(prefix):]
                if len(suffix) == 3 and suffix.isdigit():
                    idx = int(suffix)
                    if idx > max_idx:
                        max_idx = idx
        next_idx = max_idx + 1
        attempt_name = f"{date_str}-{next_idx:03d}"
        attempt_dir = day_dir / attempt_name
        attempt_dir.mkdir(parents=True, exist_ok=False)
        return _FsRunHandle(attempt_dir)
