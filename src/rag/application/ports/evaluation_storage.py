"""Ports describing persistence of evaluation runs."""

import pathlib
from dataclasses import dataclass
from typing import List, Protocol

import pandas as pd


@dataclass(frozen=True)
class RunModels:
    """Models used during the run."""
    llm: str
    embeddings: str


class RunHandle(Protocol):
    """Abstract handle over a persisted evaluation run."""

    def root(self) -> pathlib.Path:
        """Return the root directory of this run."""
        ...

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
        """Persist one pass (per-sample, summary, manifest).

        Args:
            pass_idx: 1-based pass index.
            df_samples: Per-sample metrics frame.
            metric_cols: Metric columns present in df_samples.
            models: Models identifiers.
            llm_max_retries: Max retries configured on LLM client.
            retry_passes_done: How many retry passes completed so far.
            retry_sleep_seconds: Sleep used between retry passes.

        Returns:
            Path to the pass directory created/written.
        """
        ...


class Storage(Protocol):
    """Abstract port to persist evaluation runs and passes."""

    def create_run(self, date_str: str) -> RunHandle:
        """Create or open a dated run root.

        Args:
            date_str: Date folder name, e.g., '2025-11-16'.

        Returns:
            A run handle to write passes and artifacts.
        """
        ...
