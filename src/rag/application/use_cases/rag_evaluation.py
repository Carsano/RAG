"""Use case orchestration for offline RAG evaluation runs."""

import json
import time
import os
from datetime import datetime
from typing import List, Dict, Tuple

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from src.rag.application.ports.evaluator import (
    Evaluator,
    EvaluatorOptions,
)
from src.rag.infrastructure.evaluation.ragas_evaluator import RagasEvaluator

from src.rag.infrastructure.llm.langchain_mistral_client import LGMistralLLM
from src.rag.infrastructure.embedders.mistral_embedder import MistralEmbedder

from src.rag.application.ports.evaluation_storage import RunModels
from src.rag.infrastructure.storage.evaluation_run_store import (
    FileSystemStorage,
)


class RagEvaluationUseCase:
    """Run RAG evaluation with retries and dated per-pass outputs.

    This use case owns the retry and clock logic. Adapters are responsible
    only for wiring dependencies and calling `run()`.

    Attributes:
        input_path: Path to the JSONL interactions file.
        out_root: Root output directory for evaluation runs.
        llm_max_retries: Max retries used by the LLM client.
        retry_passes: Number of retry passes for failed items.
        retry_sleep_seconds: Sleep between retry passes, in seconds.
    """

    def __init__(
        self,
        input_path: str = "logs/interactions/interactions.jsonl",
        out_root: str = "logs/evaluations/ragas_eval",
        llm_max_retries: int = 3,
        retry_passes: int = 2,
        retry_sleep_seconds: int = 60,
    ) -> None:
        self.input_path = input_path
        self.out_root = out_root
        self.llm_max_retries = llm_max_retries
        self.retry_passes = retry_passes
        self.retry_sleep_seconds = retry_sleep_seconds

    def _load_and_normalize_jsonl(self, path: str) -> List[Dict]:
        """Load interactions from JSONL and normalize fields.

        The schema is standardized for ragas:
        - `contexts` is always a list of strings.
        - `ground_truths` is always a list (possibly empty).

        Args:
            path: Path to the JSONL file.

        Returns:
            A list of dicts with: question, answer, contexts, ground_truths.
        """
        rows: List[Dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                q = obj.get("question")
                a = obj.get("answer")
                ctx = obj.get("contexts") or obj.get("context") or []
                gt = obj.get("ground_truth") or obj.get(
                    "ground_truths"
                ) or None

                if isinstance(ctx, str):
                    ctx = [ctx]
                if gt is None:
                    gt = []
                elif isinstance(gt, str):
                    gt = [gt]

                rows.append(
                    {
                        "question": q,
                        "answer": a,
                        "contexts": ctx,
                        "ground_truths": gt,
                    }
                )
        return rows

    @staticmethod
    def _compute_failed_mask(
        df_res: pd.DataFrame, metric_cols: List[str]
    ) -> pd.Series:
        """Return mask where at least one metric is NaN.

        Args:
            df_res: Per-sample evaluation DataFrame.
            metric_cols: Metric column names to check.

        Returns:
            Boolean Series where True marks a failed row.
        """
        return df_res[metric_cols].isna().any(axis=1)

    def _init_clients(self) -> Tuple[LGMistralLLM, MistralEmbedder]:
        """Initialize Mistral LLM and embedding clients from environment.

        Returns:
            Tuple of (LGMistralLLM, MistralEmbedder).

        Raises:
            SystemExit: If required environment variables are missing.
        """
        chat_model = os.getenv("MISTRAL_CHAT_MODEL")
        embed_model = os.getenv("MISTRAL_EMBED_MODEL")
        api_key = os.getenv("MISTRAL_API_KEY")

        if not api_key:
            raise SystemExit("Missing MISTRAL_API_KEY in environment.")
        if not chat_model:
            raise SystemExit("Missing MISTRAL_CHAT_MODEL in environment.")
        if not embed_model:
            raise SystemExit("Missing MISTRAL_EMBED_MODEL in environment.")

        llm_port = LGMistralLLM(
            model=chat_model,
            api_key=api_key,
            temperature=0.0,
            max_retries=self.llm_max_retries,
        )
        llm = llm_port.as_langchain()
        emb = MistralEmbedder(model=embed_model)
        return llm, emb

    @staticmethod
    def _init_evaluator() -> Evaluator:
        """Return the evaluator implementation."""
        return RagasEvaluator()

    @staticmethod
    def _select_metrics(has_reference: bool) -> List:
        """Select metrics based on reference availability.

        Args:
            has_reference: True if references exist.

        Returns:
            List of metric callables for ragas.
        """
        metrics = [faithfulness, answer_relevancy]
        if has_reference:
            metrics += [context_precision, context_recall]
        return metrics

    @staticmethod
    def _prepare_dataset(
        rows: List[Dict],
    ) -> Tuple[pd.DataFrame, Dataset, bool]:
        """Build DataFrame and HF Dataset from normalized rows.

        If `ground_truths` is present, derive a `reference` column that some
        ragas versions expect.

        Args:
            rows: Normalized interaction rows.

        Returns:
            Tuple: (DataFrame with row_id, HF Dataset, has_reference).
        """
        df = pd.DataFrame(rows)
        df.insert(0, "row_id", range(len(df)))

        has_reference = False
        if "ground_truths" in df.columns:
            has_reference = df["ground_truths"].map(
                lambda x: isinstance(x, list) and len(x) > 0
            ).any()
            if has_reference and "reference" not in df.columns:
                df["reference"] = df["ground_truths"].apply(
                    lambda lst: " ".join(lst)
                    if isinstance(lst, list)
                    else (lst or "")
                )

        ds = Dataset.from_pandas(df, preserve_index=False)
        return df, ds, has_reference

    @staticmethod
    def _run_single_pass(
        ds: Dataset,
        metrics: List,
        llm: LGMistralLLM,
        emb: MistralEmbedder,
        evaluator: Evaluator,
    ) -> pd.DataFrame:
        """Execute a single evaluation pass.

        Args:
            ds: Dataset to evaluate.
            metrics: Metrics to compute.
            llm: Chat model client.
            emb: Embedding model client.
            evaluator: Evaluator implementation.

        Returns:
            DataFrame of per-sample metrics from ragas.
        """
        opts = EvaluatorOptions(raise_exceptions=False)
        return evaluator.evaluate(
            ds=ds,
            metrics=metrics,
            llm=llm,
            embeddings=emb,
            options=opts,
        )

    def _retry_failed_items(
        self,
        base_df: pd.DataFrame,
        initial_df_samples: pd.DataFrame,
        metrics: List,
        metric_cols: List[str],
        llm: LGMistralLLM,
        emb: MistralEmbedder,
        evaluator: Evaluator,
        run,
        start_pass_idx: int,
    ) -> pd.DataFrame:
        """Retry failed items for a few passes and persist outputs.

        Args:
            base_df: Full input DataFrame with `row_id`.
            initial_df_samples: Per-sample results from the first pass.
            metrics: Metrics to compute.
            metric_cols: Metric columns to check/merge.
            llm: Chat model client.
            emb: Embedding model client.
            evaluator: Evaluator implementation.
            run: Run storage handle to write outputs.
            start_pass_idx: Index of the initial pass (usually 1).

        Returns:
            Merged per-sample DataFrame after retry passes.
        """
        df_samples = initial_df_samples.copy()

        if "row_id" not in df_samples.columns and "row_id" in base_df.columns:
            df_samples.insert(0, "row_id", base_df["row_id"])

        pass_idx = start_pass_idx
        for _ in range(self.retry_passes):
            mask = self._compute_failed_mask(df_samples, metric_cols)
            if not bool(mask.any()):
                break

            failed_df = base_df.merge(
                df_samples.loc[mask, ["row_id"]],
                on="row_id",
                how="inner",
            )
            failed_ds = Dataset.from_pandas(
                failed_df, preserve_index=False
            )

            time.sleep(self.retry_sleep_seconds)

            df_retry = self._run_single_pass(
                failed_ds, metrics, llm, emb, evaluator
            )
            if "row_id" not in df_retry.columns:
                df_retry.insert(
                    0,
                    "row_id",
                    failed_df["row_id"].reset_index(drop=True),
                )

            df_samples = df_samples.merge(
                df_retry[["row_id"] + metric_cols],
                on="row_id",
                how="left",
                suffixes=("", "_retry"),
            )
            for col in metric_cols:
                retry_col = f"{col}_retry"
                if retry_col in df_samples.columns:
                    df_samples[col] = df_samples[col].fillna(
                        df_samples[retry_col]
                    )
                    df_samples.drop(columns=[retry_col], inplace=True)

            pass_idx += 1
            run.write_pass(
                pass_idx=pass_idx,
                df_samples=df_samples,
                metric_cols=metric_cols,
                models=RunModels(
                    llm=getattr(llm, "model", str(llm)),
                    embeddings=getattr(emb, "model", str(emb)),
                ),
                llm_max_retries=self.llm_max_retries,
                retry_passes_done=pass_idx - 1,
                retry_sleep_seconds=self.retry_sleep_seconds,
            )

        return df_samples

    def run(self) -> None:
        """Execute the evaluation workflow end-to-end.

        Steps:
          1) Load and normalize interactions.
          2) Initialize clients and evaluator.
          3) Build dataset and select metrics.
          4) Run initial pass and persist outputs.
          5) Retry failed items; persist outputs per pass.

        Raises:
            SystemExit: If the input contains no valid rows.
        """
        load_dotenv()

        rows = self._load_and_normalize_jsonl(self.input_path)
        if not rows:
            raise SystemExit("No valid lines found in the JSONL input.")

        df, ds, has_reference = self._prepare_dataset(rows)
        metrics = self._select_metrics(has_reference)
        metric_cols = [m.name for m in metrics]

        llm, emb = self._init_clients()
        evaluator = self._init_evaluator()

        date_str = datetime.now().strftime("%Y-%m-%d")
        storage = FileSystemStorage(self.out_root)
        run = storage.create_run(date_str)

        # Initial pass (pass 1)
        df_samples = self._run_single_pass(ds, metrics, llm, emb, evaluator)
        run.write_pass(
            pass_idx=1,
            df_samples=df_samples,
            metric_cols=metric_cols,
            models=RunModels(
                llm=getattr(llm, "model", str(llm)),
                embeddings=getattr(emb, "model", str(emb)),
            ),
            llm_max_retries=self.llm_max_retries,
            retry_passes_done=0,
            retry_sleep_seconds=self.retry_sleep_seconds,
        )

        # Retry passes (pass 2..N)
        self._retry_failed_items(
            base_df=df,
            initial_df_samples=df_samples,
            metrics=metrics,
            metric_cols=metric_cols,
            llm=llm,
            emb=emb,
            evaluator=evaluator,
            run=run,
            start_pass_idx=1,
        )
