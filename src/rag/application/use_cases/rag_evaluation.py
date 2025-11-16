import json
import pathlib
import time
import os
from datetime import datetime, timezone
from typing import List, Dict, Tuple

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from langchain_mistralai.embeddings import MistralAIEmbeddings

from src.rag.infrastructure.llm.langchain_mistral_client import LGMistralLLM


DEFAULT_INPUT_PATH = "logs/interactions/interactions.jsonl"
DEFAULT_OUT_ROOT = "logs/evaluations/ragas_eval"


LLM_MAX_RETRIES = 3
RETRY_PASSES = 2
RETRY_SLEEP_SECONDS = 60


def _load_and_normalize_jsonl(path: str) -> List[Dict]:
    """Load interactions from a JSONL file and normalize fields.

    The function standardizes the schema to what ragas expects:
    - `contexts` is always a list of strings.
    - `ground_truths` is always a list (possibly empty).

    Args:
        path: Path to the JSONL file.

    Returns:
        A list of dictionaries with keys: question, answer, contexts,
        ground_truths.
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
            gt = obj.get("ground_truth") or obj.get("ground_truths") or None

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


def _compute_failed_mask(df_res: pd.DataFrame, metric_cols: List[str]
                         ) -> pd.Series:
    """Return a mask for rows that have at least one NaN metric.

    Args:
        df_res: Per-sample evaluation DataFrame.
        metric_cols: List of metric column names to check for NaN.

    Returns:
        A boolean Series where True indicates a failed row.
    """
    return df_res[metric_cols].isna().any(axis=1)


def _compute_summary(df_samples: pd.DataFrame, metric_cols: List[str]) -> Dict:
    """Compute mean scores per metric, ignoring non-numeric/NaN values.

    Args:
        df_samples: Per-sample evaluation DataFrame.
        metric_cols: Metric columns to aggregate.

    Returns:
        A dictionary mapping metric name to mean score.
    """
    summary: Dict[str, float] = {}
    for col in metric_cols:
        if col in df_samples.columns:
            col_mean = pd.to_numeric(df_samples[col], errors="coerce").mean()
            if pd.notna(col_mean):
                summary[col] = float(col_mean)
    return summary


def _init_clients() -> Tuple[LGMistralLLM, MistralAIEmbeddings]:
    """Initialize Mistral LLM and embedding clients from environment.

    Returns:
        Tuple of (LGMistralLLM, MistralAIEmbeddings).

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
        max_retries=LLM_MAX_RETRIES,
    )
    llm = llm_port.as_langchain()
    emb = MistralAIEmbeddings(model=embed_model, api_key=api_key)
    return llm, emb


def _select_metrics(has_reference: bool) -> List:
    """Select evaluation metrics based on reference availability.

    Args:
        has_reference: True if references are available for samples.

    Returns:
        A list of ragas metrics to evaluate.
    """
    metrics = [faithfulness, answer_relevancy]
    if has_reference:
        metrics += [context_precision, context_recall]
    return metrics


def _prepare_dataset(rows: List[Dict]) -> Tuple[pd.DataFrame, Dataset, bool]:
    """Build the evaluation DataFrame and HF Dataset from normalized rows.

    The function also derives a `reference` column when `ground_truths` is
    present, as some ragas versions expect a text reference field.

    Args:
        rows: Normalized interaction rows.

    Returns:
        A tuple of (pandas DataFrame with row_id, HF Dataset, has_reference).
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
                lambda lst: " ".join(lst) if isinstance(lst,
                                                        list) else (lst or "")
            )

    ds = Dataset.from_pandas(df, preserve_index=False)
    return df, ds, has_reference


# -----------------------------------------------------------------------------
# Evaluation passes & I/O
# -----------------------------------------------------------------------------


def _run_single_pass(
    ds: Dataset,
    metrics: List,
    llm: LGMistralLLM,
    emb: MistralAIEmbeddings,
) -> pd.DataFrame:
    """Execute a single evaluation pass and return the per-sample DataFrame.

    Args:
        ds: Dataset to evaluate.
        metrics: Metrics to compute.
        llm: Chat model client.
        emb: Embedding model client.

    Returns:
        A DataFrame of per-sample metrics from ragas.
    """
    scores = evaluate(
        ds,
        metrics=metrics,
        llm=llm,
        embeddings=emb,
        raise_exceptions=False,
    )
    return scores.to_pandas()


def _write_pass_outputs(
    outdir: pathlib.Path,
    pass_idx: int,
    df_samples: pd.DataFrame,
    metric_cols: List[str],
    llm: LGMistralLLM,
    emb: MistralAIEmbeddings,
) -> None:
    """Write per-sample, summary, and manifest for a given pass.

    Args:
        outdir: Root output directory for the run date.
        pass_idx: Pass index starting at 1.
        df_samples: Per-sample results to persist.
        metric_cols: Metric columns to summarize.
        llm: Chat model client (for manifest).
        emb: Embedding model client (for manifest).
    """
    pass_dir = outdir / f"pass_{pass_idx}"
    pass_dir.mkdir(parents=True, exist_ok=True)

    df_samples.to_csv(pass_dir / "per_sample.csv", index=False)
    summary = _compute_summary(df_samples, metric_cols)
    pd.Series(summary).to_csv(pass_dir / "summary.csv")

    total = len(df_samples)
    failed = int(df_samples[metric_cols].isna().any(axis=1).sum())
    success = total - failed

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": {"llm": llm.model, "embeddings": emb.model},
        "llm_max_retries": LLM_MAX_RETRIES,
        "retry_passes": pass_idx - 1,
        "retry_sleep": RETRY_SLEEP_SECONDS,
        "items_total": int(total),
        "items_success": int(success),
        "items_failed": int(failed),
        "metrics": summary,
    }
    with open(
        pass_dir / f"run_manifest_pass_{pass_idx}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def _retry_failed_items(
    base_df: pd.DataFrame,
    initial_df_samples: pd.DataFrame,
    metrics: List,
    metric_cols: List[str],
    llm: LGMistralLLM,
    emb: MistralAIEmbeddings,
    outdir: pathlib.Path,
    start_pass_idx: int,
) -> pd.DataFrame:
    """Retry only failed items across several passes and persist outputs.

    Args:
        base_df: Full input DataFrame with `row_id`.
        initial_df_samples: Per-sample results from the initial pass.
        metrics: Metrics to compute.
        metric_cols: Metric columns to check/merge.
        llm: Chat model client.
        emb: Embedding model client.
        outdir: Root output directory for dated run.
        start_pass_idx: The index of the initial pass (usually 1).

    Returns:
        The merged per-sample DataFrame after all retry passes.
    """
    df_samples = initial_df_samples.copy()

    if "row_id" not in df_samples.columns and "row_id" in base_df.columns:
        df_samples.insert(0, "row_id", base_df["row_id"])

    pass_idx = start_pass_idx
    for _ in range(RETRY_PASSES):
        mask = _compute_failed_mask(df_samples, metric_cols)
        if not bool(mask.any()):
            break

        failed_df = base_df.merge(
            df_samples.loc[mask, ["row_id"]],
            on="row_id",
            how="inner",
        )
        failed_ds = Dataset.from_pandas(failed_df, preserve_index=False)

        time.sleep(RETRY_SLEEP_SECONDS)

        df_retry = _run_single_pass(failed_ds, metrics, llm, emb)
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
                df_samples[col] = df_samples[col].fillna(df_samples[retry_col])
                df_samples.drop(columns=[retry_col], inplace=True)

        pass_idx += 1
        _write_pass_outputs(outdir, pass_idx, df_samples, metric_cols,
                            llm, emb)

    return df_samples


# -----------------------------------------------------------------------------
# Orchestration (main)
# -----------------------------------------------------------------------------


def main() -> None:
    """Run the full evaluation workflow with dated, per-pass outputs.

    Steps:
      1) Load and normalize input interactions.
      2) Initialize LLM and embedding clients.
      3) Prepare DataFrame and Dataset; select metrics.
      4) Run initial evaluation pass and write outputs.
      5) Retry failed items for a few passes, writing outputs per pass.
    """
    load_dotenv()

    rows = _load_and_normalize_jsonl(DEFAULT_INPUT_PATH)
    if not rows:
        raise SystemExit("No valid lines found in the JSONL input.")

    df, ds, has_reference = _prepare_dataset(rows)
    metrics = _select_metrics(has_reference)
    metric_cols = [m.name for m in metrics]

    llm, emb = _init_clients()

    date_str = datetime.now().strftime("%Y-%m-%d")
    outdir = pathlib.Path(f"{DEFAULT_OUT_ROOT}/{date_str}")
    outdir.mkdir(parents=True, exist_ok=True)

    # Initial pass (pass 1)
    df_samples = _run_single_pass(ds, metrics, llm, emb)
    _write_pass_outputs(outdir, 1, df_samples, metric_cols, llm, emb)

    # Retry passes (pass 2..N)
    df_samples = _retry_failed_items(
        base_df=df,
        initial_df_samples=df_samples,
        metrics=metrics,
        metric_cols=metric_cols,
        llm=llm,
        emb=emb,
        outdir=outdir,
        start_pass_idx=1,
    )


if __name__ == "__main__":
    main()
