
"""Utilities for building structured logs for the RAG pipeline."""

import hashlib
import json
from datetime import datetime, timezone
from uuid import uuid4


def build_rag_log(
    user_id: str,
    raw_question: str,
    input_parameters: dict,
    effective_parameters: dict,
    retrieval_chunks: list,
    rerank_chunks: list,
    prompt_final: str,
    llm_metadata: dict,
    raw_answer: str,
    clean_answer: str,
    metrics: dict,
    component_versions: dict,
    pipeline_version: str,
    latencies: dict | None = None,
) -> dict:
    """Build a JSON payload describing a full RAG pipeline audit.

    Args:
        user_id: Unique identifier of the originating user.
        raw_question: User prompt before any normalization.
        input_parameters: Parameters requested by the caller.
        effective_parameters: Parameters actually used by the pipeline.
        retrieval_chunks: Retriever outputs, scores, and context snapshots.
        rerank_chunks: Reranker outputs with final selection info.
        prompt_final: Prompt sent to the LLM (already serialized if needed).
        llm_metadata: Token usage, latency, and other LLM metadata.
        raw_answer: Raw string response returned by the LLM.
        clean_answer: Sanitized answer exposed to the UI.
        metrics: Offline metrics such as faithfulness or relevancy.
        component_versions: Mapping of component names to versions.
        pipeline_version: Manual pipeline version tag.
        latencies: Optional latencies in milliseconds per stage.

    Returns:
        dict: JSON-serializable structure ready for persistence.
    """

    request_id = str(uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    payload = {
        "metadata": {
            "request_id": request_id,
            "timestamp": timestamp,
            "user_id": user_id,
            "pipeline_version": pipeline_version,
            "latencies_ms": latencies or {},
            "components": {
                "llm_version": component_versions.get("llm_version"),
                "embedder_version": component_versions.get("embedder_version"),
                "retriever_version": component_versions.get(
                    "retriever_version"
                ),
                "reranker_version": component_versions.get("reranker_version"),
                "chunker_version": component_versions.get("chunker_version"),
            },
        },
        "user_input": {
            "raw_question": raw_question,
            "input_parameters": input_parameters,
        },
        "effective_parameters": effective_parameters,
        "retrieval": {
            "top_k_returned": len(retrieval_chunks),
            "chunks": retrieval_chunks,
        },
        "rerank": {
            "chunks_after_rerank": rerank_chunks,
        },
        "llm_generation": {
            "prompt_final": prompt_final,
            "llm_metadata": llm_metadata,
            "raw_answer": raw_answer,
            "clean_answer": clean_answer,
        },
        "rag_metrics": metrics,
    }

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    sha256 = hashlib.sha256(serialized).hexdigest()
    payload["integrity"] = {"sha256": sha256}

    return payload
