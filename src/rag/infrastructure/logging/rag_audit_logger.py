"""DuckDB-backed logger for persisting full RAG audit payloads."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import duckdb


class RAGAuditLogger:
    """Persist full audit payloads into DuckDB and a JSONL snapshot."""

    def __init__(
        self,
        *,
        db_path: str = "data/db/rag.duckdb",
        jsonl_path: str = "logs/rag_audit/rag_audit.jsonl",
    ) -> None:
        self.db_path = db_path
        self.jsonl_path = jsonl_path
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

    def persist(self, payload: Dict[str, Any]) -> None:
        """Persist the payload to disk and DuckDB.

        Args:
            payload: Structured dict returned by build_rag_log.
        """
        if not payload:
            return

        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
        )
        self._append_jsonl(serialized)
        self._persist_duckdb(payload, serialized)

    def _append_jsonl(self, serialized: str) -> None:
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as fp:
                fp.write(serialized + "\n")
        except Exception:
            # File logging should never break the chat flow.
            pass

    def _persist_duckdb(self, payload: Dict[str, Any],
                        serialized: str) -> None:
        try:
            con = duckdb.connect(self.db_path)
        except Exception:
            return

        try:
            metadata = payload.get("metadata", {})
            components = metadata.get("components", {})
            latencies = metadata.get("latencies_ms", {}) or {}
            request_id = metadata.get("request_id")
            timestamp = metadata.get("timestamp")
            user_id = metadata.get("user_id")
            llm_meta = (
                payload.get("llm_generation", {}).get("llm_metadata", {}) or {}
            )
            integrity_hash = payload.get("integrity", {}).get("sha256")

            self._upsert_user(con, user_id)
            self._insert_audit_json(
                con, request_id, timestamp, serialized, integrity_hash
            )
            self._insert_request_row(
                con,
                request_id,
                timestamp,
                user_id,
                components,
                latencies,
                llm_meta,
            )
            self._insert_metrics(
                con,
                request_id,
                payload.get("rag_metrics", {}) or {},
            )
            self._insert_chunks(
                con,
                request_id,
                payload.get("rerank", {}).get("chunks_after_rerank", []) or [],
            )
        except Exception:
            pass
        finally:
            con.close()

    @staticmethod
    def _upsert_user(con: duckdb.DuckDBPyConnection,
                     user_id: str | None) -> None:
        if not user_id:
            return
        con.execute(
            """
            INSERT INTO users (user_id)
            SELECT ?
            WHERE NOT EXISTS (
                SELECT 1 FROM users WHERE user_id = ?
            )
            """,
            [user_id, user_id],
        )

    @staticmethod
    def _insert_audit_json(
        con: duckdb.DuckDBPyConnection,
        request_id: str | None,
        timestamp: str | None,
        serialized: str,
        integrity_hash: str | None,
    ) -> None:
        con.execute(
            """
            INSERT INTO audit_json (request_id, timestamp, json_record,
                                    integrity_hash)
            VALUES (?, ?, ?, ?)
            """,
            [request_id, timestamp, serialized, integrity_hash],
        )

    @staticmethod
    def _insert_request_row(
        con: duckdb.DuckDBPyConnection,
        request_id: str | None,
        timestamp: str | None,
        user_id: str | None,
        components: Dict[str, Any],
        latencies: Dict[str, Any],
        llm_meta: Dict[str, Any],
    ) -> None:
        con.execute(
            """
            INSERT INTO request (
                request_id,
                timestamp,
                user_id,
                llm_version,
                embedder_version,
                reranker_version,
                retriever_version,
                chunker_version,
                latency_retrieval_ms,
                latency_rerank_ms,
                latency_generation_ms,
                total_latency_ms,
                input_tokens,
                output_tokens,
                total_tokens
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                request_id,
                timestamp,
                user_id,
                components.get("llm_version"),
                components.get("embedder_version"),
                components.get("reranker_version"),
                components.get("retriever_version"),
                components.get("chunker_version"),
                latencies.get("retrieval_ms"),
                latencies.get("rerank_ms"),
                latencies.get("generation_ms"),
                latencies.get("total_ms"),
                llm_meta.get("input_tokens"),
                llm_meta.get("output_tokens"),
                llm_meta.get("total_tokens"),
            ],
        )

    @staticmethod
    def _insert_metrics(
        con: duckdb.DuckDBPyConnection,
        request_id: str | None,
        metrics: Dict[str, Any],
    ) -> None:
        con.execute(
            """
            INSERT INTO rag_metrics (
                request_id,
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                request_id,
                metrics.get("faithfulness"),
                metrics.get("answer_relevancy"),
                metrics.get("context_precision"),
                metrics.get("context_recall"),
            ],
        )

    @staticmethod
    def _insert_chunks(
        con: duckdb.DuckDBPyConnection,
        request_id: str | None,
        rerank_chunks: List[Dict[str, Any]],
    ) -> None:
        if not rerank_chunks:
            return
        con.executemany(
            """
            INSERT INTO chunks_used (
                id,
                request_id,
                chunk_snapshot_text,
                chunk_source_path,
                retriever_score,
                reranker_score,
                was_selected
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    None,
                    request_id,
                    chunk.get("content_snapshot"),
                    chunk.get("source_path"),
                    chunk.get("score_retrieval"),
                    chunk.get("score_reranker"),
                    chunk.get("was_selected"),
                )
                for chunk in rerank_chunks
            ],
        )


__all__ = ["RAGAuditLogger"]
