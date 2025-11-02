"""
Simple FAISS indexer for Markdown files using Mistral embeddings.

This module scans a root directory, chunks Markdown, generates embeddings,
and builds a FAISS index. It logs progress and writes artifacts for later
retrieval.
"""
from __future__ import annotations

import os
import time
import json
import pickle
import pathlib
from typing import List, Optional

import numpy as np
import faiss
from dotenv import load_dotenv
from mistralai import Mistral
from langchain_text_splitters import MarkdownTextSplitter
from utils.logger import Logger


class Indexer:
    """
    Index Markdown files into a FAISS index using Mistral embeddings.

    The class performs a recursive scan, Markdown chunking, embedding calls,
    and FAISS indexing. It persists intermediate state to allow resuming
    long runs, and exports the final index and chunk lists.

    Attributes:
        root (pathlib.Path): Root directory that contains Markdown files.
        model (str): Mistral embedding model name.
        chunk_size (int): Max characters per chunk before overlap.
        chunk_overlap (int): Characters of overlap between chunks.
        sleep_between_calls (float): Delay between API calls in seconds.
        tmp_chunks_path (pathlib.Path): Path for tmp chunks pickle.
        tmp_embeddings_path (pathlib.Path): Path for tmp embeddings npy.
        index_path (pathlib.Path): Output FAISS index path.
        chunks_json_path (pathlib.Path): Output JSON chunks path.
        chunks_pkl_path (pathlib.Path): Output pickle chunks path.
        client (Mistral): Mistral client instance.
        valid_chunks (List[str]): Chunks that were embedded.
        embeddings_list (List[List[float]]): Collected embeddings.
        index (Optional[faiss.Index]): FAISS index instance.
        logger (Logger): Application logger.
    """

    def __init__(
        self,
        root: pathlib.Path,
        model: str = "mistral-embed",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        tmp_chunks_path: pathlib.Path | None = None,
        tmp_embeddings_path: pathlib.Path | None = None,
        index_path: pathlib.Path | None = None,
        chunks_json_path: pathlib.Path | None = None,
        chunks_pkl_path: pathlib.Path | None = None,
        sleep_between_calls: float = 10.0,
    ) -> None:
        """
        Initialize the indexer.

        Args:
            root (pathlib.Path): Root directory to scan.
            model (str): Embedding model name for Mistral.
            chunk_size (int): Chunk size in characters.
            chunk_overlap (int): Overlap size in characters.
            tmp_chunks_path (pathlib.Path | None): Tmp pickle path for chunks.
            tmp_embeddings_path (pathlib.Path | None): Tmp numpy path for
                embeddings.
            index_path (pathlib.Path | None): Output FAISS index path.
            chunks_json_path (pathlib.Path | None): Output JSON path for
                chunks.
            chunks_pkl_path (pathlib.Path | None): Output pickle path for
                chunks.
            sleep_between_calls (float): Delay between API calls in seconds.

        Raises:
            RuntimeError: If MISTRAL_API_KEY is missing.
        """
        load_dotenv()
        self.root = pathlib.Path(root)
        self.model = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.sleep_between_calls = sleep_between_calls

        # Resolve project root and ensure data/indexes exists
        project_root = pathlib.Path(__file__).resolve().parent.parent
        indexes_dir = project_root / "data" / "indexes"
        indexes_dir.mkdir(parents=True, exist_ok=True)

        # Default artifact locations under data/indexes/
        self.tmp_chunks_path = tmp_chunks_path or (
            indexes_dir / "tmp_chunks.pkl"
        )
        self.tmp_embeddings_path = tmp_embeddings_path or (
            indexes_dir / "tmp_embeddings.npy"
        )
        self.index_path = index_path or (
            indexes_dir / "faiss_index.idx"
        )
        self.chunks_json_path = chunks_json_path or (
            indexes_dir / "all_chunks.json"
        )
        self.chunks_pkl_path = chunks_pkl_path or (
            indexes_dir / "all_chunks.pkl"
        )

        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY missing from environment")
        self.client = Mistral(api_key=api_key)

        # In-memory state
        self.valid_chunks: List[str] = []
        self.embeddings_list: List[List[float]] = []
        self.index: Optional[faiss.Index] = None
        self.logger = Logger("indexer")

    def build(self) -> None:
        """
        Build the FAISS index from Markdown files.

        Scans files, chunks text, creates embeddings with Mistral, and builds a
        FAISS L2 index. Intermediate progress is saved to allow resuming.

        Side Effects:
            Writes temporary chunk and embedding files. Writes final index and
            chunk lists on success.
        """
        files = self._list_md_files(self.root)
        all_chunks = self._chunk_markdown_files(files)

        if self.tmp_chunks_path.exists():
            with open(self.tmp_chunks_path, "rb") as f:
                self.valid_chunks = pickle.load(f)
        if self.tmp_embeddings_path.exists():
            self.embeddings_list = np.load(self.tmp_embeddings_path).tolist()

        start_idx = len(self.valid_chunks)
        for i, chunk in enumerate(all_chunks[start_idx:], start=start_idx):
            self.logger.info(f"[{i}] Embedding in progress...")
            emb = self._embed_text(chunk)
            self.logger.info(f"[{i}] Embedding status: {emb is not None}")
            if emb is None:
                time.sleep(self.sleep_between_calls)
                continue

            self.embeddings_list.append(emb)
            self.valid_chunks.append(chunk)

            # Intermediate saves
            with open(self.tmp_chunks_path, "wb") as f:
                pickle.dump(self.valid_chunks, f)
            np.save(
                self.tmp_embeddings_path,
                np.array(self.embeddings_list, dtype="float32"),
            )
            self.logger.info(
                f"[{i}] Save done: {len(self.valid_chunks)} chunks, "
                f"{len(self.embeddings_list)} embeddings"
            )
            time.sleep(self.sleep_between_calls)

        # FAISS finalization
        if not self.embeddings_list:
            self.logger.warning("No embeddings generated. Nothing to index.")
            return

        embeddings = np.array(self.embeddings_list, dtype="float32")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        # Final exports
        with open(self.chunks_pkl_path, "wb") as f:
            pickle.dump(self.valid_chunks, f)
        with open(self.chunks_json_path, "w", encoding="utf-8") as f:
            json.dump(self.valid_chunks, f, ensure_ascii=False, indent=2)
        faiss.write_index(self.index, str(self.index_path))
        self.logger.info(
            f"Index written: {self.index_path} | "
            f"Chunks: {self.chunks_json_path} / {self.chunks_pkl_path}"
        )

    def _list_md_files(self, root: pathlib.Path) -> List[pathlib.Path]:
        """
        List Markdown files under the given root.

        Args:
            root (pathlib.Path): Root directory to scan.

        Returns:
            List[pathlib.Path]: Sorted list of Markdown file paths.
        """
        return [p for p in root.rglob("*.md") if p.is_file()]

    def _chunk_markdown_files(self, files: List[pathlib.Path]) -> List[str]:
        """
        Split Markdown files into chunks.

        Args:
            files (List[pathlib.Path]): Markdown file paths.

        Returns:
            List[str]: Text chunks produced by the splitter.
        """
        splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        chunks: List[str] = []
        for path in files:
            text = path.read_text(encoding="utf-8", errors="ignore")
            chunks.extend(splitter.split_text(text))
        return chunks

    def _embed_text(self, text: str) -> Optional[List[float]]:
        """
        Create an embedding for a single text.

        Args:
            text (str): The text to embed.

        Returns:
            Optional[List[float]]: Embedding vector or None on failure.
        """
        try:
            batch = self.client.embeddings.create(model=self.model,
                                                  inputs=text)
            return batch.data[0].embedding
        except Exception as e:
            self.logger.error(f"API error: {e}. Waiting 60s before retry...")
            time.sleep(60)
            try:
                batch = self.client.embeddings.create(model=self.model,
                                                      inputs=text)
                return batch.data[0].embedding
            except Exception as e2:
                self.logger.error(f"Retry failed for this text."
                                  f"Skipping. Error: {e2}")
                return None


def main() -> None:
    """
    Entry point for a manual indexing run.

    Uses the `clean_md_database` folder as root and builds the index.
    """
    ROOT = (pathlib.Path(__file__).resolve().parent.parent / "data" /
            "clean_md_database")
    indexer = Indexer(root=ROOT)
    indexer.build()


if __name__ == "__main__":
    main()
