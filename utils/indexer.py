"""
Simple FAISS indexer for Markdown files using Mistral embeddings.

This module scans a root directory, chunks Markdown, generates embeddings,
and builds a FAISS index. It logs progress and writes artifacts for later
retrieval.
"""
from __future__ import annotations

import time
import json
import pickle
import pathlib
from typing import List, Optional

import numpy as np
import faiss
from langchain_text_splitters import MarkdownTextSplitter
from utils.embedders import Embedder, MistralEmbedder
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
        embedder (Embedder): Embedding backend.
        valid_chunks (List[str]): Chunks that were embedded.
        embeddings_list (List[List[float]]): Collected embeddings.
        valid_sources (List[str]): Source file path per valid chunk.
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
        embedder: Embedder | None = None,
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
            embedder (Embedder | None): Embedding backend instance.

        Raises:
            RuntimeError: If MISTRAL_API_KEY is missing.
        """
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

        # Embedding backend
        self.embedder: Embedder = (
            embedder or MistralEmbedder(model=self.model,
                                        delay=self.sleep_between_calls)
        )

        self.valid_chunks: List[str] = []
        self.embeddings_list: List[List[float]] = []
        self.valid_sources: List[str] = []
        self.index: Optional[faiss.Index] = None
        self.logger = Logger("indexer")

    def _load_resume_state(self) -> None:
        """
        Load prior progress from temporary files if they exist.
        """
        if self.tmp_chunks_path.exists():
            with open(self.tmp_chunks_path, "rb") as f:
                self.valid_chunks = pickle.load(f)
            tmp_sources = self.tmp_chunks_path.with_name("tmp_sources.pkl")
            if tmp_sources.exists():
                with open(tmp_sources, "rb") as f:
                    self.valid_sources = pickle.load(f)
        if self.tmp_embeddings_path.exists():
            self.embeddings_list = (
                np.load(self.tmp_embeddings_path).tolist()
            )

    def _embed_and_save(self, all_chunks: List[str],
                        all_sources: List[str]) -> None:
        """
        Embed remaining chunks and persist intermediate progress.

        Args:
            all_chunks (List[str]): Full list of chunks to process.
            all_sources (List[str]): Corresponding source file paths.
        """
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
            self.valid_sources.append(all_sources[i])
            self._save_intermediate(i)
            time.sleep(self.sleep_between_calls)

    def _save_intermediate(self, i: int) -> None:
        """
        Persist intermediate chunk and embedding state to disk.

        Args:
            i (int): Index of the last processed chunk.
        """
        with open(self.tmp_chunks_path, "wb") as f:
            pickle.dump(self.valid_chunks, f)
        tmp_sources = self.tmp_chunks_path.with_name("tmp_sources.pkl")
        with open(tmp_sources, "wb") as f:
            pickle.dump(self.valid_sources, f)
        np.save(
            self.tmp_embeddings_path,
            np.array(self.embeddings_list, dtype="float32"),
        )
        self.logger.info(
            f"[{i}] Save done: {len(self.valid_chunks)} chunks, "
            f"{len(self.embeddings_list)} embeddings"
        )

    def _finalize_index(self) -> None:
        """
        Build FAISS index and write final artifacts to disk.
        """
        if not self.embeddings_list:
            self.logger.warning(
                "No embeddings generated. Nothing to index."
            )
            return

        embeddings = np.array(self.embeddings_list, dtype="float32")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        with open(self.chunks_pkl_path, "wb") as f:
            pickle.dump(self.valid_chunks, f)
        with open(self.chunks_json_path, "w", encoding="utf-8") as f:
            json.dump(self.valid_chunks, f, ensure_ascii=False, indent=2)
        sources_json = self.chunks_json_path.with_name(
            "all_chunk_sources.json")
        with open(sources_json, "w", encoding="utf-8") as f:
            json.dump(self.valid_sources, f, ensure_ascii=False, indent=2)
        faiss.write_index(self.index, str(self.index_path))
        self.logger.info(
            f"Index written: {self.index_path} | "
            f"Chunks: {self.chunks_json_path} / {self.chunks_pkl_path}"
        )

    def _filter_by_file(self, file_path: pathlib.Path) -> int:
        """Filter in-memory items that originate from a given file.

        Args:
            file_path: Absolute or relative path of the file to remove.

        Returns:
            int: Number of removed chunks.
        """
        target = str(pathlib.Path(file_path).resolve())

        # Normalize sources to absolute paths to compare reliably.
        abs_sources = [
            str(pathlib.Path(s).resolve()) for s in self.valid_sources
        ]
        keep_mask = [s != target for s in abs_sources]

        removed = len(keep_mask) - sum(keep_mask)
        if removed == 0:
            self.logger.info("No chunks to remove for this file.")
            return 0

        # Filter in-memory parallel lists using the same boolean mask.
        self.valid_chunks = [
            c for c, k in zip(self.valid_chunks, keep_mask) if k
        ]
        self.valid_sources = [
            s for s, k in zip(self.valid_sources, keep_mask) if k
        ]
        self.embeddings_list = [
            e for e, k in zip(self.embeddings_list, keep_mask) if k
        ]
        return removed

    def _rebuild_index_from_memory(self) -> None:
        """Rebuild the FAISS index from the in-memory embeddings.

        Returns:
            None: The method updates `self.index` in place.
        """
        if not self.embeddings_list:
            self.index = None
            return

        embs = np.array(self.embeddings_list, dtype="float32")
        dim = embs.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embs)

    def _list_md_files(self, root: pathlib.Path) -> List[pathlib.Path]:
        """
        List Markdown files under the given root.

        Args:
            root (pathlib.Path): Root directory to scan.

        Returns:
            List[pathlib.Path]: Sorted list of Markdown file paths.
        """
        return [p for p in root.rglob("*.md") if p.is_file()]

    def _chunk_markdown_files(self, files: List[pathlib.Path]
                              ) -> (List[str] | List[str]):
        """
        Split Markdown files into chunks.

        Args:
            files (List[pathlib.Path]): Markdown file paths.

        Returns:
            Tuple[List[str], List[str]]: (chunks, sources) parallel lists.
        """
        splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        chunks: List[str] = []
        sources: List[str] = []
        for path in files:
            text = path.read_text(encoding="utf-8", errors="ignore")
            parts = splitter.split_text(text)
            chunks.extend(parts)
            sources.extend([str(path)] * len(parts))
        return chunks, sources

    def _embed_text(self, text: str) -> Optional[List[float]]:
        """
        Create an embedding for a single text.

        Args:
            text (str): The text to embed.

        Returns:
            Optional[List[float]]: Embedding vector or None on failure.
        """
        return self.embedder.embed(text)

    def build(self) -> None:
        """
        Orchestrate indexing from Markdown files to a FAISS index.

        Performs scan, chunking, resume of prior state, embedding, and
        finalization. Side effects are delegated to dedicated helpers.
        """
        files = self._list_md_files(self.root)
        all_chunks, all_sources = self._chunk_markdown_files(files)
        self._load_resume_state()
        # Align valid_sources length with valid_chunks if needed
        if len(self.valid_sources) != len(self.valid_chunks):
            self.valid_sources = self.valid_sources[:len(self.valid_chunks)]
        self._embed_and_save(all_chunks, all_sources)
        self._finalize_index()

    def remove_file(self, file_path: pathlib.Path) -> int:
        """
        Remove all chunks originating from a given file, then rebuild index.

        Args:
            file_path (pathlib.Path): Absolute or relative path to remove.

        Returns:
            int: Number of removed chunks.
        """
        removed = self._filter_by_file(file_path)
        if removed == 0:
            return 0

        self._rebuild_index_from_memory()
        self._save_intermediate(i=max(0, len(self.valid_chunks) - 1))
        self._finalize_index()
        return removed


def main() -> None:
    """
    Entry point for a manual indexing run.

    Uses the `clean_md_database` folder as root and builds the index.
    """
    ROOT = (pathlib.Path(__file__).resolve().parent.parent / "data" /
            "clean_md_database")
    indexer = Indexer(root=ROOT)
    indexer.build()
    removed = indexer.remove_file(
        pathlib.Path("data/clean_md_database/budget/budget_2024.md"))
    indexer.logger.info(f"Removed {removed}")


if __name__ == "__main__":
    main()
