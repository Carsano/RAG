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
    Indexe récursivement des fichiers Markdown dans un index FAISS
    à partir d'embeddings Mistral.

    - Scan récursif d'un dossier racine
    - Découpage Markdown (chunk_size, chunk_overlap)
    - Embedding via API Mistral avec un retry unique
    - Index FAISS (L2) + sauvegardes intermédiaires pour reprise
    - Export: index.faiss + all_chunks.(pkl|json)
    - Fichiers temporaires: tmp_chunks.pkl + tmp_embeddings.npy
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
        load_dotenv()
        self.root = pathlib.Path(root)
        self.model = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.sleep_between_calls = sleep_between_calls

        self.tmp_chunks_path = tmp_chunks_path or pathlib.Path(
            "tmp_chunks.pkl"
        )
        self.tmp_embeddings_path = tmp_embeddings_path or pathlib.Path(
            "tmp_embeddings.npy"
        )
        self.index_path = index_path or pathlib.Path(
            "faiss_index.idx"
        )
        self.chunks_json_path = chunks_json_path or pathlib.Path(
            "all_chunks.json"
        )
        self.chunks_pkl_path = chunks_pkl_path or pathlib.Path(
            "all_chunks.pkl"
        )

        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY manquant dans l'environnement")
        self.client = Mistral(api_key=api_key)

        # Etat en mémoire
        self.valid_chunks: List[str] = []
        self.embeddings_list: List[List[float]] = []
        self.index: Optional[faiss.Index] = None
        self.logger = Logger("indexer")

    def build(self) -> None:
        files = self._list_md_files(self.root)
        all_chunks = self._chunk_markdown_files(files)

        # Reprise si tmp présents
        if self.tmp_chunks_path.exists():
            with open(self.tmp_chunks_path, "rb") as f:
                self.valid_chunks = pickle.load(f)
        if self.tmp_embeddings_path.exists():
            self.embeddings_list = np.load(self.tmp_embeddings_path).tolist()

        start_idx = len(self.valid_chunks)
        for i, chunk in enumerate(all_chunks[start_idx:], start=start_idx):
            self.logger.info(f"[{i}] Embedding en cours…")
            emb = self._embed_text(chunk)
            self.logger.info(f"[{i}] Embedding status: {emb is not None}")
            if emb is None:
                time.sleep(self.sleep_between_calls)
                continue

            self.embeddings_list.append(emb)
            self.valid_chunks.append(chunk)

            # Sauvegardes intermédiaires
            with open(self.tmp_chunks_path, "wb") as f:
                pickle.dump(self.valid_chunks, f)
            np.save(
                self.tmp_embeddings_path,
                np.array(self.embeddings_list, dtype="float32"),
            )
            self.logger.info(
                f"[{i}] Sauvegarde faite : {len(self.valid_chunks)} chunks, "
                f"{len(self.embeddings_list)} embeddings"
            )
            time.sleep(self.sleep_between_calls)

        # Finalisation FAISS
        if not self.embeddings_list:
            self.logger.warning("Aucun embedding généré. Rien à indexer.")
            return

        embeddings = np.array(self.embeddings_list, dtype="float32")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        # Exports finaux
        with open(self.chunks_pkl_path, "wb") as f:
            pickle.dump(self.valid_chunks, f)
        with open(self.chunks_json_path, "w", encoding="utf-8") as f:
            json.dump(self.valid_chunks, f, ensure_ascii=False, indent=2)
        faiss.write_index(self.index, str(self.index_path))
        self.logger.info(
            f"Index écrit: {self.index_path} | "
            f"Chunks: {self.chunks_json_path} / {self.chunks_pkl_path}"
        )

    def _list_md_files(self, root: pathlib.Path) -> List[pathlib.Path]:
        return [p for p in root.rglob("*.md") if p.is_file()]

    def _chunk_markdown_files(self, files: List[pathlib.Path]) -> List[str]:
        splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        chunks: List[str] = []
        for path in files:
            text = path.read_text(encoding="utf-8", errors="ignore")
            chunks.extend(splitter.split_text(text))
        return chunks

    def _embed_text(self, text: str) -> Optional[List[float]]:
        try:
            batch = self.client.embeddings.create(model=self.model,
                                                  inputs=text)
            return batch.data[0].embedding
        except Exception as e:
            self.logger.error(f"API error: {e}. Waiting 60s before retry…")
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
    ROOT = pathlib.Path(__file__).parent / "clean_md_database"
    indexer = Indexer(root=ROOT)
    indexer.build()


if __name__ == "__main__":
    main()
