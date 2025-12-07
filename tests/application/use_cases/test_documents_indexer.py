"""
Tests for DocumentsIndexer selective indexing helpers.
"""

from typing import List

from src.rag.application.use_cases.documents_indexer import DocumentsIndexer


class StubChunker:
    """Chunker that returns the entire document as a single chunk."""

    version = "stub-chunker"

    def __init__(self, *args, **kwargs):
        pass

    def split(self, text: str) -> List[str]:
        return [text]


class StubEmbedder:
    """Embedder that returns deterministic vectors."""

    def __init__(self):
        self.calls: list[str] = []

    def embed(self, text: str) -> List[float]:
        self.calls.append(text)
        length = float(len(text))
        return [length, length]


def test_index_files_appends_selected_markdown(tmp_path, monkeypatch) -> None:
    """index_files() should chunk, embed, and persist the selected files."""
    markdown_root = tmp_path / "clean_md_database"
    markdown_root.mkdir()
    file_a = markdown_root / "alpha.md"
    file_a.write_text("Alpha content", encoding="utf-8")

    indexes_dir = tmp_path / "indexes"
    indexes_dir.mkdir()
    embedder = StubEmbedder()
    indexer = DocumentsIndexer(
        root=markdown_root,
        chunker=StubChunker(),
        embedder=embedder,
        tmp_chunks_path=indexes_dir / "tmp_chunks.pkl",
        tmp_embeddings_path=indexes_dir / "tmp_embeddings.npy",
        index_path=indexes_dir / "faiss_index.idx",
        chunks_json_path=indexes_dir / "all_chunks.json",
        chunks_pkl_path=indexes_dir / "all_chunks.pkl",
    )

    indexer.index_files([file_a])

    assert str(file_a) in indexer.valid_sources
    assert embedder.calls == ["Alpha content"]
    assert (indexes_dir / "faiss_index.idx").exists()
    assert (indexes_dir / "all_chunk_sources.json").exists()
