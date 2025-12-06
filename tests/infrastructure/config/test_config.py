"""Tests for configuration module."""

import importlib

from src.rag.infrastructure.config import config as config_module


#############################################
# Unit tests: AppConfig wiring and defaults #
#############################################
def test_appconfig_load_provides_completion_defaults():
    """Ensure AppConfig.load returns the expected completion args."""
    cfg = config_module.AppConfig.load()

    assert cfg.completion_args == {
        "temperature": 0.2,
        "max_tokens": 300,
        "top_p": 0.22,
    }
    assert cfg.system_prompt.startswith("### RÔLE")


def test_appconfig_uses_environment_defaults(monkeypatch):
    """Ensure dataclass defaults read from environment variables."""
    monkeypatch.setenv("MISTRAL_CHAT_MODEL", "custom-chat")
    monkeypatch.setenv("MISTRAL_EMBED_MODEL", "custom-embed")
    monkeypatch.setenv("FAISS_INDEX_PATH", "custom/faiss.idx")
    monkeypatch.setenv("CHUNKS_PATH", "custom/chunks.pkl")
    monkeypatch.setenv("SOURCES_PATH", "custom/sources.json")

    reloaded = importlib.reload(config_module)
    cfg = reloaded.AppConfig()

    assert cfg.chat_model == "custom-chat"
    assert cfg.embed_model == "custom-embed"
    assert cfg.faiss_index_path == "custom/faiss.idx"
    assert cfg.chunks_path == "custom/chunks.pkl"
    assert cfg.sources_path == "custom/sources.json"
    assert reloaded.os.environ["KMP_DUPLICATE_LIB_OK"] == "True"
