"""Tests for configuration module."""

from src.rag.infrastructure.config import config as config_module


############################################################
# Unit tests: AppConfig wiring and defaults
############################################################
def test_appconfig_load_provides_completion_defaults():
    """Ensure AppConfig.load returns the expected completion args."""
    cfg = config_module.AppConfig.load()

    assert cfg.completion_args == {
        "temperature": 0.2,
        "max_tokens": 300,
        "top_p": 0.22,
    }
    assert cfg.system_prompt.startswith("### RÔLE")
