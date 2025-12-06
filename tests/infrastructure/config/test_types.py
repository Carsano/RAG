"""Unit tests for ``src.rag.infrastructure.config.types``."""


from typing import get_args

from src.rag.infrastructure.config import types as config_types


def test_role_literal_contains_expected_values():
    """Role literal should only allow the supported participants."""
    literal_values = set(get_args(config_types.Role))
    assert literal_values == {"system", "user", "assistant"}


