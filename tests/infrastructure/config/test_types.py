"""Unit tests for ``src.rag.infrastructure.config.types``."""

from dataclasses import is_dataclass
from typing import get_args, get_type_hints, get_origin

from src.rag.infrastructure.config import types as config_types


def test_role_literal_contains_expected_values():
    """Role literal should only allow the supported participants."""
    literal_values = set(get_args(config_types.Role))
    assert literal_values == {"system", "user", "assistant"}


def test_chatmessage_is_dataclass_with_expected_annotations():
    """ChatMessage keeps strict typing for role/content."""
    assert is_dataclass(config_types.ChatMessage)
    annotations = get_type_hints(config_types.ChatMessage)
    assert annotations["role"] is config_types.Role
    assert annotations["content"] is str

    msg = config_types.ChatMessage(role="assistant", content="Réponse.")
    assert msg.role == "assistant"
    assert msg.content == "Réponse."


def test_messages_alias_targets_list_of_chatmessage():
    """Messages alias simplifies passing ordered chat history."""
    assert get_origin(config_types.Messages) is list
    assert get_args(config_types.Messages) == (config_types.ChatMessage,)
