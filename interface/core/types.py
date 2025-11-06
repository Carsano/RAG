"""
types used across the core application.
"""
from dataclasses import dataclass
from typing import Literal, List, Dict, Any

Role = Literal["system", "user", "assistant"]
LLMMessage = Dict[str, Any]


@dataclass
class ChatMessage:
    role: Role
    content: str


Messages = List[ChatMessage]
