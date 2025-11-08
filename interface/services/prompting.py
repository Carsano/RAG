"""
prompting services interface.
Contains methods for constructing and managing prompts.
"""
from typing import List


def build_system_prompt(base_prompt: str, context_chunks: List[str]) -> str:
    if not context_chunks:
        return base_prompt
    ctx = "\n\n".join(context_chunks)
    extra = (
        "Veuillez utiliser les informations suivantes pour"
        "répondre à la question :\n\n"
        f"{ctx}\n\n"
        "Si les informations ne suffisent pas, dites-le."
    )
    return base_prompt + "\n\n" + extra


def clamp_dialog(messages: List[dict], max_messages: int = 5) -> List[dict]:
    convo = [m for m in messages if m["role"] in ("user", "assistant")]
    recent = convo[-max_messages:] if len(convo) > max_messages else convo
    while recent and recent[0]["role"] != "user":
        recent.pop(0)
    return recent
