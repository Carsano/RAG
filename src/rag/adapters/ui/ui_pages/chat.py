"""Streamlit chat page wiring for the RAG assistant.

Instantiates the chat service and UI components, binds them to Streamlit's
page lifecycle, and exposes the main chat entry point.
"""
import streamlit as st

from services.rag_chat import get_chat_service

from components.message import render_message
from components.feedback import render_feedback
from components.sources import render_sources

service = get_chat_service()

WELCOME = (
    "Bonjour, je suis l'assistant virtuel de la mairie. "
    "Comment puis-je vous aider aujourd'hui ?"
)


def ensure_chat_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": WELCOME}
        ]


def render():
    ensure_chat_state()
    st.title("Assistant municipal")

    # Render historique
    for idx, msg in enumerate(st.session_state.messages):
        render_message(msg)
        if msg["role"] == "assistant" and msg["content"] != WELCOME:
            render_feedback(idx, msg)

    # Input utilisateur
    prompt = st.chat_input("Votre question")
    if prompt:
        _handle_user_message(prompt)


def _handle_user_message(prompt: str):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_En cours..._")

        settings = st.session_state.get("settings", {})
        temperature = float(settings.get("temperature", 0.3))
        max_tokens = int(settings.get("max_answer_tokens", 512))
        max_sources = int(settings.get("max_sources", 5))

        if hasattr(service, "llm") and hasattr(service.llm, "args"):
            service.llm.args["temperature"] = temperature
            service.llm.args["max_tokens"] = max_tokens

        reply = service.answer(
            history=st.session_state.messages,
            question=prompt,
            max_sources=max_sources,
        )

        answer = reply["answer"]
        placeholder.write(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        show_sources = bool(settings.get("show_sources", True))
        if show_sources and reply.get("sources"):
            render_sources(reply["sources"])
