"""Streamlit chat page wiring for the RAG assistant.

Instantiates the chat service and UI components, binds them to Streamlit's
page lifecycle, and exposes the main chat entry point.
"""
import streamlit as st

from src.rag.application.use_cases.rag_chat import RAGChatService
from components.message import render_message
from components.feedback import render_feedback
from components.sources import render_sources

service = RAGChatService()

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

        reply = service.answer(
            history=st.session_state.messages,
            question=prompt,
        )

        answer = reply["answer"]
        placeholder.write(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        if reply.get("sources"):
            render_sources(reply["sources"])
