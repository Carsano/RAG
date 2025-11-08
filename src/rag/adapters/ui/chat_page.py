"""
Chat page UI for Streamlit.
Single responsibility: present messages and delegate to RAGChatService.
No business logic here.
"""
from __future__ import annotations

from typing import TypedDict, Literal
from src.rag.infrastructure.config.types import Messages
import streamlit as st
from src.rag.application.use_cases.rag_chat import RAGChatService
from src.rag.infrastructure.logging.logger import get_usage_logger

Role = Literal["system", "user", "assistant"]


class ChatMessage(TypedDict):
    role: Role
    content: str


WELCOME: str = (
    "Bonjour, je suis l'assistant virtuel de la mairie. "
    "Comment puis-je vous aider aujourd'hui ?"
)


class ChatPage:
    def __init__(self, service: RAGChatService, title: str) -> None:
        self.service = service
        self.title = title
        self.usage_logger = get_usage_logger()

    # ---------- state ----------
    def _ensure_state(self) -> None:
        if "messages" not in st.session_state:
            messages: Messages = [
                {"role": "assistant", "content": WELCOME}
            ]
            st.session_state.messages = messages
        # avoid calling set_page_config twice across reruns
        if "_page_init" not in st.session_state:
            st.set_page_config(page_title=self.title, page_icon="🏛️")
            st.session_state["_page_init"] = True

    # ---------- rendering ----------
    def _render_history(self) -> None:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    def _append(self, role: Role, content: str) -> None:
        st.session_state.messages.append({"role": role, "content": content})

    def _handle_user_prompt(self, prompt: str) -> None:
        # echo user
        self._append("user", prompt)
        with st.chat_message("user"):
            st.write(prompt)

        # assistant placeholder
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("_En cours..._")
            try:
                reply = self.service.answer(
                    history=st.session_state.messages, question=prompt
                )
                # log successful interaction
                self.usage_logger.info(f"User: {prompt} | Response: {reply}")
            except Exception as e:
                reply = f"Erreur lors de la génération de la réponse : {e}"
                # log error case
                self.usage_logger.error(f"User: {prompt} | Error: {e}")
            placeholder.write(reply)

        self._append("assistant", reply)

    # ---------- public ----------
    def render(self) -> None:
        self._ensure_state()
        st.title(self.title)

        self._render_history()

        prompt = st.chat_input("Votre question")
        if prompt:
            self._handle_user_prompt(prompt)
