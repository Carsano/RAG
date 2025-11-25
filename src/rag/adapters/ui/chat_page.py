"""Chat page UI for Streamlit.

Presents chat messages and delegates processing to RAGChatService.
Contains no business logic.
"""
from __future__ import annotations

from typing import TypedDict, Literal
import streamlit as st

from src.rag.application.use_cases.rag_chat import RAGChatService

from src.rag.infrastructure.config.types import Messages
from src.rag.infrastructure.logging.logger import get_usage_logger

Role = Literal["system", "user", "assistant"]


class ChatMessage(TypedDict):
    """Represents a single chat message with a role and content."""
    role: Role
    content: str


WELCOME: str = (
    "Bonjour, je suis l'assistant virtuel de la mairie. "
    "Comment puis-je vous aider aujourd'hui ?"
)


class ChatPage:
    """Streamlit chat page controller.

    Manages the chat state, displays messages, and communicates with
    the RAGChatService to generate assistant responses.
    """
    def __init__(self, service: RAGChatService, title: str) -> None:
        """Initialize the chat page.

        Args:
          service (RAGChatService): The RAG chat service used to generate
            responses.
          title (str): The page title displayed in the Streamlit interface.
        """
        self.service = service
        self.title = title
        self.usage_logger = get_usage_logger()

    # ---------- state ----------
    def _ensure_state(self) -> None:
        """Ensure session state is initialized for messages and page config."""
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
        """Render all messages from the chat history."""
        for idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                # Add feedback controls only for assistant messages
                if msg["role"] == "assistant" and not (
                    idx == 0 and msg.get("content") == WELCOME
                ):
                    self._render_feedback(idx, msg)

    def _render_feedback(self, idx: int, msg: ChatMessage) -> None:
        """Render a thumbs feedback widget for an assistant message.

        Args:
          idx (int): Index of the message in the session history.
          msg (ChatMessage): The assistant message to annotate with feedback.
        """
        selection_key, comment_key, submitted_key = self._feedback_keys(idx)

        selection = st.feedback("thumbs", key=selection_key)

        if selection is not None and not st.session_state.get(submitted_key):
            rating = "thumb_up" if selection == 1 else "thumb_down"
            comment = st.text_input(
                "Commentaire (optionnel)",
                key=comment_key,
                placeholder="Qu'est-ce qui était utile ou à améliorer ?",
            )
            if st.button("Envoyer le feedback", key=f"fb_send_{idx}"):
                self._submit_feedback(idx, rating, comment, msg, submitted_key)

        if st.session_state.get(submitted_key):
            st.caption("Feedback enregistré ✔︎")

    def _feedback_keys(self, idx: int) -> tuple[str, str, str]:
        """Return stable Streamlit widget keys for a given message index."""
        selection_key = f"fb_sel_{idx}"
        comment_key = f"fb_comment_{idx}"
        submitted_key = f"fb_submitted_{idx}"
        return selection_key, comment_key, submitted_key

    def _submit_feedback(
        self,
        idx: int,
        rating: str,
        comment: str | None,
        msg: ChatMessage,
        submitted_key: str,
    ) -> None:
        """Validate, log and acknowledge feedback submission."""
        try:
            answer_preview = self._preview(msg.get("content", ""), 400)
            comment_preview = self._preview(comment or "", 400)
            log_line = (
                f"Feedback | msg_index={idx} | rating={rating} "
                f"| comment={comment_preview} | answer={answer_preview}"
            )
            self.usage_logger.info(log_line)
        except Exception as e:
            st.warning(f"Impossible d'enregistrer le feedback : {e}")
            return
        st.session_state[submitted_key] = True
        st.toast("Merci pour votre feedback.")

    def _preview(self, text: str, max_len: int) -> str:
        """Single-line, length-limited preview for logging and UI."""
        if not text:
            return ""
        one_line = text.replace("\n", " ").strip()
        return (
            one_line if len(one_line) <= max_len else one_line[:max_len] + "…"
            )

    def _append(self, role: Role, content: str) -> None:
        """Append a message to the Streamlit session state.

        Args:
          role (Role): The message sender role ("user", "assistant", or
            "system").
          content (str): The message text.
        """
        st.session_state.messages.append({"role": role, "content": content})

    def _handle_user_prompt(self, prompt: str) -> None:
        """Handle a new user prompt end-to-end.

        Args:
          prompt (str): The user's message text.
        """
        self._render_user_message(prompt)
        self._produce_assistant_reply(prompt)

    def _render_user_message(self, prompt: str) -> None:
        """Append and render the user's message.

        Args:
          prompt (str): The user's message text.
        """
        self._append("user", prompt)
        with st.chat_message("user"):
            st.write(prompt)

    def _produce_assistant_reply(self, prompt: str) -> str:
        """Generate, render, log, and append the assistant's reply.

        The reply is produced by the service, displayed with a placeholder,
        and appended to the session history. Feedback controls are rendered
        inline so the user can rate the answer immediately.

        Args:
          prompt (str): The user's message text.

        Returns:
          str: The assistant's reply text (or an error message).
        """
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("_En cours..._")
            try:
                reply = self.service.answer(
                    history=st.session_state.messages,
                    question=prompt,
                )
                answer = reply["answer"]
                self.usage_logger.info(
                    f"User: {prompt} | Response: {answer}"
                )
            except Exception as e:
                answer = (
                    f"Erreur lors de la génération de la réponse : {e}"
                )
                self.usage_logger.error(
                    f"User: {prompt} | Error: {e}"
                )
            placeholder.write(answer)

            # Persist the assistant message before rendering feedback
            self._append("assistant", answer)

            # Immediately show feedback controls for this answer
            try:
                last_idx = len(st.session_state.messages) - 1
                self._render_feedback(
                    last_idx, st.session_state.messages[last_idx]
                )
            except Exception:
                # Fail-safe: ignore UI errors on reruns
                pass

            return answer

    # ---------- public ----------
    def render(self) -> None:
        """Render the chat interface and handle new user input."""
        self._ensure_state()
        st.title(self.title)

        self._render_history()

        prompt = st.chat_input("Votre question")
        if prompt:
            self._handle_user_prompt(prompt)
