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
        # Unique keys per message to keep widget state stable across reruns
        selection_key = f"fb_sel_{idx}"
        comment_key = f"fb_comment_{idx}"
        submitted_key = f"fb_submitted_{idx}"

        # Display a compact feedback widget (built-in Streamlit)
        selection = st.feedback("thumbs", key=selection_key)

        # If user selected a rating, allow optional comment + submit
        if selection is not None and not st.session_state.get(submitted_key):
            # Map numeric selection to label for logging clarity
            rating = "thumb_up" if selection == 1 else "thumb_down"
            comment = st.text_input(
                "Commentaire (optionnel)",
                key=comment_key,
                placeholder="Qu'est-ce qui était utile ou à améliorer ?",
            )
            if st.button("Envoyer le feedback", key=f"fb_send_{idx}"):
                # Log structured feedback (format into a single string)
                # that don't support printf-style args)
                try:
                    answer_preview = (msg.get(
                        "content", "") or "").replace("\n", " ").strip()
                    if len(answer_preview) > 400:
                        answer_preview = answer_preview[:400] + "…"
                    comment_preview = (
                        comment or "").replace("\n", " ").strip()
                    if len(comment_preview) > 400:
                        comment_preview = comment_preview[:400] + "…"
                    log_line = (
                        f"Feedback | msg_index={idx} | rating={rating} "
                        f"| comment={comment_preview} | "
                        f"answer={answer_preview}"
                    )
                    self.usage_logger.info(log_line)
                except Exception as e:  # Fallback in case the logger raises
                    st.warning(f"Impossible d'enregistrer le feedback : {e}")
                else:
                    st.session_state[submitted_key] = True
                    st.toast("Merci pour votre feedback.")

        # If already submitted, show a subtle acknowledgement
        if st.session_state.get(submitted_key):
            st.caption("Feedback enregistré ✔︎")

    def _append(self, role: Role, content: str) -> None:
        """Append a message to the Streamlit session state.

        Args:
          role (Role): The message sender role ("user", "assistant", or
            "system").
          content (str): The message text.
        """
        st.session_state.messages.append({"role": role, "content": content})

    def _handle_user_prompt(self, prompt: str) -> None:
        """Process the user's input prompt and generate a response.

        Args:
          prompt (str): The user's message text.
        """
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

            # Immediately show feedback controls for the freshly rendered reply
            try:
                last_idx = len(st.session_state.messages) - 1
                # We are still inside the assistant chat_message context
                # so we can render inline
                self._render_feedback(last_idx,
                                      st.session_state.messages[last_idx])
            except Exception:
                # Fail-safe: do nothing if state is not yet consistent
                pass

    # ---------- public ----------
    def render(self) -> None:
        """Render the chat interface and handle new user input."""
        self._ensure_state()
        st.title(self.title)

        self._render_history()

        prompt = st.chat_input("Votre question")
        if prompt:
            self._handle_user_prompt(prompt)
