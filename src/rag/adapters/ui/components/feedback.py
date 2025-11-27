"""Streamlit feedback UI components for chat messages.

Defines reusable controls for collecting thumbs-up/down ratings and optional
comments on assistant answers within the chat interface.
"""
import streamlit as st
from src.rag.infrastructure.logging.logger import get_usage_logger

logger = get_usage_logger()


def render_feedback(idx: int, msg: dict):
    key_sel = f"fb_sel_{idx}"
    key_com = f"fb_comment_{idx}"
    key_sub = f"fb_submitted_{idx}"

    selection = st.feedback("thumbs", key=key_sel)

    if selection is None:
        return

    if not st.session_state.get(key_sub):
        rating = "thumb_up" if selection == 1 else "thumb_down"
        comment = st.text_input(
            "Commentaire (optionnel)",
            key=key_com,
            placeholder="Qu'est-ce qui était utile ou à améliorer ?",
        )
        if st.button("Envoyer le feedback", key=f"fb_send_{idx}"):
            logger.info(
                f"Feedback | idx={idx} | rating={rating} "
                f"| msg='{msg['content']}' | comment='{comment}'"
            )
            st.session_state[key_sub] = True
            st.caption("Feedback enregistré")
    else:
        st.caption("Feedback enregistré")
