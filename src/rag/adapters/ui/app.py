"""Streamlit application entry point.

Wires the sidebar, page routing and logging, and renders the main chat UI.
"""
import streamlit as st
from layout.sidebar import render_sidebar
from ui_pages import chat
from src.rag.infrastructure.logging.logger import get_usage_logger

PAGES = {
    "Chat": chat,
}


def main():
    usage_logger = get_usage_logger()
    usage_logger.info("Streamlit app started by user")

    st.set_page_config(page_title="FULL RAG", page_icon="🏛️")

    with st.sidebar:
        render_sidebar()

        choice = st.radio(
            "Navigation",
            list(PAGES.keys()),
            key="nav_choice"
        )

    usage_logger.info(f"Page selected: {choice}")

    PAGES[choice].render()


if __name__ == "__main__":
    main()
