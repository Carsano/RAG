"""
Entry point Streamlit
"""
import streamlit as st
from layout.sidebar import render_sidebar
from pages import chat

PAGES = {
    "Chat": chat,
}


def main():
    st.set_page_config(page_title="FULL RAG", page_icon="🏛️")

    with st.sidebar:
        render_sidebar()

        choice = st.radio(
            "Navigation",
            list(PAGES.keys()),
            key="nav_choice"
        )

    PAGES[choice].render()


if __name__ == "__main__":
    main()
