"""
Entry point Streamlit
"""
import streamlit as st
from pages import chat

PAGES = {
    "Chat": chat
}


def main():
    with st.sidebar:
        choice = st.radio("Navigation", list(PAGES.keys()))

    PAGES[choice].render()


if __name__ == "__main__":
    main()