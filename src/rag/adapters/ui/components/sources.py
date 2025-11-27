"""RAG source display components for the Streamlit UI.

Provides helpers to render cited documents, titles and snippets associated
with a given assistant answer in expandable sections.
"""
import streamlit as st


def render_sources(sources):
    with st.expander("Sources citées"):
        for src in sources:
            title = src.get("title", "Source")
            snippet = src.get("snippet", "")
            with st.expander(title):
                st.markdown(snippet)
