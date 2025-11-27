"""Reusable Streamlit chat message component.

Provides helpers to render individual user and assistant messages with a
consistent layout and styling within the chat interface.
"""
import streamlit as st


def render_message(msg):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
