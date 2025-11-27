"""Sidebar layout components for the Streamlit UI.

Defines helper functions and containers used to build the application's
sidebar, such as configuration controls and navigation elements. Contains
no business logic.
"""
import streamlit as st


def render_sidebar():
    st.title("Menu")
    st.markdown("---")
    st.info("RAG local – FULL RAG")
