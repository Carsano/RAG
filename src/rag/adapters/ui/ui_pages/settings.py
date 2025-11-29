"""Streamlit settings page for the RAG assistant.

Allows the user to tweak UI / chat behaviour.
For now settings are kept in `st.session_state` only.
"""
from __future__ import annotations

from typing import Any, Dict

import streamlit as st


DEFAULT_SETTINGS: Dict[str, Any] = {
    "temperature": 0.05,
    "top_k": 10,
    "max_answer_tokens": 512,
    "show_sources": True,
    "log_interactions": True,
    "max_sources": 3,
}


def _init_settings_state() -> None:
    if "settings" not in st.session_state:
        st.session_state.settings = DEFAULT_SETTINGS.copy()


def render() -> None:
    _init_settings_state()

    st.title("Paramètres")
    st.caption("Configurez le comportement de l'assistant RAG.")

    with st.form("settings_form"):
        st.subheader("Génération")
        temperature = st.slider(
            "Température",
            min_value=0.05,
            max_value=1.0,
            value=float(st.session_state.settings.get("temperature", 0.05)),
            step=0.05,
            help="Plus la température est élevée,"
            "plus les réponses sont créatives.",
        )

        top_k = st.slider(
            "Top K",
            min_value=0,
            max_value=10,
            value=int(st.session_state.settings.get("top_k", 5)),
            step=1,
            help="Plus le top k est élevé,"
            "plus le nombre de sources utilisées est élevé.",
        )

        max_sources = st.slider(
            "Nombre maximal de sources affichées",
            min_value=1,
            max_value=10,
            value=int(st.session_state.settings.get("max_sources", 3)),
            step=1,
            help="Limite le nombre de documents affichés comme sources.",
        )

        max_answer_tokens = st.number_input(
            "Taille maximale des réponses (tokens)",
            min_value=64,
            max_value=4096,
            value=int(st.session_state.settings.get("max_answer_tokens", 512)),
            step=64,
            help="Limite la longueur des réponses générées.",
        )

        st.subheader("Affichage et journalisation")
        show_sources = st.checkbox(
            "Afficher les sources par défaut",
            value=bool(st.session_state.settings.get("show_sources", True)),
        )
        log_interactions = st.checkbox(
            "Journaliser les interactions utilisateur",
            value=bool(st.session_state.settings.get("log_interactions",
                                                     True)),
        )

        submitted = st.form_submit_button("Enregistrer les paramètres")

        if submitted:
            st.session_state.settings.update(
                {
                    "temperature": float(temperature),
                    "top_k": int(top_k),
                    "max_answer_tokens": int(max_answer_tokens),
                    "show_sources": bool(show_sources),
                    "log_interactions": bool(log_interactions),
                    "max_sources": int(max_sources),
                }
            )
            st.success("Paramètres mis à jour pour cette session.")
