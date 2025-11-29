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
    "reranker_type": "llm_reranker",
}


def _init_settings_state() -> None:
    if "settings" not in st.session_state:
        st.session_state.settings = DEFAULT_SETTINGS.copy()
    # Initialize UI slider state from settings if not already set
    if "top_k_slider" not in st.session_state:
        st.session_state.top_k_slider = int(
            st.session_state.settings.get("top_k", 10)
        )
    if "max_sources_slider" not in st.session_state:
        st.session_state.max_sources_slider = int(
            st.session_state.settings.get("max_sources", 3)
        )


def _sync_from_max_sources() -> None:
    """Ensure top_k is always >= max_sources when max_sources changes."""
    max_sources = int(st.session_state.get("max_sources_slider", 1))
    top_k = int(st.session_state.get("top_k_slider", 0))
    if max_sources > top_k:
        st.session_state.top_k_slider = max_sources


def _sync_from_top_k() -> None:
    """Ensure max_sources is always <= top_k when top_k changes."""
    max_sources = int(st.session_state.get("max_sources_slider", 1))
    top_k = int(st.session_state.get("top_k_slider", 0))
    if top_k < max_sources:
        st.session_state.max_sources_slider = top_k


def render() -> None:
    _init_settings_state()

    st.title("Paramètres")
    st.caption("Configurez le comportement de l'assistant RAG.")

    st.subheader("Récupération")

    top_k = st.slider(
        "Top K",
        min_value=1,
        max_value=10,
        value=int(st.session_state.settings.get("top_k", 5)),
        step=1,
        help="Plus le top k est élevé,"
        "plus le nombre de sources utilisées est élevé.",
        key="top_k_slider",
        on_change=_sync_from_top_k,
    )

    max_sources = st.slider(
        "Nombre maximal de sources affichées",
        min_value=1,
        max_value=10,
        value=int(st.session_state.settings.get("max_sources", 3)),
        step=1,
        help="Limite le nombre de documents affichés comme sources.",
        key="max_sources_slider",
        on_change=_sync_from_max_sources,
    )

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
        max_answer_tokens = st.number_input(
            "Taille maximale des réponses (tokens)",
            min_value=64,
            max_value=4096,
            value=int(st.session_state.settings.get("max_answer_tokens", 512)),
            step=64,
            help="Limite la longueur des réponses générées.",
        )
        available_rerankers = [
            "llm_reranker",
            "keyword_overlap_scorer",
            "cross_encoder",
        ]
        current_reranker = st.session_state.settings.get(
            "reranker_type", "llm_reranker"
        )
        default_index = available_rerankers.index(current_reranker)

        def _format_reranker_label(key: str) -> str:
            if key == "keyword_overlap_scorer":
                return "Keyword overlap (rapide, baseline)"
            if key == "cross_encoder":
                return "Cross-encoder (plus précis, plus lent)"
            if key == "llm_reranker":
                return "LLM reranker (très précis, coûteux)"
            return key

        reranker_type = st.selectbox(
            "Reranker",
            options=available_rerankers,
            format_func=_format_reranker_label,
            index=default_index,
            help=(
                "Choisissez le reranker utilisé pour ordonner les "
                "documents récupérés."
            ),
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
            top_k_value = int(
                st.session_state.get("top_k_slider", top_k)
            )
            max_sources_value = int(
                st.session_state.get("max_sources_slider", max_sources)
            )
            st.session_state.settings.update(
                {
                    "temperature": float(temperature),
                    "top_k": top_k_value,
                    "max_answer_tokens": int(max_answer_tokens),
                    "show_sources": bool(show_sources),
                    "log_interactions": bool(log_interactions),
                    "max_sources": max_sources_value,
                    "reranker_type": str(reranker_type),
                }
            )
            st.success("Paramètres mis à jour pour cette session.")
