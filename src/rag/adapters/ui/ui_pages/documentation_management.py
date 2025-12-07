"""
Streamlit page for monitoring the documentation pipeline.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import streamlit as st
import plotly.graph_objects as go

from src.rag.application.use_cases.document_pipeline_comparator import (
    DocumentPipelineComparator,
    DocumentPipelineComparisonResult,
)
from src.rag.utils.utils import get_project_root

DOC_MANAGEMENT_STYLES = """
<style>
.doc-path-entry {
    background-color: rgba(255, 255, 255, 0.88);
    color: #111827;
    padding: 0.35rem 0.6rem;
    border-radius: 0.4rem;
    margin-bottom: 0.3rem;
    border: 1px solid rgba(15, 23, 42, 0.08);
    font-size: 0.9rem;
    font-family: "SFMono-Regular", Menlo, Monaco, Consolas, "Liberation Mono",
        "Courier New", monospace;
}
</style>
"""


@dataclass(frozen=True)
class SankeyNodeConfig:
    """Declarative definition of a Sankey node's appearance and position."""

    label: str
    color: str
    x: float
    y: float


SANKEY_NODE_CONFIGS: dict[str, SankeyNodeConfig] = {
    "docs_sources": SankeyNodeConfig("Docs sources", "#4b5563", 0.02, 0.50),
    "base_markdown": SankeyNodeConfig("Base Markdown", "#3da14b", 0.48, 0.48),
    "index_faiss": SankeyNodeConfig("Index FAISS", "#3da14b", 1, 0.40),
    "sources_not_converted": SankeyNodeConfig(
        "Sources non converties", "#c31644", 0.50, 0.92
    ),
    "markdown_not_indexed": SankeyNodeConfig(
        "Markdown non indexés", "#c31644", 1, 0.68
    ),
    "markdown_orphans": SankeyNodeConfig(
        "Markdown orphelins", "#c31644", 0.75, 0.9
    ),
    "index_orphans": SankeyNodeConfig(
        "Entrées index orphelines", "#c31644", 0.95, 0.25
    ),
}

SANKEY_NODE_KEYS: list[str] = [
    "docs_sources",
    "base_markdown",
    "index_faiss",
    "sources_not_converted",
    "markdown_not_indexed",
    "markdown_orphans",
    "index_orphans",
]
SANKEY_NODE_INDEX: dict[str, int] = {
    key: idx for idx, key in enumerate(SANKEY_NODE_KEYS)
}


def _node_id(key: str) -> int:
    """Return the index of a given node key."""
    return SANKEY_NODE_INDEX[key]


def _render_path_section(title: str, paths: list[pathlib.Path]) -> None:
    """Render a collapsible list of filesystem paths."""
    count = len(paths)
    with st.expander(f"{title} ({count})", expanded=bool(paths)):
        if not paths:
            st.write("Aucun fichier signalé.")
            return
        st.markdown(DOC_MANAGEMENT_STYLES, unsafe_allow_html=True)
        for path in paths:
            st.markdown(
                f'<div class="doc-path-entry">{path}</div>',
                unsafe_allow_html=True,
            )


def _status_badge(is_in_sync: bool) -> str:
    """Return a status string describing the synchronization state."""
    return "✅ Synchronisé" if is_in_sync else "⚠️ Désaligné"


def render() -> None:
    """Display documentation health metrics inside Streamlit."""
    st.title("Gestion de la Documentation")
    st.write(
        "Supervisez la conversion et l'indexation de vos documents. "
        "Cette page inspecte les dossiers sources, la base Markdown et "
        "l'index FAISS pour détecter les écarts."
    )

    project_root = get_project_root()
    source_root = project_root / "data" / "controlled_documentation"
    markdown_root = project_root / "data" / "clean_md_database"
    manifest = project_root / "data" / "indexes" / "all_chunk_sources.json"

    comparator = DocumentPipelineComparator(
        source_root=source_root,
        markdown_root=markdown_root,
        sources_manifest=manifest,
    )

    with st.spinner("Analyse du pipeline documentaire...", show_time=True):
        result = comparator.compare()

    st.subheader("Synthèse générale")
    st.write(_status_badge(result.is_in_sync))
    st.caption(
        f"Dossiers analysés : {source_root} → {markdown_root} → {manifest}"
    )

    st.subheader("Flux global")
    _render_sankey(result)

    conversion_tab, indexing_tab = st.tabs(
        ["Conversion — Sources vers Markdown",
         "Indexation — Markdown vers Index"]
    )

    with conversion_tab:
        conversion = result.conversion
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Sources", conversion.raw_total)
        col_b.metric("Markdown", conversion.markdown_total)
        col_c.metric("Statut", "OK" if conversion.is_in_sync else "À vérifier")
        _render_path_section(
            "Fichiers sources sans conversion Markdown",
            conversion.missing_markdown_sources,
        )
        _render_path_section(
            "Markdown orphelins (pas de source)",
            conversion.orphaned_markdown_files,
        )

    with indexing_tab:
        indexing = result.indexing
        col_d, col_e, col_f = st.columns(3)
        col_d.metric("Markdown", indexing.database_total)
        col_e.metric("Indexés", indexing.indexed_total)
        col_f.metric("Statut", "OK" if indexing.is_in_sync else "À vérifier")
        _render_path_section(
            "Markdown absents de l'index", indexing.missing_from_index
        )
        _render_path_section(
            "Entrées index orphelines", indexing.orphaned_index_entries
        )


def _render_sankey(result: DocumentPipelineComparisonResult) -> None:
    """Render a Sankey diagram describing document flows."""
    conversion = result.conversion
    indexing = result.indexing
    missing_markdown = len(conversion.missing_markdown_sources)
    orphan_markdown = len(conversion.orphaned_markdown_files)
    unindexed_markdown = len(indexing.missing_from_index)
    orphan_index_entries = len(indexing.orphaned_index_entries)

    converted_from_sources = max(conversion.raw_total - missing_markdown, 0)
    markdown_total = conversion.markdown_total
    markdown_extra = max(markdown_total - converted_from_sources, 0)
    indexed_docs = max(markdown_total - unindexed_markdown, 0)

    labels = [
        SANKEY_NODE_CONFIGS[key].label for key in SANKEY_NODE_KEYS
    ]
    node_colors = [
        SANKEY_NODE_CONFIGS[key].color for key in SANKEY_NODE_KEYS
    ]
    node_x = [SANKEY_NODE_CONFIGS[key].x for key in SANKEY_NODE_KEYS]
    node_y = [SANKEY_NODE_CONFIGS[key].y for key in SANKEY_NODE_KEYS]
    sources: list[int] = []
    targets: list[int] = []
    values: list[int] = []
    link_colors: list[str] = []

    def _add_flow(src: int, tgt: int, value: int, color: str) -> None:
        if value <= 0:
            return
        sources.append(src)
        targets.append(tgt)
        values.append(value)
        link_colors.append(color)

    green_flow = "#a7d5ae"
    red_flow = "#e496ab"

    _add_flow(
        _node_id("docs_sources"),
        _node_id("base_markdown"),
        converted_from_sources,
        green_flow,
    )
    _add_flow(
        _node_id("docs_sources"),
        _node_id("sources_not_converted"),
        missing_markdown,
        red_flow,
    )
    _add_flow(
        _node_id("base_markdown"),
        _node_id("markdown_orphans"),
        orphan_markdown or markdown_extra,
        red_flow,
    )
    _add_flow(
        _node_id("base_markdown"),
        _node_id("index_faiss"),
        indexed_docs,
        green_flow,
    )
    _add_flow(
        _node_id("base_markdown"),
        _node_id("markdown_not_indexed"),
        unindexed_markdown,
        red_flow,
    )
    _add_flow(
        _node_id("index_faiss"),
        _node_id("index_orphans"),
        orphan_index_entries,
        red_flow,
    )

    if not values:
        st.info("Aucun flux disponible pour construire le diagramme.")
        return

    sankey = go.Sankey(
        arrangement="snap",
        node=dict(
            pad=18,
            thickness=18,
            line=dict(color="rgba(0,0,0,0.4)", width=0.5),
            label=labels,
            color=node_colors,
            x=node_x,
            y=node_y,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors),
    )
    fig = go.Figure(data=[sankey])

    legend_entries = [
        ("Sources (gris neutre)", "#4b5563"),
        ("Flux convertis (vert)", green_flow),
        ("Flux anormaux (rouge)", red_flow),
    ]
    for label, color in legend_entries:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=12, color=color),
                legendgroup=label,
                showlegend=True,
                name=label,
                hoverinfo="none",
            )
        )

    fig.update_layout(
        margin=dict(l=0, r=0, t=15, b=15),
        font=dict(size=13, color="#0f172a"),
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(15,23,42,0.2)",
            borderwidth=1,
            font=dict(color="#0f172a"),
        ),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    st.plotly_chart(fig, use_container_width=True)


__all__ = ["render"]
