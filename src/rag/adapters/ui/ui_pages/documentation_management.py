"""
Streamlit page for monitoring the documentation pipeline.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import plotly.graph_objects as go
import streamlit as st

from src.rag.application.use_cases.document_conversion import (
    DocumentConversionUseCase,
)
from src.rag.application.use_cases.document_pipeline_comparator import (
    DocumentPipelineComparator,
    DocumentPipelineComparisonResult,
)
from src.rag.application.use_cases.documents_indexer import DocumentsIndexer
from src.rag.infrastructure.converters.converters import DocumentConverter
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

    conversion_tab, indexing_tab, cleanup_tab = st.tabs(
        [
            "Conversion — Sources vers Markdown",
            "Indexation — Markdown vers Index",
            "Maintenance — Nettoyage de l'index",
        ]
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
        _render_conversion_selection(
            conversion.missing_markdown_sources,
            source_root,
            markdown_root,
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
        _render_indexing_selection(indexing.missing_from_index, markdown_root)
        _render_path_section(
            "Entrées index orphelines", indexing.orphaned_index_entries
        )

    with cleanup_tab:
        _render_index_cleanup_section(manifest, markdown_root)


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
    st.plotly_chart(fig, width="stretch")


def _render_indexing_selection(
    missing_paths: list[pathlib.Path],
    markdown_root: pathlib.Path,
) -> None:
    """Render controls for selecting Markdown files to index."""
    if not missing_paths:
        return

    st.caption("Sélectionner les Markdown à indexer")
    options = [str(path) for path in missing_paths]
    selected = st.multiselect(
        "Fichiers à ajouter à l'index",
        options=options,
        key="missing_index_multiselect",
        help="Choisissez un ou plusieurs fichiers puis lancez l'indexation.",
    )
    if st.button(
        "Indexer la sélection",
        disabled=not selected,
        key="index_selection_button",
    ):
        paths = [pathlib.Path(p) for p in selected]
        with st.spinner("Indexation sélective en cours..."):
            success, message = _index_selected_markdown(paths, markdown_root)
        if success:
            st.success(message)
            _rerun_app()
        else:
            st.error(message)


def _render_conversion_selection(
    missing_sources: list[pathlib.Path],
    source_root: pathlib.Path,
    markdown_root: pathlib.Path,
) -> None:
    """Render conversion actions for missing Markdown outputs."""
    if not missing_sources:
        return
    st.caption("Convertir les sources sélectionnées en Markdown")
    options = [str(path) for path in missing_sources]
    selected = st.multiselect(
        "Sélectionner des fichiers sources",
        options=options,
        key="convert_sources_multiselect",
        help="Choisissez un ou plusieurs fichiers à convertir.",
    )
    if st.button(
        "Convertir en Markdown",
        disabled=not selected,
        key="convert_sources_button",
    ):
        paths = [pathlib.Path(p) for p in selected]
        with st.spinner("Conversion des documents sources..."):
            success, message = _convert_sources_to_markdown(
                paths,
                source_root,
                markdown_root,
            )
        if success:
            st.success(message)
            _rerun_app()
        else:
            st.error(message)


def _render_index_cleanup_section(
    manifest_path: pathlib.Path, markdown_root: pathlib.Path
) -> None:
    """Render index cleanup tab with deletion controls."""
    st.caption(
        "Supprimez des documents de l'index FAISS "
        "en les sélectionnant ci-dessous."
    )
    indexed_paths, total_entries = _load_manifest_entries(manifest_path)
    unique_total = len(indexed_paths)
    faiss_root = markdown_root.parent / "indexes"
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    row1_col1.metric("Documents indexés (manifest)", unique_total)
    row1_col2.metric(
        "Fichiers .idx présents",
        _count_idx_files(faiss_root),
        help="Compte uniquement les fichiers d'index FAISS (.idx).",
    )
    row1_col3.metric("Chunks enregistrés", total_entries)

    row2_col1, row2_col2, row2_col3 = st.columns(3)
    row2_col1.metric(
        "Chunks moyens par document",
        round(total_entries / unique_total, 2) if unique_total else 0,
    )

    row2_col3.metric(
        "Dimension embeddings",
        _read_embedding_dimension(faiss_root / "faiss_index.idx"),
    )
    if not indexed_paths:
        st.info("Aucun document indexé trouvé dans le manifest.")
        return
    options = [str(path) for path in indexed_paths]
    selected = st.multiselect(
        "Documents indexés",
        options=options,
        key="indexed_docs_multiselect",
        help="Sélectionnez un ou plusieurs documents à retirer de l'index.",
    )
    if st.button(
        "Supprimer de l'index",
        disabled=not selected,
        key="cleanup_index_button",
    ):
        paths = [pathlib.Path(p) for p in selected]
        with st.spinner("Suppression des documents de l'index..."):
            success, message = _remove_indexed_files(paths, markdown_root)
        if success:
            st.success(message)
            _rerun_app()
        else:
            st.error(message)


def _index_selected_markdown(
    paths: list[pathlib.Path],
    markdown_root: pathlib.Path,
) -> tuple[bool, str]:
    """Run the selective indexing workflow for chosen Markdown files."""
    existing_paths = [path for path in paths if path.exists()]
    if not existing_paths:
        return False, "Aucun fichier valide sélectionné."

    try:
        indexer = DocumentsIndexer(root=markdown_root)
        indexer.index_files(existing_paths)
    except Exception as exc:
        return False, f"Indexation échouée: {exc}"
    return True, "Indexation terminée."


def _convert_sources_to_markdown(
    paths: list[pathlib.Path],
    source_root: pathlib.Path,
    markdown_root: pathlib.Path,
) -> tuple[bool, str]:
    """Convert selected source documents into Markdown outputs."""
    existing_paths = [path for path in paths if path.exists()]
    if not existing_paths:
        return False, "Aucun fichier source valide sélectionné."
    try:
        converter = DocumentConverter(
            input_root=source_root,
            output_root=markdown_root,
        )
        use_case = DocumentConversionUseCase(converter=converter)
        outputs = use_case.run_for_files(existing_paths)
    except Exception as exc:
        return False, f"Conversion échouée: {exc}"
    converted_count = len(outputs)
    total_requested = len(existing_paths)
    if converted_count < total_requested:
        return (
            False,
            "Certaines conversions ont échoué. "
            "Consultez les logs ou contactez l'administrateur.",
        )
    return True, f"Conversion terminée | fichiers convertis: {converted_count}"


def _load_manifest_entries(
    manifest_path: pathlib.Path,
) -> tuple[list[pathlib.Path], int]:
    """Load manifest entries along with total chunk count."""
    if not manifest_path.exists():
        return [], 0
    try:
        raw_entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [], 0
    normalized = []
    for entry in raw_entries:
        if not entry:
            continue
        candidate = pathlib.Path(entry).expanduser()
        if not candidate.is_absolute():
            candidate = manifest_path.parent / candidate
        normalized.append(candidate.resolve())
    unique = sorted(set(normalized))
    return unique, len(normalized)


def _remove_indexed_files(
    paths: list[pathlib.Path],
    markdown_root: pathlib.Path,
) -> tuple[bool, str]:
    """Remove selected files from the index."""
    if not paths:
        return False, "Aucun fichier sélectionné."
    indexer = DocumentsIndexer(root=markdown_root)
    if hasattr(indexer, "remove_files"):
        total_removed = indexer.remove_files(paths)
    else:
        total_removed = sum(indexer.remove_file(path) for path in paths)
    if total_removed == 0:
        return False, "Aucun vecteur supprimé pour les fichiers choisis."
    return True, f"Suppression terminée | chunks retirés: {total_removed}"


def _count_idx_files(index_root: pathlib.Path) -> int:
    """Count FAISS .idx files in the indexes folder."""
    if not index_root.exists():
        return 0
    return sum(1 for path in index_root.iterdir()
               if path.is_file() and path.suffix == ".idx")


def _read_embedding_dimension(index_file: pathlib.Path) -> int:
    """Read embedding dimension from the FAISS index, if available."""
    if not index_file.exists():
        return 0
    try:
        import faiss  # type: ignore

        idx = faiss.read_index(str(index_file))
        return int(getattr(idx, "d", 0))
    except Exception:  # pragma: no cover - optional dependency
        return 0


def _rerun_app() -> None:
    """Trigger a Streamlit rerun compatible with legacy/new APIs."""
    rerun = getattr(st, "rerun", None)
    if rerun:
        rerun()
        return
    experimental = getattr(st, "experimental_rerun", None)
    if experimental:
        experimental()


__all__ = ["render"]
