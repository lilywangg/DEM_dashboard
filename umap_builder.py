"""
umap_builder.py — UMAP figure builders for the three view modes
"""

from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from data import _cpd_meta_map
from renders import C, PLOT_L

# ── Color constants ────────────────────────────────────────────────────────────
_UMAP_COLORSCALE = [
    [0.0,  "#4ade80"], [0.35, "#60aaff"],
    [0.6,  "#fde68a"], [0.8,  "#fb923c"], [1.0,  "#f87171"],
]
_LINEAGE_PALETTE = px.colors.qualitative.Bold + px.colors.qualitative.Set2


# ── Default view — colored by IC50 ────────────────────────────────────────────
def _build_umap_default(ex, ey, ic, cl, dr, tgt_label) -> go.Figure:
    """All points colored by IC50 value on a green→red colorscale."""
    cd = list(zip(cl, dr, ic))
    fig = go.Figure(go.Scatter(
        x=ex, y=ey, mode="markers",
        marker=dict(
            color=ic, colorscale=_UMAP_COLORSCALE, size=4, opacity=0.75,
            colorbar=dict(
                title=dict(text=tgt_label, font=dict(color=C["text"], size=11)),
                tickfont=dict(color=C["text"]), len=0.7,
            ),
            showscale=True,
        ),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>Drug: %{customdata[1]}<br>"
            + tgt_label + ": %{customdata[2]:.3f}<extra></extra>"
        ),
        customdata=cd,
    ))
    return fig


# ── Pharmacological view — highlight selected drugs ────────────────────────────
def _build_umap_drug(ex, ey, ic, cl, dr, lineages, tgt_label, selected_drugs) -> go.Figure:
    """Selected drugs highlighted in color, all others faded."""
    cd     = list(zip(cl, dr, ic, lineages))
    mask   = np.array([d in selected_drugs for d in dr])
    traces = []

    # Background — unselected points faded
    bg_idx = [i for i, m in enumerate(mask) if not m]
    if bg_idx:
        traces.append(go.Scatter(
            x=ex[bg_idx], y=ey[bg_idx], mode="markers", name="Other",
            marker=dict(color="#333355", size=3, opacity=0.2),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>Drug: %{customdata[1]}<br>"
                + tgt_label + ": %{customdata[2]:.3f}<extra></extra>"
            ),
            customdata=[cd[i] for i in bg_idx],
            showlegend=False,
        ))

    # Foreground — one trace per selected drug for a proper legend entry
    for drug in sorted(selected_drugs):
        d_idx = [i for i, d in enumerate(dr) if d == drug]
        if not d_idx:
            continue
        info   = _cpd_meta_map.get(drug, {})
        target = info.get("gene_symbol_of_protein_target", "") or ""
        traces.append(go.Scatter(
            x=ex[d_idx], y=ey[d_idx], mode="markers",
            name=f"{drug}" + (f"  [{target}]" if target else ""),
            marker=dict(size=8, opacity=0.9),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>Drug: %{customdata[1]}<br>"
                "Lineage: %{customdata[3]}<br>"
                + tgt_label + ": %{customdata[2]:.3f}<extra></extra>"
            ),
            customdata=[cd[i] for i in d_idx],
        ))

    return go.Figure(traces)


# ── Tissue view — highlight selected lineages ──────────────────────────────────
def _build_umap_lineage(ex, ey, ic, cl, dr, lineages, tgt_label, selected_lineages) -> go.Figure:
    """Selected cancer lineages highlighted in color, all others faded."""
    all_lins      = sorted(set(lineages) - {"Unknown", ""})
    lin_color_map = {
        l: _LINEAGE_PALETTE[i % len(_LINEAGE_PALETTE)]
        for i, l in enumerate(all_lins)
    }
    cd     = list(zip(cl, dr, ic, lineages))
    traces = []

    # Background
    bg_idx = [i for i, l in enumerate(lineages) if l not in selected_lineages]
    if bg_idx:
        traces.append(go.Scatter(
            x=ex[bg_idx], y=ey[bg_idx], mode="markers", name="Other",
            marker=dict(color="#333355", size=3, opacity=0.15),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>Drug: %{customdata[1]}<br>"
                "Lineage: %{customdata[3]}<br>"
                + tgt_label + ": %{customdata[2]:.3f}<extra></extra>"
            ),
            customdata=[cd[i] for i in bg_idx],
            showlegend=False,
        ))

    # Foreground — one trace per selected lineage
    for lin in sorted(selected_lineages):
        l_idx = [i for i, l in enumerate(lineages) if l == lin]
        if not l_idx:
            continue
        traces.append(go.Scatter(
            x=ex[l_idx], y=ey[l_idx], mode="markers",
            name=lin,
            marker=dict(color=lin_color_map.get(lin, C["blue"]), size=7, opacity=0.88),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>Drug: %{customdata[1]}<br>"
                "Lineage: %{customdata[3]}<br>"
                + tgt_label + ": %{customdata[2]:.3f}<extra></extra>"
            ),
            customdata=[cd[i] for i in l_idx],
        ))

    return go.Figure(traces)


# ── Shared layout applier ──────────────────────────────────────────────────────
def _apply_umap_layout(fig: go.Figure, title: str, n: int) -> go.Figure:
    """Apply consistent dark-theme layout to any UMAP figure."""
    umap_base = {k: v for k, v in PLOT_L.items() if k not in ("xaxis", "yaxis", "legend")}
    fig.update_layout(
        **umap_base,
        title=f"{title}  (n={n:,})",
        xaxis=dict(**PLOT_L["xaxis"], title="UMAP 1", showticklabels=False),
        yaxis=dict(**PLOT_L["yaxis"], title="UMAP 2", showticklabels=False),
        height=500,
        dragmode="select",
        legend=dict(
            bgcolor="rgba(0,0,0,0.4)", font=dict(color=C["text"], size=10),
            x=0.01, y=0.99, bordercolor=C["border"], borderwidth=1,
        ),
    )
    return fig