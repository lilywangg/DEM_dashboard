"""
dashboard.py — DEM Interactive Dashboard
-----------------------------------------
"""

from __future__ import annotations
import json, warnings
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import diskcache
from dash import Dash, dcc, html, Input, Output, State, callback, ALL, ctx, no_update
from dash import exceptions as _dash_exc
from dash import DiskcacheManager
from dotenv import load_dotenv

PreventUpdate = _dash_exc.PreventUpdate
warnings.filterwarnings("ignore")
load_dotenv()

# ── Local modules ──────────────────────────────────────────────────────────────
from data import (
    BASE, RF_DIR,
    TOTAL_ROWS, TOTAL_DRUGS, TOTAL_CELLS,
    ic50_full, auc_full, IC50_LO, IC50_HI, AUC_LO, AUC_HI,
    all_drugs_full, all_cells_full, all_lineages_full,
    _meta_map, apply_filters, cfg_hash, _ensure_features_loaded,
)
from llm import _call_gemini
from evidence import _get_gene_evidence, _get_neighbor_evidence, _get_cluster_evidence
from prompts import (
    _build_global_prompt, _build_cluster_prompt,
    _build_pair_prompt, _build_point_prompt,
)
from renders import (
    C, CARD_S, LABEL_S, SECTION_S, PLOT_L, SECTION_GREY,
    dd_style, inp_style, stat_box, model_complexity,
    _llm_output_with_evidence, render_rf_metrics, render_rf_result, render_cached_runs,
)
from umap_builder import (
    _build_umap_default, _build_umap_drug, _build_umap_lineage, _apply_umap_layout,
)

# ── Background callback manager ───────────────────────────────────────────────
cache      = diskcache.Cache(str(BASE / ".dash_cache"))
bg_manager = DiskcacheManager(cache)

FILTER_SECTION_COLOR = "#9999bb"

# ── Layout helpers ─────────────────────────────────────────────────────────────
_TAB_S   = {
    "backgroundColor": C["surface"], "color": C["muted"], "fontSize": "11px",
    "fontFamily": "monospace", "border": f"1px solid {C['border']}", "padding": "6px 14px",
}
_TAB_SEL = lambda col: {
    "backgroundColor": C["card"], "color": col,
    "borderTop": f"2px solid {col}", "fontSize": "11px",
    "fontFamily": "monospace", "padding": "6px 14px",
}

# ── Filter panel ──────────────────────────────────────────────────────────────
filter_panel = html.Div(
    style={
        "backgroundColor": C["panel"], "padding": "16px", "borderRadius": "10px",
        "width": "265px", "flexShrink": "0", "overflowY": "auto",
        "maxHeight": "calc(100vh - 56px)", "border": f"1px solid {C['border']}",
    },
    children=[
        html.H4("Data Filters", style={
            "color": C["text"], "margin": "0 0 14px 0",
            "fontSize": "13px", "letterSpacing": "1px",
        }),

        html.Div("DATA QUALITY", style=SECTION_S),

        html.Div("Fit Parameters", style=LABEL_S),
        dcc.Checklist(id="q-fit",
            options=[{"label": " 2-param", "value": 2},
                     {"label": " 3-param", "value": 3}],
            value=[2, 3], className="pill-group",
            inputStyle={}, labelStyle={}),

        html.Div(style={"display": "flex", "alignItems": "center",
                        "justifyContent": "space-between",
                        "marginTop": "13px", "marginBottom": "3px"}, children=[
            html.Span("p1 CI width  (tight < 1.0)",
                      style={"color": "#555570", "fontSize": "11px", "fontWeight": "bold"}),
            dcc.Input(id="p1-num", type="number", value=10, min=0, max=10, step=0.25,
                      style={"backgroundColor": C["panel"], "color": C["text"],
                             "border": "none", "fontSize": "11px", "width": "40px",
                             "textAlign": "right", "padding": "0",
                             "fontFamily": "monospace", "outline": "none"}),
        ]),
        dcc.Slider(id="q-p1", min=0, max=10, step=0.25, value=10,
                   marks={}, tooltip={"placement": "bottom"}, className="slider-grey"),

        html.Div(style={"display": "flex", "alignItems": "center",
                        "justifyContent": "space-between",
                        "marginTop": "13px", "marginBottom": "3px"}, children=[
            html.Span("p2 CI width",
                      style={"color": "#555570", "fontSize": "11px", "fontWeight": "bold"}),
            dcc.Input(id="p2-num", type="number", value=50, min=0, max=50, step=1,
                      style={"backgroundColor": C["panel"], "color": C["text"],
                             "border": "none", "fontSize": "11px", "width": "40px",
                             "textAlign": "right", "padding": "0",
                             "fontFamily": "monospace", "outline": "none"}),
        ]),
        dcc.Slider(id="q-p2", min=0, max=50, step=1, value=50,
                   marks={}, tooltip={"placement": "bottom"}, className="slider-grey"),

        html.Div(style={"display": "flex", "alignItems": "center",
                        "justifyContent": "space-between",
                        "marginTop": "13px", "marginBottom": "3px"}, children=[
            html.Span("p4 CI width",
                      style={"color": "#555570", "fontSize": "11px", "fontWeight": "bold"}),
            dcc.Input(id="p4-num", type="number", value=5, min=0, max=5, step=0.1,
                      style={"backgroundColor": C["panel"], "color": C["text"],
                             "border": "none", "fontSize": "11px", "width": "40px",
                             "textAlign": "right", "padding": "0",
                             "fontFamily": "monospace", "outline": "none"}),
        ]),
        dcc.Slider(id="q-p4", min=0, max=5, step=0.1, value=5,
                   marks={}, tooltip={"placement": "bottom"}, className="slider-grey"),

        html.Div("Concentration Points (conc_pts_fit)", style=LABEL_S),
        dcc.RadioItems(id="q-conc",
            options=[{"label": " Any",      "value": 0},
                     {"label": " ≥ 12 pts", "value": 12},
                     {"label": " ≥ 16 pts", "value": 16}],
            value=0, className="pill-group", inputStyle={}, labelStyle={}),

        html.Div("Crosses 50% Inhibition", style=LABEL_S),
        dcc.RadioItems(id="q-crosses",
            options=[{"label": " All",            "value": "all"},
                     {"label": " Crosses only",   "value": "yes"},
                     {"label": " Does not cross", "value": "no"}],
            value="yes", className="pill-group", inputStyle={}, labelStyle={}),

        html.Div("BIOLOGICAL CHECKS", style={**SECTION_S, "marginTop": "16px"}),

        html.Div("SNP Fingerprint Status", style=LABEL_S),
        dcc.Checklist(id="q-snp",
            options=[{"label": " SNP-matched",  "value": "SNP-matched-reference"},
                     {"label": " Not tested",   "value": "SNP-not-tested"},
                     {"label": " Unconfirmed",  "value": "SNP-unconfirmed"}],
            value=["SNP-matched-reference"],
            className="pill-group", inputStyle={}, labelStyle={}),

        html.Div("No Extrapolation  (EC50 ≤ 2× top conc.)", style=LABEL_S),
        dcc.RadioItems(id="q-extrap",
            options=[{"label": " No filter", "value": "all"},
                     {"label": " Require",   "value": "yes"}],
            value="yes", className="pill-group", inputStyle={}, labelStyle={}),

        html.Div("High-Dose Inhibition  (pred_pv ≤ 0.9)", style=LABEL_S),
        dcc.RadioItems(id="q-dose",
            options=[{"label": " No filter", "value": "all"},
                     {"label": " Require",   "value": "yes"}],
            value="all", className="pill-group", inputStyle={}, labelStyle={}),

        html.Div("SAMPLE FILTERS", style={**SECTION_S, "marginTop": "16px"}),

        html.Div("Drug", style=LABEL_S),
        dcc.Dropdown(id="f-drug",
            options=[{"label": d, "value": d} for d in all_drugs_full],
            multi=True, placeholder="Search 545 drugs…", style=dd_style()),

        html.Div("Cell Line", style=LABEL_S),
        dcc.Dropdown(id="f-cell",
            options=[{"label": lbl, "value": mid} for lbl, mid in all_cells_full],
            multi=True, placeholder="Search 822 cell lines…", style=dd_style()),

        html.Div("Cancer Lineage", style=LABEL_S),
        dcc.Dropdown(id="f-lineage",
            options=[{"label": l, "value": l} for l in all_lineages_full],
            multi=True, placeholder="All lineages…", style=dd_style()),

        html.Div(id="filter-summary",
            style={"marginTop": "18px", "padding": "10px",
                   "backgroundColor": C["surface"], "borderRadius": "6px",
                   "fontSize": "12px", "lineHeight": "2.2",
                   "border": f"1px solid {C['border']}"}),
    ],
)

# ── Leaf-Agreement UMAP card ──────────────────────────────────────────────────
leaf_card = html.Div(style=CARD_S, children=[
    html.H4("Leaf-Agreement Kernel Space", style={
        "color": SECTION_GREY, "margin": "0 0 10px 0",
        "fontSize": "13px", "letterSpacing": "0.5px",
    }),
    html.Div(
        "Samples nearby in this UMAP share RF leaf assignments — "
        "the model routes them through the same gene-expression decision paths.",
        style={"color": C["muted"], "fontSize": "11px", "marginBottom": "10px"},
    ),

    dcc.Graph(
        id="umap-graph", style={"height": "500px"},
        figure={"layout": {
            "paper_bgcolor": C["card"], "plot_bgcolor": C["card"],
            "font": {"color": C["text"], "size": 11},
            "xaxis": {"showgrid": False, "showticklabels": False, "zeroline": False, "color": C["muted"]},
            "yaxis": {"showgrid": False, "showticklabels": False, "zeroline": False, "color": C["muted"]},
            "margin": {"l": 50, "r": 20, "t": 40, "b": 36},
            "annotations": [{"text": "Run RF above to generate the leaf-agreement embedding",
                              "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5,
                              "showarrow": False, "font": {"color": C["muted"], "size": 13}}],
        }},
        config={"displayModeBar": True,
                "modeBarButtonsToRemove": ["pan2d", "zoom2d", "autoScale2d"]},
    ),

    html.Div(style={"marginTop": "14px"}, children=[
        dcc.Tabs(id="umap-view-tabs", value="default", children=[
            dcc.Tab(label="Default View",         value="default",
                    style=_TAB_S, selected_style=_TAB_SEL(C["blue"])),
            dcc.Tab(label="Pharmacological View", value="drug",
                    style=_TAB_S, selected_style=_TAB_SEL(C["purple"])),
            dcc.Tab(label="Tissue View",          value="lineage",
                    style=_TAB_S, selected_style=_TAB_SEL(C["green"])),
        ]),
        html.Button("↺  Reset View", id="btn-reset-umap", n_clicks=0,
            style={"backgroundColor": "transparent", "color": C["muted"],
                "border": f"1px solid {C['border']}", "borderRadius": "5px",
                "padding": "4px 12px", "fontSize": "10px", "cursor": "pointer",
                "fontFamily": "monospace", "marginTop": "8px"}),
        html.Div(id="umap-drug-ctrl", style={"display": "none", "marginTop": "10px"}, children=[
            html.Div("Highlight Drug(s)", style=LABEL_S),
            dcc.Dropdown(id="umap-drug-filter", multi=True,
                         placeholder="Select drug(s) to highlight…", style=dd_style()),
            html.Div("Re-embeds UMAP using all pairs for selected drug(s).",
                     style={"color": C["muted"], "fontSize": "10px", "marginTop": "4px"}),
        ]),
        html.Div(id="umap-lineage-ctrl", style={"display": "none", "marginTop": "10px"}, children=[
            html.Div("Highlight Cancer Lineage(s)", style=LABEL_S),
            dcc.Dropdown(id="umap-lineage-filter", multi=True,
                         placeholder="Select lineage(s) to highlight…", style=dd_style()),
            html.Div("Re-embeds UMAP using all pairs for selected lineage(s).",
                     style={"color": C["muted"], "fontSize": "10px", "marginTop": "4px"}),
        ]),
    ]),

    html.Div(style={"marginTop": "16px", "borderTop": f"1px solid {C['border']}",
                    "paddingTop": "12px"}, children=[
        html.Div(style={"display": "flex", "alignItems": "center", "gap": "12px",
                        "flexWrap": "wrap", "marginBottom": "8px"}, children=[
            html.Span("AI Analysis", style={"color": C["purple"], "fontWeight": "bold", "fontSize": "12px"}),
            html.Span("Gemini 2.5 Flash", style={"color": C["muted"], "fontSize": "10px"}),
        ]),
        html.Div(style={"display": "flex", "gap": "10px", "alignItems": "flex-end",
                        "flexWrap": "wrap", "marginBottom": "10px"}, children=[
            html.Div([
                html.Div("Drug", style={**LABEL_S, "marginTop": "4px"}),
                dcc.Dropdown(id="ai-drug-select",
                             options=[{"label": d, "value": d} for d in all_drugs_full],
                             placeholder="Select drug…",
                             style={**dd_style(), "minWidth": "220px"}),
            ]),
            html.Div([
                html.Div("Cell Line", style={**LABEL_S, "marginTop": "4px"}),
                dcc.Dropdown(id="ai-cell-select",
                             options=[{"label": lbl, "value": mid} for lbl, mid in all_cells_full],
                             placeholder="Select cell line…",
                             style={**dd_style(), "minWidth": "240px"}),
            ]),
            html.Div([
                html.Div("\u00a0", style={**LABEL_S, "marginTop": "4px"}),
                html.Button("▶  Generate Analysis", id="btn-llm-pair", n_clicks=0,
                            disabled=True,
                            style={"backgroundColor": "transparent", "color": C["teal"],
                                   "border": f"1px solid {C['teal']}", "borderRadius": "5px",
                                   "padding": "5px 14px", "fontSize": "11px",
                                   "cursor": "not-allowed", "fontFamily": "monospace",
                                   "opacity": "0.35"}),
            ]),
        ]),
        html.Div(
            "Select Drug × Cell Line or click a point for Level 1 (Sample).  "
            "Lasso-select for Level 2 (Cluster).  Level 3 analyses the full UMAP.",
            style={"color": C["muted"], "fontSize": "10px", "marginBottom": "10px"},
        ),

        dcc.Loading(type="dot", color=C["teal"],   children=html.Div(id="llm-sample-output")),

        dcc.Loading(type="dot", color=C["purple"], children=html.Div(id="llm-cluster-output")),

        dcc.Loading(type="dot", color=C["orange"], children=html.Div(id="llm-global-output")),
    ]),
])

# ── Main content ──────────────────────────────────────────────────────────────
main_content = html.Div(style={"flex": "1", "minWidth": "0"}, children=[

    html.Div(style=CARD_S, children=[
        html.H4("Selected Data Overview", style={
            "color": SECTION_GREY, "margin": "0 0 12px 0",
            "fontSize": "13px", "letterSpacing": "0.5px",
        }),
        html.Div(id="data-stats", style={"display": "flex", "gap": "10px", "marginBottom": "10px"}),
        dcc.RadioItems(id="hist-scale",
            options=[{"label": " Density", "value": "density"},
                     {"label": " Count",   "value": "count"}],
            value="density", className="pill-group",
            inputStyle={}, labelStyle={}, style={"marginBottom": "10px"}),
        html.Div(style={"display": "flex", "gap": "14px"}, children=[
            html.Div(style={"flex": "1"}, children=[dcc.Graph(id="ic50-hist", style={"height": "230px"})]),
            html.Div(style={"flex": "1"}, children=[dcc.Graph(id="auc-hist",  style={"height": "230px"})]),
        ]),
    ]),

    html.Div(style=CARD_S, children=[
        html.H4("Random Forest Evaluation", style={
            "color": SECTION_GREY, "margin": "0 0 12px 0",
            "fontSize": "13px", "letterSpacing": "0.5px",
        }),
        html.Div(id="cache-notice", style={"fontSize": "11px", "color": C["teal"], "marginBottom": "10px"}),
        html.Div(style={"display": "flex", "gap": "18px", "flexWrap": "wrap",
                        "alignItems": "flex-end", "marginBottom": "14px"}, children=[
            html.Div([html.Div("n_estimators",        style=LABEL_S),
                      dcc.Input(id="hp-nest",   type="number", value=100, min=10,  max=500,  step=10,  style=inp_style())]),
            html.Div([html.Div("max_depth  (0=None)", style=LABEL_S),
                      dcc.Input(id="hp-depth",  type="number", value=0,   min=0,   max=50,   step=1,   style=inp_style())]),
            html.Div([html.Div("min_samples_leaf",    style=LABEL_S),
                      dcc.Input(id="hp-leaf",   type="number", value=5,   min=1,   max=50,   step=1,   style=inp_style())]),
            html.Div([html.Div("max_features (%)",    style=LABEL_S),
                      dcc.Input(id="hp-feat",   type="number", value=30,  min=5,   max=100,  step=5,   style=inp_style())]),
            html.Div([html.Div("CV folds",            style=LABEL_S),
                      dcc.Input(id="hp-folds",  type="number", value=1,   min=1,   max=10,   step=1,   style=inp_style("60px"))]),
            html.Div([html.Div("Test Split (%)",      style=LABEL_S),
                      dcc.Input(id="hp-split",  type="number", value=20,  min=10,  max=40,   step=5,   style=inp_style("60px"))]),
            html.Div([html.Div("UMAP Samples",        style=LABEL_S),
                      dcc.Input(id="hp-umap-n", type="number", value=500, min=100, max=3000, step=100, style=inp_style())]),
            html.Div([html.Div("Target Variable", style=LABEL_S),
                      dcc.RadioItems(id="hp-target",
                          options=[{"label": " log₁₀(IC50)", "value": "ic50"},
                                   {"label": " AUC",         "value": "auc"}],
                          value="ic50", className="pill-group", inputStyle={}, labelStyle={})]),
            html.Div([html.Div("\u00a0", style=LABEL_S),
                      html.Button("▶  Run RF", id="btn-rf", n_clicks=0,
                          style={"backgroundColor": C["purple"], "color": "#000",
                                 "border": "none", "borderRadius": "6px",
                                 "padding": "7px 20px", "fontWeight": "bold",
                                 "cursor": "pointer", "fontSize": "13px",
                                 "fontFamily": "monospace"})]),
        ]),

        html.Div(id="rf-log", style={
            "backgroundColor": C["surface"], "border": f"1px solid {C['border']}",
            "borderRadius": "6px", "padding": "10px 14px", "minHeight": "32px",
            "fontSize": "12px", "color": C["teal"], "fontFamily": "monospace",
            "marginBottom": "12px", "whiteSpace": "pre-wrap", "display": "none",
        }),

        html.Div(style={"marginBottom": "14px"}, children=[
            html.Div(style={"display": "flex", "alignItems": "center",
                            "gap": "16px", "marginBottom": "6px"}, children=[
                html.Span("Run History", style={"color": C["purple"], "fontWeight": "bold", "fontSize": "12px"}),
                dcc.RadioItems(id="hist-metric",
                    options=[{"label": " R²",   "value": "r2"},
                             {"label": " RMSE", "value": "rmse"}],
                    value="r2", className="pill-group", inputStyle={}, labelStyle={}),
                html.Span("  color = model complexity", style={"color": C["muted"], "fontSize": "10px"}),
            ]),
            dcc.Graph(id="rf-history-scatter", style={"height": "260px"},
                      config={"displayModeBar": False}),
        ]),

        dcc.Loading(type="circle", color=C["purple"], children=html.Div(id="rf-results")),
    ]),

    leaf_card,
    html.Div(id="cached-runs-section", children=render_cached_runs()),
])

# ── App ───────────────────────────────────────────────────────────────────────
app = Dash(
    __name__,
    title="Drug Efficacy Model (DEM) Dashboard",
    background_callback_manager=bg_manager,
)
app.layout = html.Div(
    style={"backgroundColor": C["bg"], "minHeight": "100vh", "padding": "16px",
           "fontFamily": "monospace", "color": C["text"]},
    children=[
        dcc.Store(id="umap-data-store"),
        dcc.Store(id="pending-delete-hash"),
        dcc.Store(id="delete-trigger", data=0),
        dcc.Store(id="level1-prompt-store"),
        dcc.Store(id="level2-prompt-store"),
        dcc.ConfirmDialog(id="confirm-delete",
            message="Are you sure you want to delete this cached run? This cannot be undone."),
        html.Div(style={"marginBottom": "14px"}, children=[
            html.H2("Drug Efficacy Model (DEM) Dashboard",
                    style={"color": C["text"], "margin": "0", "fontSize": "20px"}),
            html.P("Filter → Train → Explore → Explain",
                   style={"color": "#3a3a50", "margin": "2px 0 0", "fontSize": "11px"}),
        ]),
        html.Div(style={"display": "flex", "gap": "14px", "alignItems": "flex-start"},
                 children=[filter_panel, main_content]),
    ],
)


# ── Callbacks ─────────────────────────────────────────────────────────────────

@callback(
    Output("data-stats",     "children"),
    Output("ic50-hist",      "figure"),
    Output("auc-hist",       "figure"),
    Output("filter-summary", "children"),
    Output("cache-notice",   "children"),
    Input("hist-scale",  "value"),
    Input("q-fit",    "value"), Input("q-p1",     "value"),
    Input("q-p2",     "value"), Input("q-p4",     "value"),
    Input("q-crosses","value"), Input("q-conc",   "value"),
    Input("q-snp",    "value"), Input("q-extrap", "value"),
    Input("q-dose",   "value"),
    Input("f-drug",   "value"), Input("f-cell",   "value"),
    Input("f-lineage","value"),
)
def update_overview(hist_scale, fit, p1, p2, p4, crosses, conc, snp, extrap, dose,
                    drugs, cells, lineages):
    sub = apply_filters(fit, p1, p2, p4, crosses, conc, snp, extrap, dose, drugs, cells, lineages)
    n   = len(sub)
    nd  = sub["cpd_name"].nunique()
    nc  = sub["ModelID"].nunique()
    pct = f"{100 * n / TOTAL_ROWS:.1f}%"

    stats_row = [
        stat_box("Samples",    f"{n:,} / {TOTAL_ROWS:,}", C["blue"]),
        stat_box("Drugs",      f"{nd} / {TOTAL_DRUGS}",   C["purple"]),
        stat_box("Cell Lines", f"{nc} / {TOTAL_CELLS}",   C["teal"]),
        stat_box("% of Total", pct,                       C["orange"]),
    ]

    BIN_IC50 = dict(start=IC50_LO, end=IC50_HI, size=(IC50_HI - IC50_LO) / 48)
    BIN_AUC  = dict(start=AUC_LO,  end=AUC_HI,  size=(AUC_HI  - AUC_LO)  / 48)
    norm     = "probability density" if hist_scale == "density" else ""
    y_title  = "Density" if hist_scale == "density" else "Count"

    sel_ic50 = sub["log10_ic50"].dropna()
    sel_ic50 = sel_ic50[(sel_ic50 >= IC50_LO) & (sel_ic50 <= IC50_HI)]
    sel_auc  = sub["area_under_curve"].dropna()
    sel_auc  = sel_auc[(sel_auc >= AUC_LO) & (sel_auc <= AUC_HI)]

    ic50_fig = go.Figure()
    ic50_fig.add_trace(go.Histogram(
        x=ic50_full[(ic50_full >= IC50_LO) & (ic50_full <= IC50_HI)],
        xbins=BIN_IC50, name="Full dataset", marker_color="#cccccc", opacity=0.35, histnorm=norm))
    ic50_fig.add_trace(go.Histogram(
        x=sel_ic50, xbins=BIN_IC50, name="Selected", marker_color=C["blue"], opacity=0.75, histnorm=norm))
    ic50_fig.update_layout(**PLOT_L, barmode="overlay",
                           title="log₁₀(IC50) Distribution",
                           xaxis_title="log₁₀(IC50)", yaxis_title=y_title)

    auc_fig = go.Figure()
    auc_fig.add_trace(go.Histogram(
        x=auc_full[(auc_full >= AUC_LO) & (auc_full <= AUC_HI)],
        xbins=BIN_AUC, name="Full dataset", marker_color="#cccccc", opacity=0.35, histnorm=norm))
    auc_fig.add_trace(go.Histogram(
        x=sel_auc, xbins=BIN_AUC, name="Selected", marker_color=C["teal"], opacity=0.75, histnorm=norm))
    auc_fig.update_layout(**PLOT_L, barmode="overlay",
                          title="AUC Distribution", xaxis_title="AUC", yaxis_title=y_title)

    summary = [
        html.Span(f"{n:,}", style={"color": C["purple"], "fontWeight": "bold"}),
        html.Span(" rows selected"), html.Br(),
        html.Span(f"{nd}", style={"color": C["blue"], "fontWeight": "bold"}),
        html.Span(f" / {TOTAL_DRUGS} drugs"), html.Br(),
        html.Span(f"{nc}", style={"color": C["teal"], "fontWeight": "bold"}),
        html.Span(f" / {TOTAL_CELLS} cell lines"), html.Br(),
        html.Span(pct, style={"color": C["orange"], "fontWeight": "bold"}),
        html.Span(" of full dataset"),
    ]

    n_cached = len(list(RF_DIR.glob("*.json")))
    notice = ([html.Span(
        f"✓ {n_cached} cached RF run(s) — matching config loads instantly.",
        style={"color": C["teal"]})] if n_cached else [])

    return stats_row, ic50_fig, auc_fig, summary, notice


@callback(Output("p1-num", "value"), Input("q-p1", "value"))
def _p1_to_num(v): return v

@callback(Output("q-p1", "value"), Input("p1-num", "value"), prevent_initial_call=True)
def _num_to_p1(v): return v if v is not None else 10

@callback(Output("p2-num", "value"), Input("q-p2", "value"))
def _p2_to_num(v): return v

@callback(Output("q-p2", "value"), Input("p2-num", "value"), prevent_initial_call=True)
def _num_to_p2(v): return v if v is not None else 50

@callback(Output("p4-num", "value"), Input("q-p4", "value"))
def _p4_to_num(v): return v

@callback(Output("q-p4", "value"), Input("p4-num", "value"), prevent_initial_call=True)
def _num_to_p4(v): return v if v is not None else 5


@callback(
    Output("rf-results",          "children"),
    Output("umap-data-store",     "data"),
    Output("cached-runs-section", "children"),
    Input("btn-rf", "n_clicks"),
    State("q-fit",    "value"), State("q-p1",    "value"),
    State("q-p2",     "value"), State("q-p4",    "value"),
    State("q-crosses","value"), State("q-conc",  "value"),
    State("q-snp",    "value"), State("q-extrap","value"),
    State("q-dose",   "value"),
    State("f-drug",   "value"), State("f-cell",  "value"),
    State("f-lineage","value"),
    State("hp-nest",   "value"), State("hp-depth", "value"),
    State("hp-leaf",   "value"), State("hp-feat",  "value"),
    State("hp-folds",  "value"), State("hp-target","value"),
    State("hp-split",  "value"), State("hp-umap-n","value"),
    background=True,
    progress=[Output("rf-log","children"), Output("rf-log","style")],
    running=[
        (Output("btn-rf","disabled"), True,  False),
        (Output("btn-rf","children"), "Running…", "▶  Run RF"),
    ],
    prevent_initial_call=True,
)
def run_rf(set_progress, n_clicks,
           fit, p1, p2, p4, crosses, conc, snp, extrap, dose, drugs, cells, lineages,
           n_est, depth, leaf, feat, folds, target, test_split, umap_n):

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold, train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    import warnings; warnings.filterwarnings("ignore")
    import time as _time
    import data as _d

    n_est      = int(n_est      or 100)
    depth      = int(depth      or 0) or None
    leaf       = int(leaf       or 5)
    feat       = float(feat     or 30) / 100.0
    folds      = int(folds      or 1)
    test_split = float(test_split or 20) / 100.0
    umap_n     = int(umap_n     or 500)
    target     = target or "ic50"

    LOG_STYLE_ON = {
        "backgroundColor": C["surface"], "border": f"1px solid {C['border']}",
        "borderRadius": "6px", "padding": "10px 14px", "minHeight": "32px",
        "fontSize": "12px", "color": C["teal"], "fontFamily": "monospace",
        "marginBottom": "12px", "whiteSpace": "pre-wrap", "display": "block",
    }
    _logs = []

    def log(msg, pct=None):
        ts = _time.strftime("%H:%M:%S")
        _logs.append(f"[{ts}]  {msg}")
        bar = ""
        if pct is not None:
            filled = int(pct / 5)
            bar = f"\n         [{'█'*filled}{'░'*(20-filled)}] {pct:.0f}%"
        set_progress(["\n".join(_logs) + bar, LOG_STYLE_ON])

    def _err(msg):
        return (html.Div(msg, style={"color": C["red"], "fontSize": "13px", "padding": "12px"}),
                None, render_cached_runs())

    h           = cfg_hash(fit, p1, p2, p4, crosses, conc, snp, extrap, dose,
                           drugs, cells, lineages, n_est, depth, leaf, feat,
                           folds, target, test_split, umap_n)
    result_path = RF_DIR / f"{h}.json"

    log(f"Searching for cached result  [{h}] …", 2)
    if result_path.exists():
        log("✓  Cache hit — loading result.", 10)
        with open(result_path) as f:
            result = json.load(f)
        log("✓  Done.", 100)
        cached = True
    else:
        log("No cache found — running RF from scratch.", 5)
        log("Applying quality filters…", 8)
        sub = apply_filters(fit, p1, p2, p4, crosses, conc, snp, extrap, dose, drugs, cells, lineages)

        if target == "auc":
            pairs = (sub.groupby(["ModelID", "cpd_name"])["area_under_curve"]
                       .mean().reset_index().dropna()
                       .rename(columns={"area_under_curve": "log10_ic50"}))
        else:
            sub = sub[sub["_crosses_50"] & sub["_valid_math"]].copy()
            _inner = sub["p3_total_decline"] / (0.5 - sub["p4_baseline"]) - 1
            sub["log10_ic50"] = (
                sub["p1_center"]
                + np.log10(_inner.clip(lower=1e-30)) / sub["p2_slope"]
            ) * np.log10(2)
            pairs = (sub.groupby(["ModelID", "cpd_name"])["log10_ic50"]
                        .mean().reset_index().dropna())

        lo, hi = pairs["log10_ic50"].quantile(0.01), pairs["log10_ic50"].quantile(0.99)
        pairs  = pairs[(pairs["log10_ic50"] >= lo) & (pairs["log10_ic50"] <= hi)]
        log(f"✓  {len(pairs):,} (cell-line × drug) pairs after filtering.", 10)

        if len(pairs) < 100:
            return _err("⚠  < 100 pairs after filtering. Relax quality filters.")

        if _d._gene_matrix is None or _d._fp_sub is None:
            log("Loading gene expression features…", 13)
            _ensure_features_loaded()
            log(f"✓  Gene matrix: {_d._gene_matrix.shape[1]} genes | "
                f"Drug FP: {_d._fp_sub.shape[1]-1} features.", 19)
        else:
            log("✓  Gene & drug features already loaded.", 19)

        big = (pairs
               .merge(_d._fp_sub, on="cpd_name", how="inner")
               .merge(_d._gene_matrix.reset_index(), on="ModelID", how="inner")
               .dropna(subset=["log10_ic50"]))
        fcols = [c for c in big.columns if c not in {"ModelID", "cpd_name", "log10_ic50"}]
        X     = np.clip(big[fcols].values.astype(np.float32), -3.4e38, 3.4e38)
        y     = big["log10_ic50"].values
        log(f"Feature matrix: {X.shape[0]:,} × {X.shape[1]:,} features.", 22)

        if len(big) < 50:
            return _err("⚠  Too few pairs after feature join.")

        RF_P          = dict(n_estimators=n_est, max_depth=depth, min_samples_leaf=leaf,
                             max_features=feat, n_jobs=-1, random_state=42)
        all_cells_arr = big["ModelID"].unique()

        if folds == 1:
            log("Training RF (1 train/test split)…", 30)
            tr_c, _ = train_test_split(all_cells_arr, test_size=test_split, random_state=42)
            tr_m    = big["ModelID"].isin(tr_c)
            rf      = RandomForestRegressor(**RF_P)
            rf.fit(X[tr_m], y[tr_m])
            fold_results = [{"fold": 1,
                "n_train":    int(tr_m.sum()),
                "n_test":     int((~tr_m).sum()),
                "r2_train":   round(float(r2_score(y[tr_m],  rf.predict(X[tr_m]))),  4),
                "r2_test":    round(float(r2_score(y[~tr_m], rf.predict(X[~tr_m]))), 4),
                "rmse_train": round(float(np.sqrt(mean_squared_error(y[tr_m],  rf.predict(X[tr_m])))),  4),
                "rmse_test":  round(float(np.sqrt(mean_squared_error(y[~tr_m], rf.predict(X[~tr_m])))), 4),
            }]
            log(f"  Fold 1  R²={fold_results[0]['r2_test']:.4f}  RMSE={fold_results[0]['rmse_test']:.4f}", 55)
        else:
            fold_results = []
            for i, (tri, tei) in enumerate(
                KFold(folds, shuffle=True, random_state=42).split(all_cells_arr), 1
            ):
                fold_pct = 22 + int(50 * i / folds)
                log(f"Training fold {i}/{folds}…", fold_pct)
                tc = set(all_cells_arr[tri])
                tm = big["ModelID"].isin(tc)
                rf = RandomForestRegressor(**RF_P)
                rf.fit(X[tm], y[tm])
                fr_i = {"fold": i,
                    "n_train":    int(tm.sum()),
                    "n_test":     int((~tm).sum()),
                    "r2_train":   round(float(r2_score(y[tm],  rf.predict(X[tm]))),  4),
                    "r2_test":    round(float(r2_score(y[~tm], rf.predict(X[~tm]))), 4),
                    "rmse_train": round(float(np.sqrt(mean_squared_error(y[tm],  rf.predict(X[tm])))),  4),
                    "rmse_test":  round(float(np.sqrt(mean_squared_error(y[~tm], rf.predict(X[~tm])))), 4),
                }
                fold_results.append(fr_i)
                log(f"  ✓ Fold {i}/{folds}  R²={fr_i['r2_test']:.4f}  RMSE={fr_i['rmse_test']:.4f}",
                    fold_pct + int(25 / folds))

        _fi_pairs = sorted(zip(fcols, rf.feature_importances_.tolist()),
                           key=lambda x: x[1], reverse=True)[:50]
        _feature_importances = [{"feature": f, "importance": round(imp, 7)}
                                 for f, imp in _fi_pairs]

        # ── Save full leaf assignments for on-demand re-embedding ─────────────
        log("Saving full leaf assignments for re-embedding…", 78)
        L_full = rf.apply(X).astype(np.int32)
        np.save(RF_DIR / f"{h}_leaves.npy", L_full)
        np.savez(RF_DIR / f"{h}_meta.npz",
            model_ids = big["ModelID"].values,
            drugs     = big["cpd_name"].values,
            ic50      = y,
        )
        log("✓  Leaf assignments saved.", 82)

        # ── Default UMAP embedding on umap_n random samples ───────────────────
        N_EMBED  = min(umap_n, len(big))
        rng      = np.random.default_rng(42)
        idx      = rng.choice(len(big), N_EMBED, replace=False)
        L_sub    = L_full[idx]
        y_sub    = y[idx]
        mid_sub  = big["ModelID"].values[idx]
        drug_sub = big["cpd_name"].values[idx]
        cl_names = [_meta_map.get(m, {}).get("CellLineName", m) for m in mid_sub]
        lin_sub  = [_meta_map.get(m, {}).get("OncotreeLineage", "Unknown") for m in mid_sub]

        log(f"Computing leaf kernel + UMAP on {N_EMBED:,} samples…", 85)
        import umap as _umap
        from kernel import leaf_agreement_kernel
        K_sub       = leaf_agreement_kernel(L_sub, L_sub)
        dist_matrix = (1.0 - K_sub).astype(np.float32)
        reducer     = _umap.UMAP(metric="precomputed", n_neighbors=15,
                                 min_dist=0.1, random_state=42, n_jobs=1)
        emb = reducer.fit_transform(dist_matrix)
        log("✓  UMAP embedding done.", 95)

        tgt_label = "AUC" if target == "auc" else "log₁₀(IC50)"
        result = {
            "config_hash":  h,
            "n_pairs":      len(big),
            "n_features":   len(fcols),
            "rf_params":    {"n_estimators": n_est, "max_depth": depth,
                             "min_samples_leaf": leaf, "max_features": feat},
            "filter_settings": {
                "fit_params": fit, "p1_ci_max": p1, "p2_ci_max": p2, "p4_ci_max": p4,
                "crosses_50": crosses, "min_conc_points": conc, "snp_status": snp,
                "no_extrapolation": extrap, "high_dose_ok": dose,
                "drugs": drugs or "all", "cells": cells or "all", "lineages": lineages or "all",
                "test_split": test_split, "umap_n": umap_n,
            },
            "target":           target,
            "folds":            folds,
            "fold_results":     fold_results,
            "mean_r2_test":     round(float(np.mean([f["r2_test"]   for f in fold_results])), 4),
            "mean_rmse_test":   round(float(np.mean([f["rmse_test"] for f in fold_results])), 4),
            "mean_r2_train":    round(float(np.mean([f["r2_train"]  for f in fold_results])), 4),
            "feature_importances": _feature_importances,
            "embedding": {
                "x": emb[:, 0].tolist(), "y": emb[:, 1].tolist(),
                "ic50":      y_sub.tolist(),
                "cell_line": cl_names,
                "drug":      drug_sub.tolist(),
                "model_id":  mid_sub.tolist(),
                "lineage":   lin_sub,
            },
        }
        log("Saving result to cache…", 97)
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        log("Done.", 100)
        cached = False

    emb       = result.get("embedding", {})
    tgt_label = "AUC" if result.get("target") == "auc" else "log₁₀(IC50)"
    umap_store = ({**emb, "tgt_label": tgt_label,
                   "feature_importances": result.get("feature_importances", []),
                   "config_hash": h}
                  if emb else None)

    return render_rf_metrics(result, cached, current_h=h), umap_store, render_cached_runs(h)


@callback(
    Output("rf-history-scatter", "figure"),
    Input("hist-metric",     "value"),
    Input("umap-data-store", "data"),
    Input("delete-trigger",  "data"),
)
def update_history_scatter(metric, _umap_store, _del_trigger):
    rows = []
    for jf in sorted(RF_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime):
        try:
            with open(jf) as f:
                r = json.load(f)
            rp = r.get("rf_params", {})
            rows.append({
                "hash":       r.get("config_hash", jf.stem[:12]),
                "target":     r.get("target", "ic50"),
                "n_pairs":    r.get("n_pairs", 0),
                "r2_test":    r.get("mean_r2_test"),
                "rmse_test":  r.get("mean_rmse_test"),
                "n_est":      rp.get("n_estimators", "—"),
                "depth":      rp.get("max_depth") or "None",
                "leaf":       rp.get("min_samples_leaf", "—"),
                "feat":       rp.get("max_features", "—"),
                "folds":      r.get("folds", "—"),
                "complexity": model_complexity(rp),
            })
        except Exception:
            continue

    if not rows:
        fig = go.Figure()
        fig.update_layout(**PLOT_L, height=260,
                          title="No cached runs yet — click ▶ Run RF to populate")
        return fig

    y_col   = "r2_test"  if metric == "r2" else "rmse_test"
    y_title = "Test R²"  if metric == "r2" else "Test RMSE"
    complexities = [r["complexity"] for r in rows]
    cmin_val     = min(complexities)
    cmax_val     = max(complexities)
    if cmin_val == cmax_val:
        cmin_val = max(0.0, cmin_val - 0.05)
        cmax_val = min(1.0, cmax_val + 0.05)

    def _hover(r):
        tl   = "AUC" if r["target"] == "auc" else "IC50"
        r2   = f"{r['r2_test']:.4f}"   if isinstance(r["r2_test"],   float) else "—"
        rmse = f"{r['rmse_test']:.4f}" if isinstance(r["rmse_test"], float) else "—"
        return (f"<b>{r['hash']}</b>  [{tl}]<br>"
                f"n_pairs: {r['n_pairs']:,}<br>"
                f"n_est={r['n_est']}  depth={r['depth']}  leaf={r['leaf']}  feat={r['feat']}<br>"
                f"folds: {r['folds']}<br>R²: {r2}   RMSE: {rmse}")

    colorbar_kw = dict(
        colorscale=[[0, C["teal"]], [0.5, C["purple"]], [1, C["orange"]]],
        cmin=cmin_val, cmax=cmax_val,
        colorbar=dict(
            title=dict(text="Complexity", font=dict(color=C["text"], size=10)),
            tickfont=dict(color=C["text"]), len=0.55, thickness=12,
            y=0.28, yanchor="middle",
        ),
    )
    fig = go.Figure()
    for tgt, symbol, label in [("ic50", "circle", "log₁₀(IC50)"),
                                ("auc",  "triangle-up", "AUC")]:
        sub = [r for r in rows if r["target"] == tgt]
        if not sub:
            continue
        fig.add_trace(go.Scatter(
            x=[r["n_pairs"] for r in sub],
            y=[r[y_col]     for r in sub],
            mode="markers", name=label,
            marker=dict(**colorbar_kw, color=[r["complexity"] for r in sub],
                        symbol=symbol, size=12, opacity=0.88,
                        line=dict(width=1, color=C["border"]),
                        showscale=(tgt == "ic50")),
            text=[_hover(r) for r in sub],
            hovertemplate="%{text}<extra></extra>",
            customdata=[r["hash"] for r in sub],
        ))
    base_layout = {k: v for k, v in PLOT_L.items() if k != "legend"}
    fig.update_layout(
        **base_layout,
        title=f"RF Run History  —  {y_title} vs Samples  (● IC50 · ▲ AUC · color = complexity)",
        xaxis_title="Number of Samples (n_pairs)",
        yaxis_title=y_title, height=260, clickmode="event",
        legend=dict(bgcolor="rgba(13,13,20,0.85)", font=dict(color=C["text"], size=10),
                    x=1.02, y=0.98, xanchor="left", yanchor="top",
                    bordercolor=C["border"], borderwidth=1),
    )
    return fig


@callback(
    Output("rf-results",      "children", allow_duplicate=True),
    Output("umap-data-store", "data",    allow_duplicate=True),
    Input("rf-history-scatter", "clickData"),
    prevent_initial_call=True,
)
def show_clicked_run(click_data):
    if not click_data:
        raise PreventUpdate
    h           = click_data["points"][0]["customdata"]
    result_path = RF_DIR / f"{h}.json"
    if not result_path.exists():
        return (html.Div(f"Cache file not found: {h}",
                         style={"color": C["red"], "fontSize": "12px", "padding": "8px"}),
                None)
    with open(result_path) as f:
        result = json.load(f)
    emb        = result.get("embedding", {})
    tgt_label  = "AUC" if result.get("target") == "auc" else "log₁₀(IC50)"
    umap_store = ({**emb, "tgt_label": tgt_label,
                   "feature_importances": result.get("feature_importances", []),
                   "config_hash": h}
                  if emb else None)
    panel = html.Div(
        style={"border": f"2px solid {C['purple']}", "borderRadius": "8px",
               "padding": "14px", "marginTop": "10px"},
        children=[
            html.Div(f"Inspecting  {h}",
                     style={"color": C["teal"], "fontFamily": "monospace",
                            "fontSize": "12px", "fontWeight": "bold", "marginBottom": "12px"}),
            render_rf_result(result, cached=True, current_h=h),
        ],
    )
    return panel, umap_store


@callback(
    Output("umap-drug-ctrl",      "style"),
    Output("umap-lineage-ctrl",   "style"),
    Output("umap-drug-filter",    "options"),
    Output("umap-lineage-filter", "options"),
    Input("umap-view-tabs",  "value"),
    Input("umap-data-store", "data"),
)
def update_umap_controls(tab, emb_data):
    show = {"display": "block", "marginTop": "10px"}
    hide = {"display": "none"}
    if not emb_data:
        return hide, hide, [], []

    # Try to load full counts from meta file
    h         = emb_data.get("config_hash")
    meta_path = RF_DIR / f"{h}_meta.npz"
    if meta_path.exists():
        meta      = np.load(meta_path, allow_pickle=True)
        all_drugs = meta["drugs"].tolist()
        all_lins  = [_meta_map.get(m, {}).get("OncotreeLineage", "Unknown")
                     for m in meta["model_ids"]]
        from collections import Counter
        drug_counts = Counter(all_drugs)
        lin_counts  = Counter(all_lins)
        drug_opts = [{"label": f"{d}  (n={drug_counts[d]:,})", "value": d}
                     for d in sorted(drug_counts)]
        lin_opts  = [{"label": f"{l}  (n={lin_counts[l]:,})", "value": l}
                     for l in sorted(lin_counts) if l not in ("Unknown", "")]
    else:
        # fallback to embedding sample counts
        drug_opts = [{"label": d, "value": d}
                     for d in sorted(set(emb_data.get("drug", [])))]
        lin_opts  = [{"label": l, "value": l}
                     for l in sorted(set(emb_data.get("lineage", [])) - {"Unknown", ""})]

    if tab == "drug":    return show, hide, drug_opts, lin_opts
    if tab == "lineage": return hide, show, drug_opts, lin_opts
    return hide, hide, drug_opts, lin_opts

@callback(
    Output("umap-drug-filter",    "value"),
    Output("umap-lineage-filter", "value"),
    Output("umap-view-tabs",      "value"),
    Input("btn-reset-umap", "n_clicks"),
    prevent_initial_call=True,
)
def reset_umap_view(_):
    return [], [], "default"

@callback(
    Output("umap-graph", "figure"),
    Input("umap-data-store", "data"),
    Input("umap-view-tabs",  "value"),
)
def update_umap_figure(emb_data, tab):
    if not emb_data:
        fig = go.Figure()
        umap_base = {k: v for k, v in PLOT_L.items() if k not in ("xaxis", "yaxis")}
        fig.update_layout(
            **umap_base, height=500,
            xaxis=dict(**PLOT_L["xaxis"], showticklabels=False),
            yaxis=dict(**PLOT_L["yaxis"], showticklabels=False),
            annotations=[dict(text="Run RF to generate the leaf-agreement embedding",
                              xref="paper", yref="paper", x=0.5, y=0.5,
                              showarrow=False, font=dict(color=C["muted"], size=13))],
        )
        return fig
    ex        = np.array(emb_data["x"])
    ey        = np.array(emb_data["y"])
    ic        = np.array(emb_data["ic50"])
    cl        = emb_data["cell_line"]
    dr        = emb_data["drug"]
    tgt_label = emb_data.get("tgt_label", "Response")
    fig       = _build_umap_default(ex, ey, ic, cl, dr, tgt_label)
    return _apply_umap_layout(fig, "Leaf-Agreement UMAP", len(ex))


@callback(
    Output("umap-graph", "figure", allow_duplicate=True),
    Input("umap-drug-filter",    "value"),
    Input("umap-lineage-filter", "value"),
    State("umap-data-store",     "data"),
    State("umap-view-tabs",      "value"),
    prevent_initial_call=True,
)
def reembed_filtered(sel_drugs, sel_lineages, emb_data, tab):
    """Re-embed UMAP using all pairs for the selected drug or lineage."""
    if not emb_data:
        raise PreventUpdate
    
    if tab == "drug" and not sel_drugs:
        # selection cleared — fall back to default view
        if not emb_data:
            raise PreventUpdate
        ex = np.array(emb_data["x"])
        ey = np.array(emb_data["y"])
        ic = np.array(emb_data["ic50"])
        fig = _build_umap_default(ex, ey, ic, emb_data["cell_line"],
                                emb_data["drug"], emb_data.get("tgt_label", "Response"))
        return _apply_umap_layout(fig, "Leaf-Agreement UMAP", len(ex))

    if tab == "lineage" and not sel_lineages:
        if not emb_data:
            raise PreventUpdate
        ex = np.array(emb_data["x"])
        ey = np.array(emb_data["y"])
        ic = np.array(emb_data["ic50"])
        fig = _build_umap_default(ex, ey, ic, emb_data["cell_line"],
                                emb_data["drug"], emb_data.get("tgt_label", "Response"))
        return _apply_umap_layout(fig, "Leaf-Agreement UMAP", len(ex))

    h           = emb_data.get("config_hash")
    leaves_path = RF_DIR / f"{h}_leaves.npy"
    meta_path   = RF_DIR / f"{h}_meta.npz"
    if not leaves_path.exists() or not meta_path.exists():
        raise PreventUpdate

    L_full    = np.load(leaves_path)
    meta      = np.load(meta_path, allow_pickle=True)
    model_ids = meta["model_ids"]
    drugs     = meta["drugs"]
    ic50      = meta["ic50"]
    lineages  = np.array([_meta_map.get(m, {}).get("OncotreeLineage", "Unknown")
                          for m in model_ids])
    cl_names  = np.array([_meta_map.get(m, {}).get("CellLineName", m)
                          for m in model_ids])

    if tab == "drug" and sel_drugs:
        mask  = np.array([d in sel_drugs for d in drugs])
        title = f"Pharmacological View — {', '.join(sel_drugs[:3])}"
    else:
        mask  = np.array([l in sel_lineages for l in lineages])
        title = f"Tissue View — {', '.join(sel_lineages[:3])}"

    if mask.sum() < 3:
        raise PreventUpdate

    idx = np.where(mask)[0]
    if len(idx) > 3000:
        rng = np.random.default_rng(42)
        idx = rng.choice(idx, 3000, replace=False)

    L_sub     = L_full[idx]
    y_sub     = ic50[idx]
    cl_sub    = cl_names[idx].tolist()
    dr_sub    = drugs[idx].tolist()
    lin_sub   = lineages[idx].tolist()
    tgt_label = emb_data.get("tgt_label", "log₁₀(IC50)")

    from kernel import leaf_agreement_kernel
    import umap as _umap
    K_sub       = leaf_agreement_kernel(L_sub, L_sub)
    dist_matrix = (1.0 - K_sub).astype(np.float32)
    reducer     = _umap.UMAP(metric="precomputed",
                             n_neighbors=min(15, len(idx) - 1),
                             min_dist=0.1, random_state=42, n_jobs=1)
    emb = reducer.fit_transform(dist_matrix)
    ex  = emb[:, 0]
    ey  = emb[:, 1]

    # Color all points by IC50, but label by drug/lineage in hover
    from umap_builder import _UMAP_COLORSCALE
    import plotly.graph_objects as go

    if tab == "drug":
        # one trace per drug colored by IC50
        traces = []
        sel_set = set(sel_drugs)
        for drug in sorted(sel_set):
            d_idx = [i for i, d in enumerate(dr_sub) if d == drug]
            if not d_idx:
                continue
            info   = _meta_map  # just for hover
            traces.append(go.Scatter(
                x=[ex[i] for i in d_idx],
                y=[ey[i] for i in d_idx],
                mode="markers",
                name=drug,
                marker=dict(
                    color=[float(y_sub[i]) for i in d_idx],
                    colorscale=_UMAP_COLORSCALE,
                    size=6, opacity=0.85,
                    showscale=(drug == sorted(sel_set)[0]),
                    colorbar=dict(
                        title=dict(text=tgt_label, font=dict(color=C["text"], size=11)),
                        tickfont=dict(color=C["text"]), len=0.7,
                    ) if drug == sorted(sel_set)[0] else {},
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>Drug: %{customdata[1]}<br>"
                    + tgt_label + ": %{customdata[2]:.3f}<extra></extra>"
                ),
                customdata=[[cl_sub[i], dr_sub[i], float(y_sub[i]), lin_sub[i]] for i in d_idx],
            ))
        fig = go.Figure(traces)
    else:
        # one trace per lineage colored by IC50
        traces = []
        sel_set = set(sel_lineages)
        lin_colors = {}
        from umap_builder import _LINEAGE_PALETTE
        for i, l in enumerate(sorted(sel_set)):
            lin_colors[l] = _LINEAGE_PALETTE[i % len(_LINEAGE_PALETTE)]

        for lin in sorted(sel_set):
            l_idx = [i for i, l in enumerate(lin_sub) if l == lin]
            if not l_idx:
                continue
            traces.append(go.Scatter(
                x=[ex[i] for i in l_idx],
                y=[ey[i] for i in l_idx],
                mode="markers",
                name=lin,
                marker=dict(
                    color=[float(y_sub[i]) for i in l_idx],
                    colorscale=_UMAP_COLORSCALE,
                    size=6, opacity=0.85,
                    showscale=(lin == sorted(sel_set)[0]),
                    colorbar=dict(
                        title=dict(text=tgt_label, font=dict(color=C["text"], size=11)),
                        tickfont=dict(color=C["text"]), len=0.7,
                    ) if lin == sorted(sel_set)[0] else {},
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>Drug: %{customdata[1]}<br>"
                    "Lineage: " + lin + "<br>"
                    + tgt_label + ": %{customdata[2]:.3f}<extra></extra>"
                ),
                customdata=[[cl_sub[i], dr_sub[i], float(y_sub[i]), lin_sub[i]] for i in l_idx],
            ))
        fig = go.Figure(traces)

    return _apply_umap_layout(fig, title, len(idx))


@callback(
    Output("btn-llm-pair", "disabled"),
    Output("btn-llm-pair", "style"),
    Input("ai-drug-select", "value"),
    Input("ai-cell-select", "value"),
)
def toggle_pair_btn(drug, cell_line):
    _base = {"backgroundColor": "transparent", "color": C["teal"],
             "border": f"1px solid {C['teal']}", "borderRadius": "5px",
             "padding": "5px 14px", "fontSize": "11px", "fontFamily": "monospace"}
    if drug and cell_line:
        return False, {**_base, "cursor": "pointer", "opacity": "1"}
    return True, {**_base, "cursor": "not-allowed", "opacity": "0.35"}


@callback(
    Output("llm-sample-output",   "children"),
    Output("level1-prompt-store", "data"),
    Input("btn-llm-pair", "n_clicks"),
    State("ai-drug-select",  "value"),
    State("ai-cell-select",  "value"),
    State("umap-data-store", "data"),
    prevent_initial_call=True,
)
def llm_pair(_n_clicks, drug, cell_line, emb_data):
    if not drug or not cell_line:
        return no_update, no_update
    evidence  = _get_gene_evidence([cell_line], drug=drug, top_n=3)
    prompt    = _build_pair_prompt(drug, cell_line, evidence)
    text      = _call_gemini(prompt)
    cl_name   = _meta_map.get(cell_line, {}).get("CellLineName", cell_line)
    top_feats = sorted(
        [{"feature": e["feature"], "importance": abs(e.get("_z", 0))}
         for e in evidence if "_z" in e],
        key=lambda x: x["importance"], reverse=True,
    )
    if not top_feats:
        top_feats = (emb_data or {}).get("feature_importances", [])[:8]
    return (
        _llm_output_with_evidence(
            f"Level 1 — Sample: {cl_name} × {drug}", text, C["teal"], evidence, top_feats,
            feat_label="Top Features — Local z-score (vs global)"),
        prompt,
    )


@callback(
    Output("llm-sample-output",   "children",  allow_duplicate=True),
    Output("level1-prompt-store", "data",       allow_duplicate=True),
    Input("umap-graph",    "clickData"),
    State("umap-data-store", "data"),
    prevent_initial_call=True,
)
def llm_point(click_data, emb_data):
    if not click_data or not emb_data:
        return no_update, no_update
    pt  = click_data["points"][0]
    ex  = np.array(emb_data["x"])
    ey  = np.array(emb_data["y"])
    idx = int(np.argmin((ex - pt["x"]) ** 2 + (ey - pt["y"]) ** 2))
    dr        = emb_data["drug"][idx]
    cl        = emb_data["cell_line"][idx]
    evidence  = _get_neighbor_evidence(idx, emb_data, top_n=3)
    prompt    = _build_point_prompt(idx, emb_data, evidence)
    text      = _call_gemini(prompt)
    top_feats = sorted(
        [{"feature": e["feature"], "importance": abs(e.get("_z", 0))}
         for e in evidence if "_z" in e],
        key=lambda x: x["importance"], reverse=True,
    )
    if not top_feats:
        top_feats = emb_data.get("feature_importances", [])[:8]
    return (
        _llm_output_with_evidence(
            f"Level 1 — Sample: {cl} × {dr}", text, C["teal"], evidence, top_feats,
            feat_label="Top Features — Local z-score (vs neighbors)"),
        prompt,
    )


@callback(
    Output("llm-cluster-output",  "children"),
    Output("level2-prompt-store", "data"),
    Input("umap-graph",    "selectedData"),
    State("umap-data-store", "data"),
    prevent_initial_call=True,
)
def llm_cluster(selected_data, emb_data):
    if not selected_data or not emb_data:
        return no_update, no_update
    points = selected_data.get("points", [])
    if len(points) < 3:
        return no_update, no_update
    evidence  = _get_cluster_evidence(points, emb_data, top_n=3)
    tgt_label = emb_data.get("tgt_label", "Response")
    prompt    = _build_cluster_prompt(points, tgt_label, evidence)
    text      = _call_gemini(prompt)
    top_feats = sorted(
        [{"feature": e["feature"], "importance": abs(e.get("_z", 0))}
         for e in evidence if "_z" in e],
        key=lambda x: x["importance"], reverse=True,
    )
    if not top_feats:
        top_feats = emb_data.get("feature_importances", [])[:8]
    return (
        _llm_output_with_evidence(
            f"Level 2 — Cluster  ({len(points)} points selected)",
            text, C["purple"], evidence, top_feats,
            feat_label="Top Features — Local z-score (vs global)"),
        prompt,
    )


@callback(
    Output("llm-global-output", "children"),
    Input("umap-data-store", "data"),
    background=True,
    prevent_initial_call=True,
)
def llm_global(emb_data):
    if not emb_data:
        return no_update
    prompt    = _build_global_prompt(emb_data)
    text      = _call_gemini(prompt)
    top_feats = emb_data.get("feature_importances", [])[:8]
    return _llm_output_with_evidence(
        "Level 3 — Global UMAP", text, C["orange"], [], top_feats,
        feat_label="Top Features — RF Weighted Impurity Decrease")


@callback(
    Output("pending-delete-hash", "data"),
    Output("confirm-delete",      "displayed"),
    Input({"type": "del-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def store_pending_delete(n_clicks_list):
    if not any(n or 0 for n in n_clicks_list):
        raise PreventUpdate
    tid = ctx.triggered_id
    if isinstance(tid, dict) and tid.get("type") == "del-btn":
        return tid["index"], True
    raise PreventUpdate


@callback(
    Output("cached-runs-section", "children", allow_duplicate=True),
    Output("delete-trigger",      "data"),
    Input("confirm-delete",  "submit_n_clicks"),
    State("pending-delete-hash", "data"),
    State("delete-trigger",      "data"),
    prevent_initial_call=True,
)
def execute_delete(submit_clicks, hash_to_delete, trigger_val):
    if not submit_clicks or not hash_to_delete:
        raise PreventUpdate
    for ext in [".json", "_leaves.npy", "_meta.npz"]:
        p = RF_DIR / f"{hash_to_delete}{ext}"
        if p.exists():
            p.unlink()
    return render_cached_runs(current_h=None), (trigger_val or 0) + 1


@callback(
    Output("rf-results",          "children",  allow_duplicate=True),
    Output("umap-data-store",     "data",      allow_duplicate=True),
    Output("cached-runs-section", "children",  allow_duplicate=True),
    Input({"type": "load-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def load_cached_run(n_clicks_list):
    if not any(n or 0 for n in n_clicks_list):
        raise PreventUpdate
    tid = ctx.triggered_id
    if not (isinstance(tid, dict) and tid.get("type") == "load-btn"):
        raise PreventUpdate
    h           = tid["index"]
    result_path = RF_DIR / f"{h}.json"
    if not result_path.exists():
        raise PreventUpdate
    with open(result_path) as f:
        result = json.load(f)
    emb        = result.get("embedding", {})
    tgt_label  = "AUC" if result.get("target") == "auc" else "log₁₀(IC50)"
    umap_store = ({**emb, "tgt_label": tgt_label,
                   "feature_importances": result.get("feature_importances", []),
                   "config_hash": h}
                  if emb else None)
    return (
        render_rf_metrics(result, cached=True, current_h=h),
        umap_store,
        render_cached_runs(current_h=h),
    )


if __name__ == "__main__":
    print("Dashboard → http://127.0.0.1:8050")
    app.run(debug=False)