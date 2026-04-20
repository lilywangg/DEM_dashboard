"""
renders.py — UI rendering helpers, style constants, and Dash component builders
"""

from __future__ import annotations
from pathlib import Path

import json
from dash import html

from data import RF_DIR

# ── Colour palette ─────────────────────────────────────────────────────────────
C = dict(
    bg="#000000", panel="#0f0f17", card="#0d0d14", surface="#1a1a2e",
    border="#2a2a45", muted="#8888aa", text="#e0e0ff",
    blue="#60aaff", purple="#c084fc", green="#4ade80",
    yellow="#fde68a", orange="#fb923c", red="#f87171",
    teal="#2dd4bf", pink="#f472b6",
)

SECTION_GREY        = "#e0e0ff"
FILTER_SECTION_COLOR = "#9999bb"

# ── Shared style dicts ─────────────────────────────────────────────────────────
LABEL_S = {
    "color": "#555570", "fontWeight": "bold", "fontSize": "11px",
    "marginBottom": "3px", "marginTop": "13px", "letterSpacing": "0.5px",
}
SECTION_S = {
    "color": FILTER_SECTION_COLOR, "fontSize": "11px", "fontWeight": "bold",
    "letterSpacing": "1.5px", "marginBottom": "8px", "marginTop": "6px",
    "borderBottom": f"1px solid {C['border']}", "paddingBottom": "5px",
}
CARD_S = {
    "backgroundColor": C["card"], "padding": "16px",
    "borderRadius": "8px", "marginBottom": "14px",
    "border": f"1px solid {C['border']}",
}
PLOT_L = dict(
    paper_bgcolor=C["card"], plot_bgcolor=C["card"],
    font=dict(color=C["text"], size=11),
    margin=dict(l=50, r=20, t=36, b=36),
    xaxis=dict(
        gridcolor=C["border"], zerolinecolor=C["border"],
        color=C["muted"], title_font=dict(color=C["text"]),
    ),
    yaxis=dict(
        gridcolor=C["border"], zerolinecolor=C["border"],
        color=C["muted"], title_font=dict(color=C["text"]),
    ),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["text"], size=10)),
)


def dd_style() -> dict:
    return {
        "backgroundColor": C["surface"], "color": C["text"],
        "border": f"1px solid {C['border']}", "fontSize": "12px",
    }


def inp_style(w: str = "88px") -> dict:
    return {
        "backgroundColor": C["surface"], "color": C["text"],
        "border": f"1px solid {C['border']}", "borderRadius": "4px",
        "fontSize": "12px", "width": w, "padding": "4px 8px",
    }


def stat_box(label: str, value: str, color: str) -> html.Div:
    """Single stat card used in the data overview row."""
    return html.Div(
        style={
            "backgroundColor": C["surface"], "borderRadius": "6px",
            "padding": "10px 14px", "textAlign": "center", "flex": "1",
            "border": f"1px solid {C['border']}",
        },
        children=[
            html.Div(value, style={"color": color, "fontWeight": "bold", "fontSize": "18px"}),
            html.Div(label, style={"color": C["muted"], "fontSize": "10px", "marginTop": "2px"}),
        ],
    )


def model_complexity(rp: dict) -> float:
    """0–1 complexity score derived from RF hyperparameters."""
    n_est = rp.get("n_estimators", 100)
    depth = rp.get("max_depth") or 50
    leaf  = rp.get("min_samples_leaf", 5)
    feat  = rp.get("max_features", 0.3)
    import numpy as np
    c_est  = float(np.clip((n_est - 10) / 490, 0, 1))
    c_dep  = float(np.clip((depth - 1) / 49, 0, 1))
    c_leaf = float(1 - np.clip((leaf - 1) / 49, 0, 1))
    c_feat = float(np.clip((feat - 0.05) / 0.95, 0, 1))
    return round(0.30 * c_est + 0.30 * c_dep + 0.20 * c_leaf + 0.20 * c_feat, 4)


# ── LLM output renderer ────────────────────────────────────────────────────────
def _llm_output_with_evidence(
    level_label: str,
    text: str,
    color: str,
    evidence: list[dict],
    top_features: list[dict] | None = None,
    feat_label: str = "Top Features — Weighted Impurity Decrease",
) -> html.Div:
    """
    Render LLM analysis output with:
      - Groundedness badge (citation count)
      - Top features bar chart (z-scores or RF importances)
      - Evidence table (gene, drug FP, drug target rows)
    """
    from llm import _count_citations

    th_s = {
        "padding": "4px 8px", "backgroundColor": C["card"], "color": color,
        "fontWeight": "bold", "fontSize": "10px",
        "border": f"1px solid {C['border']}", "fontFamily": "monospace",
    }
    td_s = {
        "padding": "4px 8px", "fontSize": "11px", "color": C["text"],
        "borderBottom": f"1px solid {C['border']}", "fontFamily": "monospace",
    }

    # ── Groundedness badge ─────────────────────────────────────────────────────
    n_cit       = _count_citations(text)
    badge_color = C["green"] if n_cit >= 1 else C["orange"]
    badge_text  = f"Grounded ({n_cit} citations)" if n_cit >= 1 else "Ungrounded"
    ground_badge = html.Span(badge_text, style={
        "backgroundColor": badge_color, "color": "#000",
        "fontSize": "9px", "fontWeight": "bold", "borderRadius": "3px",
        "padding": "1px 6px", "marginLeft": "8px",
        "fontFamily": "monospace", "verticalAlign": "middle",
    })

    # ── Top features bar chart ─────────────────────────────────────────────────
    feat_section = []
    if top_features:
        max_imp = top_features[0]["importance"] if top_features else 1.0
        bars = []
        for ft in top_features[:8]:
            pct = int(100 * ft["importance"] / max(max_imp, 1e-9))
            bars.append(html.Div(
                style={"display": "flex", "alignItems": "center",
                       "gap": "6px", "marginBottom": "3px"},
                children=[
                    html.Span(ft["feature"], style={
                        "color": C["text"], "fontSize": "10px",
                        "fontFamily": "monospace", "width": "120px",
                        "flexShrink": "0", "overflow": "hidden",
                        "textOverflow": "ellipsis", "whiteSpace": "nowrap",
                    }),
                    html.Div(
                        style={"flex": "1", "height": "8px",
                               "backgroundColor": C["card"],
                               "borderRadius": "2px", "overflow": "hidden"},
                        children=[html.Div(style={
                            "width": f"{pct}%", "height": "100%",
                            "backgroundColor": color, "borderRadius": "2px",
                        })],
                    ),
                    html.Span(f"{ft['importance']:.4f}", style={
                        "color": C["muted"], "fontSize": "9px",
                        "fontFamily": "monospace", "width": "52px",
                        "textAlign": "right",
                    }),
                ],
            ))
        feat_section = [
            html.Div(feat_label, style={
                "color": color, "fontSize": "10px", "fontWeight": "bold",
                "marginTop": "10px", "marginBottom": "4px",
                "fontFamily": "monospace", "letterSpacing": "0.5px",
            }),
            *bars,
        ]

    # ── Evidence table ─────────────────────────────────────────────────────────
    dir_color = {"High": C["red"], "Low": C["green"], "Target": C["purple"]}
    modality_color = {
        "Gene":        C["blue"],
        "Drug FP":     C["orange"],
        "Drug Target": C["purple"],
    }
    has_modality = any("modality" in e for e in evidence)
    ev_rows = []
    for e in evidence:
        dc  = dir_color.get(e.get("direction", ""), C["text"])
        mod = e.get("modality", "")
        mc  = modality_color.get(mod, C["muted"])
        row_cells = [
            html.Td(e["id"],      style={**td_s, "color": color, "fontWeight": "bold"}),
            html.Td(e["feature"], style={**td_s, "color": C["text"]}),
            html.Td(e["value"],   style={**td_s, "color": dc}),
            html.Td(e["stat"],    style={**td_s, "color": C["muted"]}),
            html.Td(e["source"],  style={**td_s, "color": C["muted"]}),
        ]
        if has_modality:
            row_cells.insert(1, html.Td(mod, style={**td_s, "color": mc, "fontWeight": "bold"}))
        ev_rows.append(html.Tr(row_cells))

    ev_section = []
    if evidence:
        header_cols = ["ID", "Feature", "Value", "Stat", "Source"]
        if has_modality:
            header_cols.insert(1, "Modality")
        ev_section = [
            html.Div("Evidence Table", style={
                "color": color, "fontSize": "10px", "fontWeight": "bold",
                "marginTop": "10px", "marginBottom": "4px",
                "fontFamily": "monospace", "letterSpacing": "0.5px",
            }),
            html.Div(style={"overflowX": "auto"}, children=[
                html.Table(
                    style={"width": "100%", "borderCollapse": "collapse"},
                    children=[
                        html.Thead(html.Tr([html.Th(c, style=th_s) for c in header_cols])),
                        html.Tbody(ev_rows),
                    ],
                )
            ]),
        ]

    return html.Div(
        style={
            "backgroundColor": C["surface"], "borderRadius": "6px",
            "padding": "12px", "border": f"1px solid {color}",
            "marginBottom": "8px",
        },
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "center", "marginBottom": "6px"},
                children=[
                    html.Span(level_label, style={
                        "color": color, "fontWeight": "bold",
                        "fontSize": "11px", "fontFamily": "monospace",
                    }),
                    ground_badge,
                ],
            ),
            html.Div(text, style={
                "color": C["text"], "fontSize": "12px",
                "lineHeight": "1.65", "whiteSpace": "pre-wrap",
            }),
            *feat_section,
            *ev_section,
        ],
    )


# ── RF metrics panel ───────────────────────────────────────────────────────────
def render_rf_metrics(result: dict, cached: bool, current_h: str | None = None) -> html.Div:
    """Render RF result: badge, stat boxes, fold table, and run parameters."""
    fr        = result["fold_results"]
    tgt       = result.get("target", "ic50")
    tgt_label = "AUC" if tgt == "auc" else "log₁₀(IC50)"
    h         = result["config_hash"]

    badge = html.Span(
        f"  {'✓ from cache' if cached else '✓ saved'}  [{h}]",
        style={"color": C["teal"], "fontSize": "11px"},
    )

    cards = html.Div(
        style={"display": "flex", "gap": "10px", "marginBottom": "14px"},
        children=[
            stat_box("Test R²",    f"{result['mean_r2_test']:.4f}",   C["green"]),
            stat_box("Test RMSE",  f"{result['mean_rmse_test']:.4f}", C["orange"]),
            stat_box("Train R²",   f"{result['mean_r2_train']:.4f}",  C["blue"]),
            stat_box("Pairs",      f"{result['n_pairs']:,}",           C["purple"]),
            stat_box("Features",   f"{result['n_features']:,}",        C["teal"]),
        ],
    )

    th_s = {
        "padding": "6px 12px", "backgroundColor": C["surface"],
        "color": C["blue"], "fontSize": "11px",
        "border": f"1px solid {C['border']}",
    }
    td_s = {
        "padding": "5px 12px", "fontSize": "12px", "color": C["text"],
        "borderBottom": f"1px solid {C['border']}",
    }
    fold_table = html.Table(
        style={"width": "100%", "borderCollapse": "collapse"},
        children=[
            html.Thead(html.Tr([html.Th(c, style=th_s) for c in
                ["Fold", "n_train", "n_test", "R² train", "R² test",
                 "RMSE train", "RMSE test"]])),
            html.Tbody([html.Tr([
                html.Td(f["fold"],              style=td_s),
                html.Td(f"{f['n_train']:,}",    style=td_s),
                html.Td(f"{f['n_test']:,}",     style=td_s),
                html.Td(f"{f['r2_train']:.4f}", style=td_s),
                html.Td(f"{f['r2_test']:.4f}",  style={
                    **td_s, "color": (
                        C["green"]  if f["r2_test"] > 0.7 else
                        C["yellow"] if f["r2_test"] > 0.4 else C["red"]
                    ),
                }),
                html.Td(f"{f['rmse_train']:.4f}", style=td_s),
                html.Td(f"{f['rmse_test']:.4f}",  style=td_s),
            ]) for f in fr]),
        ],
    )

    rp = result["rf_params"]
    fs = result.get("filter_settings", {})
    param_rows = [
        ("target",           tgt_label),
        ("n_estimators",     str(rp.get("n_estimators", "—"))),
        ("max_depth",        str(rp.get("max_depth") or "None")),
        ("min_samples_leaf", str(rp.get("min_samples_leaf", "—"))),
        ("max_features",     str(rp.get("max_features", "—"))),
        ("cv_folds",         str(result.get("folds", "—"))),
        ("fit_params",       str(fs.get("fit_params", "—"))),
        ("p1_ci_max",        str(fs.get("p1_ci_max", "—"))),
        ("p2_ci_max",        str(fs.get("p2_ci_max", "—"))),
        ("p4_ci_max",        str(fs.get("p4_ci_max", "—"))),
        ("crosses_50",       str(fs.get("crosses_50", "—"))),
        ("min_conc_points",  str(fs.get("min_conc_points", "—"))),
        ("no_extrapolation", str(fs.get("no_extrapolation", "—"))),
        ("high_dose_ok",     str(fs.get("high_dose_ok", "—"))),
        ("drugs",            str(fs.get("drugs", "all"))),
        ("lineages",         str(fs.get("lineages", "all"))),
    ]
    pt_s = {"padding": "3px 10px", "fontSize": "11px", "color": C["text"],
            "borderBottom": f"1px solid {C['border']}"}
    pk_s = {**pt_s, "color": C["muted"]}
    param_table = html.Details(
        style={"marginTop": "14px"},
        children=[
            html.Summary("Run Parameters", style={
                "color": C["blue"], "cursor": "pointer",
                "fontSize": "12px", "fontWeight": "bold",
            }),
            html.Table(
                style={"width": "100%", "borderCollapse": "collapse", "marginTop": "6px"},
                children=[html.Tbody([
                    html.Tr([html.Td(k, style=pk_s), html.Td(v, style=pt_s)])
                    for k, v in param_rows
                ])],
            ),
        ],
    )

    return html.Div([
        html.Div(
            style={"display": "flex", "alignItems": "center", "marginBottom": "12px"},
            children=[
                html.Span("Results", style={
                    "color": C["purple"], "fontWeight": "bold", "fontSize": "14px",
                }),
                badge,
            ],
        ),
        cards,
        fold_table,
        param_table,
    ])


def render_rf_result(result: dict, cached: bool, current_h: str | None = None) -> html.Div:
    """Alias for render_rf_metrics — used in the clicked-run inspection panel."""
    return render_rf_metrics(result, cached, current_h)


# ── Cached runs card ───────────────────────────────────────────────────────────
def render_cached_runs(current_h: str | None = None) -> html.Div:
    """
    Render the Cached Runs card showing:
      - filter_comparison baseline results (from preprocessing)
      - all saved RF runs with Load and Delete buttons
    """

    # Load all cached RF run JSON files
    history_rows = []
    for jf in RF_DIR.glob("*.json"):
        try:
            with open(jf) as f:
                r = json.load(f)
            rp = r.get("rf_params", {})
            history_rows.append({
                "hash":    r.get("config_hash", jf.stem[:12]),
                "target":  r.get("target", "ic50"),
                "pairs":   r.get("n_pairs", "—"),
                "feats":   r.get("n_features", "—"),
                "n_est":   rp.get("n_estimators", "—"),
                "depth":   str(rp.get("max_depth") or "None"),
                "leaf":    rp.get("min_samples_leaf", "—"),
                "feat":    rp.get("max_features", "—"),
                "folds":   r.get("folds", "—"),
                "r2_test": r.get("mean_r2_test", "—"),
                "rmse":    r.get("mean_rmse_test", "—"),
            })
        except Exception:
            continue

    history_rows.sort(
        key=lambda r: r["r2_test"] if isinstance(r["r2_test"], float) else -1,
        reverse=True,
    )

    hist_section = []
    if history_rows:
        hh_s = {
            "padding": "5px 10px", "backgroundColor": C["surface"],
            "color": C["blue"], "fontSize": "11px",
            "border": f"1px solid {C['border']}",
        }
        ht_s = {
            "padding": "4px 10px", "fontSize": "11px", "color": C["text"],
            "borderBottom": f"1px solid {C['border']}",
        }
        del_btn_s = {
            "backgroundColor": "transparent", "color": C["red"],
            "border": f"1px solid {C['red']}", "borderRadius": "3px",
            "cursor": "pointer", "fontSize": "11px", "padding": "1px 7px",
            "lineHeight": "1", "fontFamily": "monospace", "fontWeight": "bold",
        }
        load_btn_s = {
            "backgroundColor": "transparent", "color": C["teal"],
            "border": f"1px solid {C['teal']}", "borderRadius": "3px",
            "cursor": "pointer", "fontSize": "11px", "padding": "1px 9px",
            "lineHeight": "1", "fontFamily": "monospace",
        }
        hist_section = [
            html.Div("All Cached RF Runs", style={
                "color": C["muted"], "fontSize": "11px", "marginBottom": "6px",
            }),
            html.Div(style={"overflowX": "auto"}, children=[
                html.Table(
                    style={"width": "100%", "borderCollapse": "collapse"},
                    children=[
                        html.Thead(html.Tr([html.Th(c, style=hh_s) for c in
                            ["Hash", "Target", "Pairs", "Feats", "n_est", "depth",
                             "leaf", "max_feat", "folds", "R²_test", "RMSE", "", ""]])),
                        html.Tbody([html.Tr([
                            html.Td(row["hash"], style={
                                **ht_s, "fontFamily": "monospace",
                                "color": C["teal"] if row["hash"] == current_h else C["text"],
                            }),
                            html.Td("AUC" if row["target"] == "auc" else "IC50", style=ht_s),
                            html.Td(f"{row['pairs']:,}" if isinstance(row["pairs"], int) else row["pairs"], style=ht_s),
                            html.Td(f"{row['feats']:,}" if isinstance(row["feats"], int) else row["feats"], style=ht_s),
                            html.Td(str(row["n_est"]),  style=ht_s),
                            html.Td(str(row["depth"]),  style=ht_s),
                            html.Td(str(row["leaf"]),   style=ht_s),
                            html.Td(str(row["feat"]),   style=ht_s),
                            html.Td(str(row["folds"]),  style=ht_s),
                            html.Td(
                                f"{row['r2_test']:.4f}" if isinstance(row["r2_test"], float) else str(row["r2_test"]),
                                style={**ht_s, "color": (
                                    C["green"]  if isinstance(row["r2_test"], float) and row["r2_test"] > 0.7 else
                                    C["yellow"] if isinstance(row["r2_test"], float) and row["r2_test"] > 0.4 else
                                    C["red"]
                                )},
                            ),
                            html.Td(
                                f"{row['rmse']:.4f}" if isinstance(row["rmse"], float) else str(row["rmse"]),
                                style=ht_s,
                            ),
                            html.Td(
                                html.Button("Load",
                                    id={"type": "load-btn", "index": row["hash"]},
                                    n_clicks=0, style=load_btn_s,
                                    title=f"Load run {row['hash']} into dashboard"),
                                style={**ht_s, "textAlign": "center", "padding": "3px 6px"},
                            ),
                            html.Td(
                                html.Button("×",
                                    id={"type": "del-btn", "index": row["hash"]},
                                    n_clicks=0, style=del_btn_s,
                                    title=f"Delete cached run {row['hash']}"),
                                style={**ht_s, "textAlign": "center", "padding": "3px 6px"},
                            ),
                        ]) for row in history_rows]),
                    ],
                )
            ]),
        ]

    if not hist_section:
        return html.Div()

    return html.Div(style=CARD_S, children=[
        html.H4("Cached Runs", style={
            "color": SECTION_GREY, "margin": "0 0 12px 0",
            "fontSize": "13px", "letterSpacing": "0.5px",
        }),
        *hist_section,
    ])