"""
prompts.py — LLM prompt builders for all three analysis levels
"""

from __future__ import annotations
from collections import Counter

import numpy as np

from data import ctrp_raw, _meta_map, _cpd_meta_map

# ── Citation and instruction constants ────────────────────────────────────────
_CITATION_RULES = """\
CITATION RULES — machine-parsed, do NOT paraphrase or remove tags:
  After naming a gene               → append  [Evidence: <GENE_NAME>]
  After citing a structural feature → append  [Stats: Structural_Feature]
  After citing a similarity value   → append  [Stats: Similarity]
  After citing median / IC50 value  → append  [Stats: Median_IC50]
  After naming a cancer type        → append  [Context: Disease]
  After citing a neighbor count     → append  [Stats: Neighbor_Count]
Rules: (1) Copy numbers exactly from evidence. (2) Do NOT invent values.
(3) Every factual sentence must contain at least one tag. (4) No hedging.\
"""

_DRUG_GENE_INSTRUCTION = (
    "Hypothesize how the Drug Structural Features (chemical MoA / binding scaffold) "
    "interact with the Gene Expression Features (cellular context) to produce the "
    "observed sensitivity. Be specific — name the drug target, the aberrant gene, "
    "and the pathway linking them."
)


# ── Shared evidence block renderer ────────────────────────────────────────────
def _evidence_prompt_block(evidence: list[dict]) -> str:
    """Format evidence dicts as a structured block for prompt injection."""
    if not evidence:
        return ""

    lines      = ["Pre-computed evidence (structured by modality):"]
    gene_items = [e for e in evidence if e.get("modality") == "Gene"]
    fp_items   = [e for e in evidence if e.get("modality") == "Drug FP"]
    tgt_items  = [e for e in evidence if e.get("modality") == "Drug Target"]
    other      = [e for e in evidence
                  if e.get("modality") not in ("Gene", "Drug FP", "Drug Target")]

    def _fmt(e):
        return (f"  {e['id']} [{e.get('modality', '?')}] "
                f"{e['feature']}  —  {e['value']}  ({e['stat']})")

    if gene_items:
        lines.append("  Gene Expression (CCLE):")
        lines.extend(_fmt(e) for e in gene_items)
    if fp_items:
        lines.append("  Drug Structural Fingerprints:")
        lines.extend(_fmt(e) for e in fp_items)
    if tgt_items:
        lines.append("  Drug Target:")
        lines.extend(_fmt(e) for e in tgt_items)
    for e in other:
        lines.append(_fmt(e))

    lines.append(f"\n{_CITATION_RULES}")
    return "\n".join(lines)


# ── Level 3 — Global UMAP ─────────────────────────────────────────────────────
def _build_global_prompt(emb_data: dict) -> str:
    """
    Level 3 prompt — analyses the entire UMAP embedding.
    Uses global RF feature importances and top drugs/lineages.
    """
    drugs     = emb_data.get("drug", [])
    lineages  = emb_data.get("lineage", [])
    ics       = np.array(emb_data.get("ic50", [0.0]))
    tgt_label = emb_data.get("tgt_label", "log₁₀(IC50)")
    imps      = emb_data.get("feature_importances", [])[:8]
    top_feats = [f["feature"] for f in imps]

    top_drugs = Counter(drugs).most_common(5)
    top_lins  = Counter(lineages).most_common(5)

    drug_strs = []
    for d, cnt in top_drugs:
        info   = _cpd_meta_map.get(d, {})
        target = (info.get("gene_symbol_of_protein_target", "") or "?")
        mech   = (info.get("target_or_activity_of_compound", "") or "")[:60]
        drug_strs.append(f"  {d} (n={cnt}, target={target}): {mech}")

    return (
        "You are a cancer pharmacogenomics expert. "
        "Analyze this UMAP of RF leaf-agreement space.\n\n"
        f"N = {len(drugs):,} (cell-line × drug) pairs embedded by Random Forest leaf agreement.\n\n"
        "Top drugs:\n" + "\n".join(drug_strs) + "\n\n"
        f"Top lineages: {', '.join(f'{l} (n={c})' for l, c in top_lins)}\n\n"
        f"{tgt_label} — mean: {ics.mean():.3f}, range: {ics.min():.3f}–{ics.max():.3f}\n\n"
        f"Top RF features (global weighted impurity decrease): {', '.join(top_feats) or 'N/A'}\n\n"
        "In 3 sentences, describe the global structure of this drug-response landscape. "
        "What do the dominant drugs, lineages, and top features reveal about the biology "
        "captured by this model? Reference specific drugs and genes."
    )


# ── Level 2 — Cluster ─────────────────────────────────────────────────────────
def _build_cluster_prompt(
    points: list, tgt_label: str, evidence: list[dict] | None = None
) -> str:
    """
    Level 2 prompt — lasso-selected cluster analysis.
    Includes gene z-scores, drug fingerprint z-scores, and pathway enrichment.
    """

    cd    = [p.get("customdata", ["?", "?", 0.0]) for p in points]
    cls   = [x[0] for x in cd]
    drugs = [x[1] for x in cd]
    ics   = np.array([float(x[2]) for x in cd])
    lins  = [x[3] if len(x) > 3 else "?" for x in cd]

    top_drugs = Counter(drugs).most_common(5)
    top_lins  = Counter(lins).most_common(3)

    drug_strs = []
    for d, cnt in top_drugs:
        info   = _cpd_meta_map.get(d, {})
        target = info.get("gene_symbol_of_protein_target", "?") or "?"
        mech   = (info.get("target_or_activity_of_compound", "") or "")[:80]
        drug_strs.append(f"  {d} (n={cnt}, target={target}): {mech}")

    ev_block = _evidence_prompt_block(evidence or [])

    return (
        "You are a clinical bioinformatics analyst. "
        f"A user lasso-selected {len(points)} (cell-line × drug) pairs "
        "from the RF leaf-agreement UMAP.\n"
        "These samples share decision paths in the Random Forest — "
        "they are biologically similar.\n\n"
        f"Top drugs in cluster:\n" + "\n".join(drug_strs) + "\n\n"
        f"Top lineages: {', '.join(f'{l} (n={c})' for l, c in top_lins)}\n"
        f"{tgt_label} — mean: {ics.mean():.3f}, range: {ics.min():.3f}–{ics.max():.3f}\n"
        f"Unique cell lines: {len(set(cls))}, Unique drugs: {len(set(drugs))}\n"
        + ("\n" + ev_block + "\n\n" if ev_block else "\n")
        + "Generate exactly 3 sentences:\n"
          "1. What shared molecular mechanism unifies this cluster (cite gene evidence)?\n"
          "2. How do the Drug Structural Features interact with these gene-expression drivers?\n"
          "3. What does the RF leaf-agreement tell us about shared biological vulnerability?\n"
          f"{_DRUG_GENE_INSTRUCTION}"
    )


# ── Level 1 — Dropdown pair ───────────────────────────────────────────────────
def _build_pair_prompt(
    drug: str, cell_line: str, evidence: list[dict] | None = None
) -> str:
    """
    Level 1 prompt — drug × cell line pair selected from dropdowns.
    Uses global z-scores since there is no UMAP neighborhood context.
    """
    info    = _cpd_meta_map.get(drug, {})
    target  = info.get("gene_symbol_of_protein_target", "unknown") or "unknown"
    mech    = (info.get("target_or_activity_of_compound", "") or "")[:120] or "unknown mechanism"
    status  = info.get("cpd_status", "") or ""
    cl_meta = _meta_map.get(cell_line, {})
    cl_name = cl_meta.get("CellLineName", cell_line)
    lineage = cl_meta.get("OncotreeLineage", "unknown")
    disease = cl_meta.get("OncotreePrimaryDisease", lineage)

    pair_ic50 = (
        ctrp_raw[
            (ctrp_raw["cpd_name"] == drug) & (ctrp_raw["ModelID"] == cell_line)
        ]["log10_ic50"].dropna()
    )
    if len(pair_ic50):
        ic50_mean = float(pair_ic50.mean())
        pct_ic    = float(np.mean(ctrp_raw["log10_ic50"].dropna() < ic50_mean)) * 100
        sens      = ("highly sensitive" if pct_ic < 20
                     else "resistant" if pct_ic > 80 else "intermediate")
        ic50_str  = f"{ic50_mean:.3f} ({sens}, {pct_ic:.0f}th percentile)"
    else:
        ic50_str = "not measured in dataset"

    ev_block = _evidence_prompt_block(evidence or [])

    return (
        "You are a clinical bioinformatics analyst generating a structured "
        "drug sensitivity rationale.\n\n"
        f"Cell line:  {cl_name} ({cell_line})  —  "
        f"{disease} [Context: Disease] ({lineage} lineage)\n"
        f"Drug:       {drug}  [{status}]\n"
        f"  Target gene(s):  {target}\n"
        f"  Mechanism:       {mech}\n"
        f"log₁₀(IC50): {ic50_str}\n\n"
        + (ev_block + "\n\n" if ev_block else "")
        + "Generate exactly 3 sentences:\n"
          f"1. Mechanistic hypothesis: why does {cl_name} show this response to {drug}?\n"
          f"   Reference the drug target ({target}) and the {disease} biology.\n"
          "2. How do the Drug Structural Features (chemical scaffold / MoA) interact with\n"
          "   the Gene Expression Features to produce this sensitivity?\n"
          "3. Cite the IC50 and at least one gene evidence item with proper tags.\n"
          f"{_DRUG_GENE_INSTRUCTION}"
    )


# ── Level 1 — UMAP point click ────────────────────────────────────────────────
def _build_point_prompt(
    idx: int, emb_data: dict, evidence: list[dict] | None = None
) -> str:
    """
    Level 1 prompt — single point clicked on UMAP.
    Uses 15-nearest-neighbor z-scores for local context.
    """
    dr  = emb_data["drug"][idx]
    cl  = emb_data["cell_line"][idx]
    ic  = float(emb_data["ic50"][idx])
    lin = (emb_data.get("lineage") or ["Unknown"] * len(emb_data["drug"]))[idx]
    tgt = emb_data.get("tgt_label", "Response")

    info   = _cpd_meta_map.get(dr, {})
    target = info.get("gene_symbol_of_protein_target", "unknown") or "unknown"
    mech   = info.get("target_or_activity_of_compound", "unknown mechanism") or "unknown mechanism"
    status = info.get("cpd_status", "") or ""

    pct  = float(np.mean(np.array(emb_data["ic50"]) < ic)) * 100
    sens = "highly sensitive" if pct < 20 else "resistant" if pct > 80 else "intermediate"

    ev_block  = _evidence_prompt_block(evidence or [])
    nbr_count = min(15, len(emb_data["x"]) - 1)

    return (
        "You are a clinical bioinformatics analyst generating a structured "
        "drug sensitivity rationale.\n\n"
        f"Cell line:  {cl}  ({lin} cancer) [Context: Disease]\n"
        f"Drug:       {dr}  [{status}]\n"
        f"  Target gene(s):  {target}\n"
        f"  Mechanism:       {mech}\n"
        f"{tgt}: {ic:.3f} ({sens}, {pct:.0f}th percentile)\n"
        f"Neighborhood: {nbr_count} nearest neighbors in RF leaf-agreement space "
        f"[Stats: Neighbor_Count]\n\n"
        + (ev_block + "\n\n" if ev_block else "")
        + "Generate exactly 3 sentences:\n"
          f"1. Why does {cl} show {sens} response to {dr}? "
          f"Reference the drug target ({target}) and {lin} cancer biology.\n"
          "2. How do the Drug Structural Features (z-score vs neighbors) interact with the\n"
          "   Gene Expression Features to produce this sensitivity?\n"
          "3. Cite at least one gene [Evidence: GENE] and one structural feature "
          "[Stats: Structural_Feature].\n"
          f"{_DRUG_GENE_INSTRUCTION}"
    )