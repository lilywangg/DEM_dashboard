"""
evidence.py — Evidence computation for LLM grounding

Three evidence functions corresponding to the three analysis levels:
  - _get_gene_evidence()     : Level 1 dropdown pair (vs global)
  - _get_neighbor_evidence() : Level 1 UMAP point click (vs 15 neighbors)
  - _get_cluster_evidence()  : Level 2 lasso cluster (vs global)
"""

from __future__ import annotations
from collections import Counter

import numpy as np

from data import _meta_map, _cpd_meta_map, _ensure_features_loaded, _gene_matrix, _fp_sub

def _normalise_fp_label(feat: str) -> str:
    """Normalise drug fingerprint feature names to avoid duplicates."""
    return (feat.replace("morgan_count_", "Morgan_")
                .replace("morgan_", "Morgan_")
                .replace("rdkit_", "RDKit_"))


# ── Level 1 — Dropdown pair evidence (vs global) ──────────────────────────────
def _get_gene_evidence(
    model_ids: list, drug: str | None = None, top_n: int = 3
) -> list[dict]:
    """
    Compute gene-expression evidence for a set of ModelIDs vs the global dataset.
    Used by Level 1 dropdown pair selection.

    Returns a list of evidence dicts with E-IDs ready for the evidence table
    and prompt injection.
    """
    _ensure_features_loaded()

    # Access globals after loading
    import data as _d
    gm = _d._gene_matrix

    evidence: list[dict] = []
    eid = 101

    if gm is not None and model_ids:
        present = [m for m in model_ids if m in gm.index]
        if present:
            cluster_expr = gm.loc[present]
            global_mean  = gm.mean()
            global_std   = gm.std().replace(0, 1e-9)
            cluster_mean = cluster_expr.mean()
            z_scores     = (cluster_mean - global_mean) / global_std

            for gene, _ in z_scores.abs().nlargest(top_n).items():
                z   = float(z_scores[gene])
                val = float(cluster_mean[gene])
                pct = float((gm[gene] < val).mean() * 100)
                evidence.append({
                    "id":        f"E-{eid}",
                    "feature":   gene,
                    "value":     f"{val:.3f}",
                    "stat":      f"z={z:+.2f}, {pct:.0f}th pct",
                    "direction": "High" if z > 0 else "Low",
                    "source":    "CCLE Expression",
                    "_z":        z,
                    "_pct":      pct,
                })
                eid += 1

    if drug:
        info   = _cpd_meta_map.get(drug, {})
        target = (info.get("gene_symbol_of_protein_target") or "").strip()
        mech   = (info.get("target_or_activity_of_compound") or "")[:80]
        if target:
            evidence.append({
                "id":        f"E-{eid}",
                "feature":   f"{drug} → {target}",
                "value":     target,
                "stat":      mech or "drug target",
                "direction": "Target",
                "source":    "PubChem / CTRP",
            })

    return evidence


# ── Level 1 — UMAP point click evidence (vs 15 neighbors) ─────────────────────
def _get_neighbor_evidence(
    idx: int, emb_data: dict, top_n: int = 3
) -> list[dict]:
    """
    Compute evidence for a single UMAP point vs its 15 nearest neighbors.
    Used by Level 1 point click.

    Computes:
      • Top-N gene z-scores: this cell line vs its 15 UMAP neighbors
      • Top-N drug fingerprint z-scores: this drug vs neighbors' drugs
      • Drug target metadata
    """
    _ensure_features_loaded()

    import data as _d
    gm  = _d._gene_matrix
    fps = _d._fp_sub

    evidence: list[dict] = []
    eid = 101

    ex        = np.array(emb_data["x"])
    ey        = np.array(emb_data["y"])
    model_ids = emb_data.get("model_id", [])
    drugs     = emb_data.get("drug", [])

    # Find 15 nearest neighbors in UMAP 2D space
    dists       = (ex - ex[idx]) ** 2 + (ey - ey[idx]) ** 2
    nbr_indices = np.argsort(dists)[1:16]  # skip self

    # Gene features: this cell line vs its neighbors
    if gm is not None and idx < len(model_ids):
        mid      = model_ids[idx]
        nbr_mids = [model_ids[i] for i in nbr_indices if i < len(model_ids)]
        if mid in gm.index:
            pt_expr     = gm.loc[mid]
            nbr_present = [m for m in nbr_mids if m in gm.index]
            if nbr_present:
                nbr_expr = gm.loc[nbr_present]
                nbr_std  = nbr_expr.std().replace(0, 1e-9)
                z_scores = (pt_expr - nbr_expr.mean()) / nbr_std
                for gene, _ in z_scores.abs().nlargest(top_n).items():
                    z   = float(z_scores[gene])
                    val = float(pt_expr[gene])
                    evidence.append({
                        "id":        f"E-{eid}",
                        "feature":   gene,
                        "value":     f"{val:.3f}",
                        "stat":      f"z={z:+.2f} vs 15 neighbors",
                        "direction": "High" if z > 0 else "Low",
                        "source":    "CCLE Expression",
                        "modality":  "Gene",
                        "_z":        z,
                    })
                    eid += 1

    # Drug fingerprint features: this drug vs neighbors' drugs
    if fps is not None and idx < len(drugs):
        drug      = drugs[idx]
        nbr_drugs = [drugs[i] for i in nbr_indices if i < len(drugs)]
        fp_cols   = [c for c in fps.columns if c != "cpd_name"]
        fp_drug   = fps[fps["cpd_name"] == drug]
        fp_nbrs   = fps[fps["cpd_name"].isin(nbr_drugs)]

        if not fp_drug.empty and not fp_nbrs.empty and fp_cols:
            drug_vals = fp_drug[fp_cols].iloc[0]
            nbr_std   = fp_nbrs[fp_cols].std()
            nbr_std   = nbr_std.where(nbr_std > 0.01, np.nan)  # require meaningful variance
            z_fp      = ((drug_vals - fp_nbrs[fp_cols].mean()) / nbr_std).fillna(0)
            z_fp      = z_fp.clip(-10, 10)  # cap to avoid near-zero std blowup

            seen_feats: set[str] = set()
            for feat, _ in z_fp.abs().nlargest(top_n * 3).items():
                feat_label = _normalise_fp_label(feat)
                if feat_label in seen_feats:
                    continue
                seen_feats.add(feat_label)
                z   = float(z_fp[feat])
                val = float(drug_vals[feat])
                evidence.append({
                    "id":        f"E-{eid}",
                    "feature":   feat_label,
                    "value":     f"{val:.3f}",
                    "stat":      f"z={z:+.2f} vs 15 neighbors",
                    "direction": "High" if z > 0 else "Low",
                    "source":    "Drug Fingerprint",
                    "modality":  "Drug FP",
                    "_z":        z,
                })
                eid += 1
                if len([e for e in evidence if e.get("modality") == "Drug FP"]) >= top_n:
                    break

    # Drug target metadata
    if idx < len(drugs):
        drug   = drugs[idx]
        info   = _cpd_meta_map.get(drug, {})
        target = (info.get("gene_symbol_of_protein_target") or "").strip()
        mech   = (info.get("target_or_activity_of_compound") or "")[:80]
        if target:
            evidence.append({
                "id":        f"E-{eid}",
                "feature":   f"{drug} → {target}",
                "value":     target,
                "stat":      mech or "drug target",
                "direction": "Target",
                "source":    "PubChem / CTRP",
                "modality":  "Drug Target",
            })

    return evidence


# ── Level 2 — Lasso cluster evidence (vs global) ──────────────────────────────
def _get_cluster_evidence(
    points: list, emb_data: dict, top_n: int = 3
) -> list[dict]:
    """
    Compute split-modality evidence for a lasso-selected cluster vs the global dataset.
    Used by Level 2 cluster analysis.

    Computes:
      • Top-N gene z-scores:            cluster mean vs global mean
      • Top-N drug fingerprint z-scores: cluster drugs vs all drugs globally
      • Dominant drug target metadata
    """
    _ensure_features_loaded()

    import data as _d
    gm  = _d._gene_matrix
    fps = _d._fp_sub

    evidence: list[dict] = []
    eid = 101

    model_ids        = _get_selected_model_ids(points, emb_data)
    drugs_in_cluster = [p.get("customdata", ["?", "?"])[1] for p in points]
    top_drug         = (Counter(drugs_in_cluster).most_common(1)[0][0]
                        if drugs_in_cluster else None)

    # Gene features: cluster vs global
    if gm is not None and model_ids:
        present = [m for m in model_ids if m in gm.index]
        if present:
            cluster_expr = gm.loc[present]
            global_std   = gm.std().replace(0, 1e-9)
            z_scores     = (cluster_expr.mean() - gm.mean()) / global_std
            for gene, _ in z_scores.abs().nlargest(top_n).items():
                z   = float(z_scores[gene])
                val = float(cluster_expr[gene].mean())
                pct = float((gm[gene] < val).mean() * 100)
                evidence.append({
                    "id":        f"E-{eid}",
                    "feature":   gene,
                    "value":     f"{val:.3f}",
                    "stat":      f"z={z:+.2f}, {pct:.0f}th pct globally",
                    "direction": "High" if z > 0 else "Low",
                    "source":    "CCLE Expression",
                    "modality":  "Gene",
                    "_z":        z,
                    "_pct":      pct,
                })
                eid += 1

    # Drug fingerprint features: cluster drugs vs all drugs globally
    if fps is not None and drugs_in_cluster:
        cluster_drugs = list(set(drugs_in_cluster))
        fp_cols       = [c for c in fps.columns if c != "cpd_name"]
        fp_cluster    = fps[fps["cpd_name"].isin(cluster_drugs)]

        if not fp_cluster.empty and fp_cols:
            global_std = fps[fp_cols].std()
            global_std = global_std.where(global_std > 0.01, np.nan)
            z_fp       = ((fp_cluster[fp_cols].mean() - fps[fp_cols].mean()) / global_std).fillna(0)
            z_fp       = z_fp.clip(-10, 10)

            seen_feats: set[str] = set()
            for feat, _ in z_fp.abs().nlargest(top_n * 3).items():
                feat_label = _normalise_fp_label(feat)
                if feat_label in seen_feats:
                    continue
                seen_feats.add(feat_label)
                z   = float(z_fp[feat])
                val = float(fp_cluster[feat].mean())
                evidence.append({
                    "id":        f"E-{eid}",
                    "feature":   feat_label,
                    "value":     f"{val:.3f}",
                    "stat":      f"z={z:+.2f} cluster vs global",
                    "direction": "High" if z > 0 else "Low",
                    "source":    "Drug Fingerprint",
                    "modality":  "Drug FP",
                    "_z":        z,
                })
                eid += 1
                if len([e for e in evidence if e.get("modality") == "Drug FP"]) >= top_n:
                    break

    # Dominant drug target metadata
    if top_drug:
        info   = _cpd_meta_map.get(top_drug, {})
        target = (info.get("gene_symbol_of_protein_target") or "").strip()
        mech   = (info.get("target_or_activity_of_compound") or "")[:80]
        if target:
            evidence.append({
                "id":        f"E-{eid}",
                "feature":   f"{top_drug} → {target}",
                "value":     target,
                "stat":      mech or "drug target",
                "direction": "Target",
                "source":    "PubChem / CTRP",
                "modality":  "Drug Target",
            })

    return evidence


# ── Coordinate matching helper ─────────────────────────────────────────────────
def _get_selected_model_ids(points: list, emb_data: dict) -> list[str]:
    """
    Map lasso-selected UMAP points back to ModelIDs via nearest-coordinate matching.
    """
    if not emb_data or not points:
        return []

    ex        = np.array(emb_data["x"])
    ey        = np.array(emb_data["y"])
    model_ids = emb_data.get("model_id", [])
    result    = []

    for p in points:
        px, py = p.get("x"), p.get("y")
        if px is None:
            continue
        idx = int(np.argmin((ex - px) ** 2 + (ey - py) ** 2))
        if idx < len(model_ids):
            result.append(model_ids[idx])

    return result