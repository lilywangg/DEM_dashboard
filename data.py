"""
data.py — Data loading, filtering, and lazy feature management
"""

from __future__ import annotations
import json, hashlib
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).parent
DATA    = BASE / "data"
OUT_DIR = BASE / "outputs"
RF_DIR  = OUT_DIR / "rf_results"
OUT_DIR.mkdir(exist_ok=True)
RF_DIR.mkdir(exist_ok=True)

# ── Load CTRP data ─────────────────────────────────────────────────────────────
print("Loading CTRP data...")
CTRP_COLS = [
    "cpd_name", "ModelID", "log10_ic50", "area_under_curve",
    "fit_num_param", "conc_pts_fit",
    "p1_conf_width",
    "p2_conf_int_high", "p2_conf_int_low",
    "p4_conf_int_high", "p4_conf_int_low",
    "p1_center", "p2_slope", "p3_total_decline", "p4_baseline",
    "snp_fp_status", "apparent_ec50_umol", "top_test_conc_umol", "pred_pv_high_conc",
]
ctrp_raw = pd.read_csv(
    DATA / "ctrp_final.csv",
    usecols=CTRP_COLS, low_memory=False
)

# Pre-compute quality flags
p3, p4 = ctrp_raw["p3_total_decline"], ctrp_raw["p4_baseline"]
p2     = ctrp_raw["p2_slope"]
inner  = p3 / (0.5 - p4) - 1
ctrp_raw["_valid_math"]  = (inner > 0) & (p2.abs() > 1e-6)
ctrp_raw["_crosses_50"]  = (p4 < 0.5) & ((p3 + p4) > 0.5)
ctrp_raw["_p1_ci"]       = ctrp_raw["p1_conf_width"].abs()
ctrp_raw["_p2_ci"]       = (ctrp_raw["p2_conf_int_high"] - ctrp_raw["p2_conf_int_low"]).abs()
ctrp_raw["_p4_ci"]       = (ctrp_raw["p4_conf_int_high"] - ctrp_raw["p4_conf_int_low"]).abs()
ec50r = ctrp_raw["apparent_ec50_umol"] / ctrp_raw["top_test_conc_umol"].replace(0, np.nan)
ctrp_raw["_no_extrap"]    = ec50r <= 2.0
ctrp_raw["_high_dose_ok"] = ctrp_raw["pred_pv_high_conc"] <= 0.9

# Dataset-level stats
TOTAL_ROWS  = len(ctrp_raw)
TOTAL_DRUGS = ctrp_raw["cpd_name"].nunique()
TOTAL_CELLS = ctrp_raw["ModelID"].nunique()

# IC50 / AUC range for histogram binning (1st–99th percentile)
ic50_full = ctrp_raw["log10_ic50"].dropna()
IC50_LO, IC50_HI = float(ic50_full.quantile(0.01)), float(ic50_full.quantile(0.99))
auc_full  = ctrp_raw["area_under_curve"].dropna()
AUC_LO,  AUC_HI  = float(auc_full.quantile(0.01)),  float(auc_full.quantile(0.99))

# ── Load compound metadata ─────────────────────────────────────────────────────
print("Loading compound metadata...")
_cpd_meta_path = DATA / "metacompound.txt"
if _cpd_meta_path.exists():
    _cpd_meta_raw = pd.read_csv(
        _cpd_meta_path, sep="\t",
        usecols=["cpd_name", "gene_symbol_of_protein_target",
                 "target_or_activity_of_compound", "inclusion_rationale", "cpd_status"],
        dtype=str,
    ).fillna("")
    _cpd_meta_map = _cpd_meta_raw.set_index("cpd_name").to_dict("index")
else:
    _cpd_meta_map = {}

# ── Load cell line metadata ────────────────────────────────────────────────────
print("Loading metadata...")
_meta = pd.read_csv(
    DATA / "Model.csv",
    usecols=["ModelID", "CellLineName", "OncotreeLineage", "OncotreePrimaryDisease"],
    low_memory=False,
)
_meta_map = _meta.set_index("ModelID").to_dict("index")

# Dropdown options
all_drugs_full = sorted(ctrp_raw["cpd_name"].dropna().unique())
all_cells_full = sorted(
    (f"{mid}  ({_meta_map.get(mid, {}).get('CellLineName', mid)})", mid)
    for mid in ctrp_raw["ModelID"].dropna().unique()
)
all_lineages_full = sorted(
    _meta[_meta["ModelID"].isin(ctrp_raw["ModelID"])]["OncotreeLineage"].dropna().unique()
)

# ── Lazy gene matrix & drug fingerprints ──────────────────────────────────────
_gene_matrix = None
_fp_sub      = None


def _ensure_features_loaded() -> None:
    """Load gene expression matrix and drug fingerprints once, lazily."""
    global _gene_matrix, _fp_sub
    if _gene_matrix is not None and _fp_sub is not None:
        return

    print("Loading gene & drug features (lazy)...")
    _genes = pd.read_csv(DATA / "genes_final.csv")
    _lincs = pd.read_csv(DATA / "GSE70138_Broad_LINCS_gene_info_2017-03-06.txt", sep="\t")
    _fp    = pd.read_csv(DATA / "drug_fingerprints.csv")

    # Filter gene columns — drop metadata cols and ENSG IDs
    non_g = {"SequencingID", "ModelID", "IsDefaultEntryForModel",
              "ModelConditionID", "IsDefaultEntryForMC"}
    nc_   = [c for c in _genes.columns if c not in non_g and not c.startswith("ENSG")]

    # Keep genes expressed in ≥10% of cell lines
    km_   = (_genes[nc_] < 0.1).mean(axis=0) <= 0.9
    nc_   = _genes[nc_].columns[km_].tolist()

    # Top 1,000 by prior feature importance, union with 978 LINCS landmark genes
    top1k = _genes[nc_].var().nlargest(1000).index.tolist()

    lg_   = [g for g in _lincs[_lincs["pr_is_lm"] == 1]["pr_gene_symbol"] if g in nc_]
    gs_   = sorted(set(top1k) | set(lg_))
    _gene_matrix = _genes[["ModelID"] + gs_].set_index("ModelID")

    # Drug fingerprints — morgan counts + RDKit descriptors only
    mc_   = [c for c in _fp.columns if c.startswith("morgan_count")]
    excl_ = set(
        [c for c in _fp.columns if c.startswith("morgan_bit")] +
        [c for c in _fp.columns if c.startswith("maccs_")]
    )
    basic_ = {"mol_weight", "logp", "hbd", "hba", "tpsa",
               "rotatable_bonds", "aromatic_rings", "heavy_atoms"}
    rdk_  = [c for c in _fp.columns
              if c not in excl_ and c not in basic_
              and not c.startswith("morgan_count") and c != "cpd_name"]
    _fp_sub = _fp[["cpd_name"] + mc_ + rdk_]

    print(f"✓ Gene matrix: {_gene_matrix.shape[1]} genes | "
          f"Drug FP: {_fp_sub.shape[1] - 1} features.")


# ── Filter helper ──────────────────────────────────────────────────────────────
def apply_filters(
    fit_params, p1_ci, p2_ci, p4_ci, crosses, min_conc,
    snp, no_extrap, high_dose, drugs, cells, lineages
) -> pd.DataFrame:
    """Return a filtered subset of ctrp_raw based on UI filter selections."""
    m = pd.Series(True, index=ctrp_raw.index)

    if fit_params:        m &= ctrp_raw["fit_num_param"].isin(fit_params)
    if p1_ci < 10:        m &= ctrp_raw["_p1_ci"] <= p1_ci
    if p2_ci < 50:        m &= ctrp_raw["_p2_ci"] <= p2_ci
    if p4_ci < 5:         m &= ctrp_raw["_p4_ci"] <= p4_ci
    if crosses == "yes":  m &= ctrp_raw["_crosses_50"]
    elif crosses == "no": m &= ~ctrp_raw["_crosses_50"]
    if min_conc and min_conc > 0:
        m &= ctrp_raw["conc_pts_fit"] >= min_conc
    if snp:               m &= ctrp_raw["snp_fp_status"].isin(snp)
    if no_extrap == "yes": m &= ctrp_raw["_no_extrap"]
    if high_dose == "yes": m &= ctrp_raw["_high_dose_ok"]
    if drugs:             m &= ctrp_raw["cpd_name"].isin(drugs)
    if cells:             m &= ctrp_raw["ModelID"].isin(cells)
    if lineages:
        lin = ctrp_raw["ModelID"].map(
            lambda x: _meta_map.get(x, {}).get("OncotreeLineage", ""))
        m &= lin.isin(lineages)

    return ctrp_raw[m]


def cfg_hash(*args) -> str:
    """MD5 hash of all RF config args — used as cache key for saved runs."""
    def s(x): return sorted(x) if isinstance(x, list) and x else (x or "")
    key = json.dumps([str(s(a)) for a in args], sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()[:12]