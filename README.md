# Drug Efficacy Model (DEM) Dashboard

An interactive dashboard for predicting and explaining cancer cell line drug sensitivity, built as part of the Spring 2026 Emory CS 584–SK collaboration.

Reproduces the Random Forest pipeline from [Griffiths et al. 2024](https://doi.org/10.1101/2024.01.17.24301444) with an interactive UMAP visualization and grounded LLM analysis layer.

---

## Overview

**Filter** — Apply data quality filters to the CTRP drug response dataset (curve quality, SNP verification, extrapolation checks).

**Train** — Train a Random Forest regressor on cancer cell line gene expression + drug fingerprint features to predict log₁₀(IC50) or AUC.

**Explore** — Visualize results in a leaf-agreement UMAP. Points that cluster together share the same RF decision paths — they are biologically similar according to the model.

**Explain** — Three levels of LLM analysis powered by Gemini 2.5 Flash:
- **Level 1** — Single sample: why does this cell line respond to this drug?
- **Level 2** — Cluster: what shared biology unifies a lasso-selected group?
- **Level 3** — Global: what does the full drug-response landscape reveal?

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/lilywangg/dem_dashboard.git
cd dem_dashboard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add data files
Place the following files in the `data/` folder (available upon request):
```
data/
├── ctrp_final.csv
├── drug_fingerprints.csv
├── genes_final.csv
├── Model.csv
├── metacompound.txt
└── GSE70138_Broad_LINCS_gene_info_2017-03-06.txt
```

### 4. Set up your Gemini API key
```bash
cp .env.example .env
```
Then edit `.env` and add your key:
```
GEMINI_API_KEY=your_key_here
```
Get a key at [aistudio.google.com](https://aistudio.google.com). The dashboard uses **Gemini 2.5 Flash** (~$0.30 / 1M input tokens). Responses are cached locally so repeat queries are free. See current pricing at [ai.google.dev](https://ai.google.dev).

### 5. Run the dashboard
```bash
python dashboard.py
```
Open [http://127.0.0.1:8050](http://127.0.0.1:8050) in your browser.

---

## Detailed Workflow

### 1. Data Quality Filters

Before training, the CTRP dose-response data is filtered to retain only reliable IC50 measurements. The filters are grouped into two categories.

**Data Quality** filters operate on the curve-fitting parameters of the 4-parameter Hill equation fitted to each dose-response curve:

- **Fit Parameters** — Whether the curve was fitted with 2 or 3 free parameters. A 2-parameter fit constrains more of the curve shape; a 3-parameter fit is more flexible but may overfit noisy curves. Both are included by default.
- **p1 CI width** — Confidence interval on p1 (the EC50, i.e. the concentration at 50% inhibition). A wide CI means the IC50 estimate is unreliable. Values below 1.0 indicate tight, well-determined EC50 estimates.
- **p2 CI width** — Confidence interval on p2 (the Hill slope). Controls how steep the curve is. A wide CI means the slope is poorly determined.
- **p4 CI width** — Confidence interval on p4 (the baseline viability at high drug concentration). A wide CI means the lower asymptote of the curve is uncertain.
- **Concentration Points** — Minimum number of concentration measurements used to fit the curve. More points produce more reliable fits. Options: any, ≥12, or ≥16.
- **Crosses 50% Inhibition** — Whether the dose-response curve actually reaches 50% inhibition within the tested concentration range. If not, the IC50 must be extrapolated beyond measured data, making it unreliable. **This is the most important filter** — default is "Crosses only."

**Biological Checks** filter for data quality beyond curve fitting:

- **SNP Fingerprint Status** — Whether the cell line's genetic identity was verified by SNP fingerprinting against a reference. Options: SNP-matched (identity confirmed), Not tested, or Unconfirmed. Default is SNP-matched only.
- **No Extrapolation** — Requires the apparent EC50 to fall within 2× the highest tested concentration, ensuring the IC50 is interpolated from actual measurements rather than extrapolated beyond them.
- **High-Dose Inhibition** — Requires that the predicted viability at the highest tested concentration is ≤ 0.9 (i.e. the drug kills at least 10% of cells at max dose). If viability stays near 1.0 at max dose, the curve is essentially flat and the IC50 is meaningless.

**Sample Filters** restrict the dataset to specific drugs, cell lines, or cancer lineages of interest before training.

The filter summary panel shows how many samples, drugs, and cell lines remain after all filters are applied, as a fraction of the full dataset.

---

### 2. Random Forest Hyperparameters

After filtering, a Random Forest regressor is trained on the filtered dataset. Each (cell line × drug) pair is represented by a feature vector combining cell line gene expression and drug structural fingerprints (Morgan count fingerprints + RDKit descriptors). Gene features are selected by taking the union of the top 1,000 most variable genes across cell lines and the 978 LINCS L1000 landmark genes — a curated set of genes shown to capture the majority of transcriptomic variation genome-wide. This typically yields ~1,000–1,100 genes from the full CCLE expression matrix.
The available hyperparameters are:

- **n_estimators** — Number of trees in the forest. More trees improve stability but increase training time. Default: 100.
- **max_depth** — Maximum depth of each tree. Setting to 0 means unlimited depth (trees grow until leaves are pure). Deeper trees can capture more complex patterns but may overfit. Default: 0 (None).
- **min_samples_leaf** — Minimum number of samples required at each leaf node. Higher values regularize the model and reduce overfitting. Default: 5.
- **max_features (%)** — Fraction of features randomly considered at each split. Lower values increase diversity between trees (stronger regularization). Default: 30%.
- **CV folds** — Number of cross-validation folds. With 1 fold, a single train/test split is used. With k folds, the dataset is split k times for more robust evaluation. Splits are done at the cell line level to prevent data leakage — a cell line appears entirely in train or entirely in test.
- **Test Split (%)** — Fraction of cell lines held out for testing. Default: 20%.
- **UMAP Samples** — Number of (cell line × drug) pairs randomly sampled for the default UMAP visualization. Higher values give a more complete picture but take longer to embed. Default: 500. Note that the 'Pharmacological View' and 'Tissue View' will display all the samples from the original dataset regardless of the UMAP sample setting. 
- **Target Variable** — Whether to predict log₁₀(IC50) (the concentration at 50% inhibition) or AUC (area under the dose-response curve). IC50 requires the curve to cross 50% inhibition; AUC can be computed for all curves.

The **Run History scatter plot** sits above the results panel and tracks all cached runs visually. Each point is one run — the x-axis shows number of training samples, the y-axis shows test R² or RMSE (toggle with the radio buttons), the color encodes model complexity derived from the hyperparameters (teal = simple model, orange = complex model), and the shape encodes the target variable (● = log₁₀(IC50), ▲ = AUC). Hovering over a point shows its full hyperparameter configuration. Clicking a point loads that run.

Each run is automatically saved to `outputs/rf_results/` as a JSON file keyed by a hash of all hyperparameter and filter settings. Identical configurations load instantly from cache without retraining.

---

### 3. Accessing Past Runs

There are two ways to restore previously trained models without retraining.

**Run History scatter plot** —  **Clicking a point loads that run** — its metrics appear below and the UMAP embedding is restored.

**Cached Runs table** — Located at the bottom of the dashboard. Lists all saved runs sorted by R² (best first), showing hash, target variable, number of pairs and features, hyperparameters, and performance metrics. Each row has a **Load** button to restore the run and a **×** button to permanently delete it (with confirmation). Deleting a run also removes its associated leaf assignment files.

When a run is loaded from either location, the UMAP embedding is restored and all LLM analysis features become available immediately.

---

### 4. UMAP and LLM Analysis

**How the UMAP works**

The UMAP is not embedded in raw feature space. Instead it uses a *leaf-agreement kernel*: after training, `rf.apply()` returns the leaf node each sample lands in for every tree. Two samples with high leaf agreement — landing in the same leaf across many trees — are considered biologically similar, meaning the model routes them through identical decision paths. This pairwise similarity matrix is converted to a distance matrix and passed to UMAP with `metric="precomputed"`.

The key insight is that points clustering together in this UMAP are not just similar in gene expression or drug structure individually — they share the same *joint* drug-response logic according to the Random Forest. A lung cancer cell line and a breast cancer cell line may cluster together if they share the same gene expression features that make them sensitive to the same drug class.

**Three UMAP views**

- **Default View** — A random sample of up to N points (controlled by the UMAP Samples hyperparameter, default 500) from the full training set, colored by log₁₀(IC50). Green = sensitive (low IC50, drug works at low concentration), red = resistant (high IC50, drug requires high concentration). This gives an overview of the full drug-response landscape the model has learned.

- **Pharmacological View** — Select one or more drugs from the dropdown. The dashboard loads the full leaf assignment matrix saved during training and re-embeds **all** pairs for the selected drug(s) — not just the N-sample default. This gives a complete view of how a specific drug's sensitivity landscape is structured across all tested cell lines, revealing which cell lines cluster together as sensitive vs. resistant to that drug. Capped at 3,000 points for UMAP performance. Points are colored by IC50.

- **Tissue View** — Same as Pharmacological View but filtered by cancer lineage instead of drug. Re-embeds all pairs for the selected lineage(s) to show drug sensitivity patterns within a specific tissue context, revealing which drugs cluster together as effective for that cancer type. 

To reset either filtered view back to the default, clear the dropdown selection or click the **↺ Reset View** button.

**Three levels of LLM analysis**

All analysis is powered by Gemini 2.5 Flash. Before calling the LLM, the dashboard pre-computes structured evidence from the data — gene expression z-scores and drug fingerprint z-scores relative to appropriate reference populations — and injects this as grounded context into the prompt. Every response includes a groundedness badge showing how many factual claims were backed by pre-computed evidence. The raw evidence items are always shown in the Evidence Table below each analysis, allowing independent verification of the LLM's claims.

- **Level 1 — Single Sample**: Click any point on the UMAP, or select a drug and cell line from the dropdowns and click Generate Analysis. For point clicks, gene expression z-scores are computed relative to the 15 nearest UMAP neighbors (what makes this point unusual in its local neighborhood?). For dropdown selections, z-scores are computed relative to the global dataset. The LLM generates a 3-sentence mechanistic hypothesis explaining why that specific cell line responds to that specific drug, citing gene expression patterns, drug structural features, and drug target metadata.

- **Level 2 — Cluster**: Use the lasso or box select tool on the UMAP to select a group of points. Gene expression z-scores are computed by comparing the cluster's mean expression to the global dataset mean, and drug fingerprint z-scores compare the cluster's drugs to all drugs globally. The LLM identifies the shared molecular mechanism unifying the cluster, links the drug structural scaffold to the gene expression context, and explains what the RF leaf-agreement reveals about the shared biological vulnerability.

- **Level 3 — Global**: Fires automatically when a UMAP embedding is loaded. Uses the top drugs by frequency, top cancer lineages, the IC50 distribution, and the global RF feature importances (weighted impurity decrease) to describe the overall drug-response landscape captured by the model in 3 sentences.

---

## Project Structure

```
DEM_Dashboard/
├── dashboard.py        # App entry point — layout and callbacks
├── data.py             # Data loading, filtering, feature management
├── evidence.py         # Z-score evidence computation for LLM grounding
├── prompts.py          # LLM prompt builders for all three analysis levels
├── llm.py              # Gemini API call with caching and retry logic
├── renders.py          # UI components, style constants, RF metrics rendering
├── umap_builder.py     # UMAP figure builders (default, drug, lineage views)
├── kernel.py           # Leaf-agreement kernel implementation
├── data/               # Data files (not included in repo — see Setup)
├── outputs/
│   └── rf_results/     # Cached RF runs (auto-created)
├── .env.example        # API key template
├── requirements.txt    # Python dependencies
└── README.md
```

---

## Data Sources

| File | Source |
|------|--------|
| `ctrp_final.csv` | [Cancer Therapeutics Response Portal (CTRP)](https://ctd2-data.nci.nih.gov/Public/Broad/CTRPv2.0_2015_ctd2_ExpandedDataset/) |
| `genes_final.csv` | [DepMap / CCLE](https://depmap.org) |
| `drug_fingerprints.csv` | Computed from SMILES via RDKit |
| `Model.csv` | [DepMap Model metadata](https://depmap.org) |
| `metacompound.txt` | CTRP compound metadata |
| `GSE70138_...txt` | [LINCS L1000 landmark genes](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70138) |

---

## Caching

- **RF results** — saved to `outputs/rf_results/` as JSON (metrics, embeddings, feature importances) plus `.npy`/`.npz` files (full leaf assignments for re-embedding). Keyed by a hash of all hyperparameters and filter settings.
- **Gemini responses** — cached in `.gemini_cache/` by prompt hash. Repeat queries are free.

---

## Reference

Griffiths, J.I. et al. (2024). Genome-wide analysis identifies mediators of drug sensitivity in triple-negative breast cancer. *Nature Communications*.
