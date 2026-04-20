"""
kernel.py
---------
Leaf-agreement similarity kernel for the DEM.

The kernel between two samples i and j is the fraction of trees in which
they land in the same leaf node:

    K(i, j) = (1 / T) * sum_t [ leaf_t(i) == leaf_t(j) ]

where T = number of trees and leaf_t(x) is the leaf index of sample x in tree t.

Usage
-----
  from kernel import leaf_agreement_kernel, load_leaf_assignments

  # Load pre-computed leaf assignments
  leaf_df = load_leaf_assignments("outputs/leaf_assignments.parquet")

  # Query-to-train kernel: shape (n_query, n_train)
  K = leaf_agreement_kernel(leaf_query, leaf_train)

  # Full square kernel on the training set: shape (n_train, n_train)
  K = leaf_agreement_kernel(leaf_train, leaf_train)
"""

import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

META_COLS = {"ModelID", "cpd_name", "log10_ic50"}  # non-leaf columns in parquet


def load_leaf_assignments(path: str) -> pd.DataFrame:
    """Load leaf assignments from parquet or compressed CSV."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, compression="gzip" if path.endswith(".gz") else None)


def get_leaf_matrix(leaf_df: pd.DataFrame) -> np.ndarray:
    """Extract the (n_samples, n_trees) integer leaf-ID array from the DataFrame."""
    tree_cols = [c for c in leaf_df.columns if c.startswith("tree_")]
    return leaf_df[tree_cols].values.astype(np.int32)


# ── Core kernel ───────────────────────────────────────────────────────────────

def leaf_agreement_kernel(
    leaf_a: np.ndarray,
    leaf_b: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Compute the leaf-agreement kernel between two leaf matrices.

    Parameters
    ----------
    leaf_a : (n_a, T) int array — leaf IDs for set A
    leaf_b : (n_b, T) int array — leaf IDs for set B
    batch_size : rows of A to process at once (controls peak memory)

    Returns
    -------
    K : (n_a, n_b) float32 array — fraction of trees with matching leaves
    """
    n_a, T = leaf_a.shape
    n_b    = leaf_b.shape[0]
    K      = np.empty((n_a, n_b), dtype=np.float32)

    # Process in batches over A to keep memory bounded
    for start in range(0, n_a, batch_size):
        end   = min(start + batch_size, n_a)
        chunk = leaf_a[start:end]  # (chunk, T)

        # Broadcast comparison: (chunk, 1, T) == (1, n_b, T) → (chunk, n_b, T)
        matches = (chunk[:, np.newaxis, :] == leaf_b[np.newaxis, :, :])  # bool
        K[start:end] = matches.mean(axis=2)  # fraction of trees

    return K


# ── Convenience wrappers ──────────────────────────────────────────────────────

def query_kernel(
    query_leaf: np.ndarray,
    train_leaf: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Query-to-train kernel.  Shape: (n_query, n_train).
    """
    return leaf_agreement_kernel(query_leaf, train_leaf, batch_size=batch_size)


def self_kernel(
    leaf: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Square kernel on a single set.  Shape: (n, n).
    """
    return leaf_agreement_kernel(leaf, leaf, batch_size=batch_size)


# ── Nearest-neighbor lookup ────────────────────────────────────────────────────

def top_k_neighbors(
    query_leaf: np.ndarray,
    train_leaf: np.ndarray,
    train_meta: pd.DataFrame,
    k: int = 10,
) -> list[pd.DataFrame]:
    """
    For each query sample, return the top-k most similar training samples.

    Parameters
    ----------
    query_leaf  : (n_q, T) leaf IDs for query samples
    train_leaf  : (n_tr, T) leaf IDs for training samples
    train_meta  : DataFrame aligned with train_leaf (must include ModelID, cpd_name, target)
    k           : number of neighbors to return per query

    Returns
    -------
    List of length n_q; each element is a DataFrame of k rows from train_meta
    with an added 'similarity' column.
    """
    K = leaf_agreement_kernel(query_leaf, train_leaf)  # (n_q, n_tr)
    results = []
    for i in range(K.shape[0]):
        idx  = np.argsort(K[i])[::-1][:k]
        rows = train_meta.iloc[idx].copy()
        rows["similarity"] = K[i][idx]
        results.append(rows.reset_index(drop=True))
    return results


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys

    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    leaf_path = os.path.join(out_dir, "leaf_assignments.parquet")
    if not os.path.exists(leaf_path):
        leaf_path = os.path.join(out_dir, "leaf_assignments.csv.gz")
    if not os.path.exists(leaf_path):
        print("No leaf_assignments file found. Run train_dem.py first.")
        sys.exit(1)

    print("Loading leaf assignments...")
    leaf_df = load_leaf_assignments(leaf_path)
    L       = get_leaf_matrix(leaf_df)
    print(f"  Shape: {L.shape}  (n_samples × n_trees)")

    # Demo: 5×5 sub-kernel on first 5 samples
    K_demo = leaf_agreement_kernel(L[:5], L[:5])
    print("\nLeaf-agreement kernel (first 5 samples × first 5 samples):")
    print(np.round(K_demo, 3))

    # Demo: top-3 neighbors for first query sample
    meta_cols = [c for c in leaf_df.columns if c in META_COLS]
    train_meta = leaf_df[meta_cols].reset_index(drop=True)
    neighbors = top_k_neighbors(L[:1], L, train_meta, k=3)
    print("\nTop-3 neighbors for sample 0:")
    print(neighbors[0])
