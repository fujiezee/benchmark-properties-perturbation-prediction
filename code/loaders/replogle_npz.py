"""A Replogle 2022 or Nadig 2025 raw single-cell h5ad (dense X) as a response table.

Mean log1p(CP10k) per target gene minus non-targeting cells, for all cells and
two random halves (controls split too). CP10k uses the cell's UMI_count over
the whole transcriptome. Reads X in row blocks, so a 66 GB file fits in 64 GB.

Usage: python replogle_npz.py H5AD OUT_NPZ
"""

import sys

import h5py
import numpy as np
import scipy.sparse as sp

SEED = 0
BLOCK = 20000
CONTROL = "non-targeting"

path, out = sys.argv[1], sys.argv[2]
f = h5py.File(path, "r")


def column(group: str, name: str) -> np.ndarray:
    node = f[group][name]
    if isinstance(node, h5py.Group):  # anndata >= 0.8 categorical
        values = node["categories"][:][node["codes"][:]]
    else:  # older layout: codes, with categories under __categories
        values = node[:]
        cats = f[group].get("__categories")
        if cats is not None and name in cats:
            values = cats[name][:][values]
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in values])


genes = column("var", "gene_name")
target = column("obs", "gene")
umi = f["obs"]["UMI_count"][:].astype(np.float64)
groups = [CONTROL, *sorted(set(target) - {CONTROL})]
index = {g: i for i, g in enumerate(groups)}
row = np.array([index[t] for t in target])
half = np.random.default_rng(SEED).integers(0, 2, len(target))
k, n_genes = len(groups), len(genes)
sums = np.zeros((3 * k, n_genes))
counts = np.bincount(row, minlength=k).astype(np.float64)
half_counts = np.bincount(row + k * half, minlength=2 * k).astype(np.float64)
X = f["X"]
for start in range(0, X.shape[0], BLOCK):
    stop = min(start + BLOCK, X.shape[0])
    block = np.log1p(X[start:stop].astype(np.float64) * (1e4 / umi[start:stop])[:, None])
    r, n = row[start:stop], stop - start
    cols = np.r_[np.arange(n), np.arange(n)]
    onehot = sp.csr_matrix((np.ones(2 * n), (np.r_[r, k + r + k * half[start:stop]], cols)), shape=(3 * k, n))
    sums += onehot @ block
    print(f"{stop}/{X.shape[0]}", flush=True) if (stop // BLOCK) % 10 == 0 else None
full = sums[:k] / np.maximum(counts, 1)[:, None]
halves = sums[k:] / np.maximum(half_counts, 1)[:, None]
h1, h2 = halves[:k], halves[k:]
np.savez_compressed(
    out, genes=genes, targets=np.array(groups[1:]),
    delta=(full[1:] - full[0]).astype(np.float32), half1=(h1[1:] - h1[0]).astype(np.float32),
    half2=(h2[1:] - h2[0]).astype(np.float32), cells=counts[1:].astype(np.int32))
print(f"wrote {out}: {k - 1} targets x {n_genes} genes, {int(counts[0])} control cells")
