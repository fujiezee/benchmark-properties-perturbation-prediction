"""Gasperini et al. 2019 (Shendure lab) at-scale K562 CRISPRi screen, TSS-targeting guides, as a response table.

High MOI (about 28 guides per cell), from scPerturb's h5ad (Zenodo 7041849; GEO's
FTP served this file corrupted and then flagged the box). For each gene with a
TSS-targeting guide group, mean log1p(CP10k) of cells carrying it minus cells
not carrying it; the other guides in a cell fall on both sides and average out.
CP10k uses total_umis (whole transcriptome). Halves split cells at random.

Usage: python gasperini_npz.py H5AD OUT_NPZ [MIN_CELLS]
"""

import re
import sys

import anndata as ad
import numpy as np
import scipy.sparse as sp

SEED = 0
path, out = sys.argv[1], sys.argv[2]
MIN_CELLS = int(sys.argv[3]) if len(sys.argv) > 3 else 50

a = ad.read_h5ad(path)
X = sp.csr_matrix(a.X, dtype=np.float32)
X = sp.diags((1e4 / a.obs["total_umis"].to_numpy()).astype(np.float32)) @ X
X.data = np.log1p(X.data)
X = X.tocsr()
tss = [re.findall(r"(?:^|_)([^_]+)_TSS(?=_|$)", s) for s in a.obs["gene"].astype(str)]
targets = sorted({g for t in tss for g in t})
col = {g: j for j, g in enumerate(targets)}
rows = np.repeat(np.arange(len(tss)), [len(t) for t in tss])
cols = np.array([col[g] for t in tss for g in t], dtype=np.int64)
M = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(tss), len(targets)))
M.data[:] = 1  # a gene listed twice in a cell still counts once
half = np.random.default_rng(SEED).integers(0, 2, len(tss))


def delta(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Mm = M[mask]
    Xm = X[mask]
    n_in = np.asarray(Mm.sum(0)).ravel()
    s_in = np.asarray((Mm.T @ Xm).todense())
    total = np.asarray(Xm.sum(0)).ravel()
    n_out = mask.sum() - n_in
    return s_in / np.maximum(n_in, 1)[:, None] - (total[None, :] - s_in) / np.maximum(n_out, 1)[:, None], n_in


full, n = delta(np.ones(len(tss), bool))
h1, _ = delta(half == 0)
h2, _ = delta(half == 1)
keep = n >= MIN_CELLS
np.savez_compressed(out, genes=np.asarray(a.var_names).astype(str), targets=np.array(targets)[keep],
                    delta=full[keep].astype(np.float32), half1=h1[keep].astype(np.float32),
                    half2=h2[keep].astype(np.float32), cells=n[keep].astype(np.int32))
print(f"wrote {out}: {keep.sum()} TSS targets (of {len(targets)}, >= {MIN_CELLS} cells; median {int(np.median(n[keep]))}) x {X.shape[1]} genes")
