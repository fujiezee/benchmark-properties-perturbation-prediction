"""A scPerturb h5ad (Zenodo 7041849) as a response table.

Cells with one perturbation; target gene = first match of GENE_REGEX on the
perturbation label (group 1); controls = labels matching CONTROL_REGEX. Mean
log1p(CP10k) per gene minus controls, all cells and two random halves
(controls split too). X must be raw counts.

--ignore-nperts: trust the label alone (Adamson 2016 gives nearly every
single-construct cell nperts = 2).

Usage: python scperturb_npz.py H5AD OUT_NPZ CONTROL_REGEX GENE_REGEX [MIN_CELLS] [--ignore-nperts]
"""

import re
import sys

import anndata as ad
import numpy as np
import scipy.sparse as sp

SEED = 0
args = [x for x in sys.argv[1:] if x != "--ignore-nperts"]
path, out, control_re, gene_re = args[:4]
MIN_CELLS = int(args[4]) if len(args) > 4 else 30

a = ad.read_h5ad(path)
labels = np.array([str(x) for x in a.obs["perturbation"]])  # missing labels become "nan"
single = (a.obs["nperts"].astype(int).to_numpy() <= 1 if "nperts" in a.obs and "--ignore-nperts" not in sys.argv
          else np.ones(len(labels), bool))
is_control = np.array([bool(re.fullmatch(control_re, x)) for x in labels])
gene = np.array([(m.group(1) if (m := re.match(gene_re, x)) else "") for x in labels], dtype=object)
gene[is_control] = "__control__"
keep = single & (gene != "")
X = sp.csr_matrix(a.X[keep], dtype=np.float32)
assert np.allclose(X.data[:1000], np.round(X.data[:1000])), "X is not raw counts"
gene = gene[keep]
lib = np.asarray(X.sum(1)).ravel()
X = sp.diags(1e4 / lib) @ X
X.data = np.log1p(X.data)
X = X.tocsr()
counts = {g: (gene == g).sum() for g in np.unique(gene)}
targets = sorted(g for g, n in counts.items() if g != "__control__" and n >= MIN_CELLS)
groups = ["__control__", *targets]
index = {g: i for i, g in enumerate(groups)}
row = np.array([index.get(g, -1) for g in gene])
ok = np.flatnonzero(row >= 0)
half = np.random.default_rng(SEED).integers(0, 2, len(gene))
k = len(groups)
G = sp.csr_matrix((np.ones(2 * len(ok)), (np.r_[row[ok], k + row[ok] + k * half[ok]], np.r_[ok, ok])), shape=(3 * k, len(gene)))
n = np.asarray(G.sum(1)).ravel()
M = np.asarray((G @ X).todense()) / np.maximum(n, 1)[:, None]
full, h1, h2 = M[:k], M[k:2 * k], M[2 * k:]
np.savez_compressed(out, genes=np.asarray(a.var_names).astype(str), targets=np.array(targets),
                    delta=(full[1:] - full[0]).astype(np.float32), half1=(h1[1:] - h1[0]).astype(np.float32),
                    half2=(h2[1:] - h2[0]).astype(np.float32), cells=n[1:k].astype(np.int32))
print(f"wrote {out}: {len(targets)} targets (>= {MIN_CELLS} cells, median {int(np.median(n[1:k]))}), {int(n[0])} control cells, {X.shape[1]} genes")
