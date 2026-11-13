"""Response tables with halves, and the pooled similarity between two of them.

A context is (genes, targets, delta, half1, half2): mean response per target
against controls, and the same from two random halves of the cells (controls
split too). Similarity of two contexts over a set of knockdowns is the pooled
ratio of sums, sum <A, B> / sqrt(sum <A1, A2> sum <B1, B2>): the halves' inner
product is unbiased for a response's squared norm, and summing before dividing
keeps the estimate consistent at low signal. The knockdown's own gene is
excluded.
"""

import pathlib as _p, sys as _s
_h = _p.Path(__file__).resolve().parent
_s.path[:0] = [str(_h), str(_h / "lib")]

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

SEED = 0


@dataclass
class Context:
    name: str
    genes: np.ndarray
    targets: list[str]
    delta: np.ndarray
    half1: np.ndarray
    half2: np.ndarray


def from_npz(name: str, path: str) -> Context:
    z = np.load(path)
    return Context(name, z["genes"], z["targets"].tolist(), z["delta"], z["half1"], z["half2"])


def from_dataset(name: str) -> Context:
    from vcell.config import Paths
    from vcell.data.dataset import TARGET_COL, PerturbationDataset

    data = PerturbationDataset.load(Paths.for_dataset(name))
    X = data.adata.X.tocsr()
    labels = data.adata.obs[TARGET_COL].astype(str).to_numpy()
    targets = [t for t in data.perturbations.index if t != data.control_label]
    half = np.random.default_rng(SEED).integers(0, 2, len(labels))
    index = {g: i for i, g in enumerate([data.control_label, *targets])}
    rows = np.array([index.get(t, -1) for t in labels])
    keep = np.flatnonzero(rows >= 0)
    k = len(index)
    onehot = (
        sp.csr_matrix((np.ones(len(keep)), (rows[keep], keep)), shape=(3 * k, len(labels)))
        + sp.csr_matrix((np.ones(len(keep)), (rows[keep] + k + k * half[keep], keep)), shape=(3 * k, len(labels)))
    ).tocsr()
    n = np.asarray(onehot.sum(1)).ravel()
    m = np.asarray((onehot @ X).todense()) / np.maximum(n, 1)[:, None]
    full, h1, h2 = m[:k], m[k : 2 * k], m[2 * k :]
    return Context(name, np.asarray(data.genes), targets, full[1:] - full[0], h1[1:] - h1[0], h2[1:] - h2[0])


def restrict(c: Context, genes: np.ndarray) -> Context:
    gi = {g: j for j, g in enumerate(c.genes)}
    cols = [gi[g] for g in genes]
    col = {g: j for j, g in enumerate(genes)}
    out = []
    for M in (c.delta, c.half1, c.half2):
        M = np.nan_to_num(M[:, cols].astype(np.float64))
        for i, t in enumerate(c.targets):
            if (j := col.get(t)) is not None:
                M[i, j] = 0.0
        out.append(M)
    return Context(c.name, genes, c.targets, *out)


def shared_genes(contexts: list[Context]) -> np.ndarray:
    return np.array(sorted(set.intersection(*(set(c.genes.tolist()) for c in contexts))))


def per_knockdown(a: Context, b: Context, knockdowns: list[str]) -> dict[str, np.ndarray]:
    """Inner product and each side's signal power per knockdown (rows in the given order)."""
    ia = {t: i for i, t in enumerate(a.targets)}
    ib = {t: i for i, t in enumerate(b.targets)}
    ra = [ia[t] for t in knockdowns]
    rb = [ib[t] for t in knockdowns]
    return {
        "dot": (a.delta[ra] * b.delta[rb]).sum(1),
        "power_a": (a.half1[ra] * a.half2[ra]).sum(1),
        "power_b": (b.half1[rb] * b.half2[rb]).sum(1),
    }


def pooled(dot: np.ndarray, power_a: np.ndarray, power_b: np.ndarray) -> float:
    pa, pb = power_a.sum(), power_b.sum()
    return float(dot.sum() / np.sqrt(pa * pb)) if pa > 0 and pb > 0 else float("nan")


def scale_genes(c: Context) -> Context:
    """Divide each gene by its spread across knockdowns, so how strongly a screen measures a gene does not weight it."""
    sd = c.delta.std(axis=0) + 1e-6
    return Context(c.name, c.genes, c.targets, c.delta / sd, c.half1 / sd, c.half2 / sd)
