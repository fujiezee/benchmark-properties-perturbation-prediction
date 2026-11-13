"""Score the GEARS prediction against the held-out truth, and against the ceiling.

A cosine on its own says little here: the test knockdowns are measured from a few
dozen cells each, so the observed delta carries real noise and no predictor can
exceed what the measurement reproduces about itself. The split-half reliability
of the truth is that limit, so this reports both and the fraction attained.

Truth and halves come from the same CSR that GEARS trained on, so the comparison
is in one space. Each condition's cells are split in two by a seeded shuffle, and
each half is referenced to its own half of the controls -- the same construction
as `metrics.reliability`, which is `2r/(1+r)` on the clipped row cosine.

Usage: gears_eval.py <parts.npz> <pred.npz> <split.json>
"""

import json
import pathlib
import sys

import numpy as np
import scipy.sparse as sp


def rows_cos(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12)


def reliability(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    r = np.clip(rows_cos(h1, h2), 0, None)
    return 2 * r / (1 + r)


parts, pred_path, split_path = (pathlib.Path(p) for p in sys.argv[1:4])
with np.load(parts, allow_pickle=False) as z:
    X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    cond, genes = z["condition"], z["genes"]
with np.load(pred_path, allow_pickle=False) as z:
    pred, p_targets, p_genes = z["delta"], z["targets"], z["genes"]
if not np.array_equal(genes, p_genes):
    sys.exit("gene order differs between the parts file and the prediction")
print(f"  {X.shape[0]:,} cells x {X.shape[1]} genes, {len(p_targets)} predicted knockdowns")

rng = np.random.default_rng(0)


def halves(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.flatnonzero(mask)
    rng.shuffle(idx)
    return idx[: len(idx) // 2], idx[len(idx) // 2 :]


def mean_rows(idx: np.ndarray) -> np.ndarray:
    return np.asarray(X[idx].mean(0)).ravel() if len(idx) else np.zeros(X.shape[1])


ctrl_mask = cond == "ctrl"
ctrl_full = mean_rows(np.flatnonzero(ctrl_mask))
c1, c2 = halves(ctrl_mask)
ctrl_h1, ctrl_h2 = mean_rows(c1), mean_rows(c2)
print(f"  controls {int(ctrl_mask.sum()):,} cells, halves {len(c1):,} / {len(c2):,}")

# GEARS writes "<gene>" while the matrix labels a condition "<gene>+ctrl"
by_cond = {c: i for i, c in enumerate(np.unique(cond))}
truth, h1s, h2s, kept, n_cells = [], [], [], [], []
for t in p_targets:
    key = f"{t}+ctrl" if f"{t}+ctrl" in by_cond else t
    if key not in by_cond:
        continue
    m = cond == key
    if m.sum() < 4:
        continue
    a, b = halves(m)
    truth.append(mean_rows(np.flatnonzero(m)) - ctrl_full)
    h1s.append(mean_rows(a) - ctrl_h1)
    h2s.append(mean_rows(b) - ctrl_h2)
    kept.append(t)
    n_cells.append(int(m.sum()))

T = np.array(truth, np.float64)
P = pred[[list(p_targets).index(t) for t in kept]].astype(np.float64)
R = reliability(np.array(h1s, np.float64), np.array(h2s, np.float64))
C = rows_cos(P, T)
print(f"  scored {len(kept)} knockdowns, median {int(np.median(n_cells))} cells each")

# A model's score means nothing without what a model-free predictor gets. The
# mean training delta ignores which gene was knocked down, so whatever it scores
# is the part of the cosine that comes from every knockdown sharing a direction.
split = json.loads(split_path.read_text())
train_names = {c.split("+")[0] for c in split["train"]}
train_rows = []
for t in train_names:
    key = f"{t}+ctrl" if f"{t}+ctrl" in by_cond else t
    if key in by_cond:
        m = cond == key
        if m.sum() >= 4:
            train_rows.append(mean_rows(np.flatnonzero(m)) - ctrl_full)
mean_delta = np.mean(train_rows, 0) if train_rows else np.zeros(X.shape[1])
M = rows_cos(np.repeat(mean_delta[None], len(kept), 0), T)
print(f"  mean-training-delta baseline from {len(train_rows)} train knockdowns")
print()
print(f"  {'quantity':34}{'median':>10}{'mean':>10}")
print(f"  {'GEARS cosine to truth':34}{np.median(C):>10.4f}{C.mean():>10.4f}")
print(f"  {'mean training delta, same truth':34}{np.median(M):>10.4f}{M.mean():>10.4f}")
print(f"  {'truth split-half reliability':34}{np.median(R):>10.4f}{R.mean():>10.4f}")
ok = R > 0.1
print(f"  {'fraction of the ceiling attained':34}{np.median(C[ok] / R[ok]):>10.4f}"
      f"{(C[ok] / R[ok]).mean():>10.4f}   (on {int(ok.sum())} with reliability > 0.1)")
print(f"  {'  same, for the mean baseline':34}{np.median(M[ok] / R[ok]):>10.4f}"
      f"{(M[ok] / R[ok]).mean():>10.4f}")
print(f"  {'GEARS minus mean baseline':34}{np.median(C - M):>10.4f}{(C - M).mean():>10.4f}")
print()
print("  by reliability band")
print(f"    {'band':14}{'n':>5}{'reliability':>13}{'GEARS':>9}{'mean delta':>12}{'GEARS - mean':>14}")
for lo, hi in ((0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 1.01)):
    m = (R >= lo) & (R < hi)
    if m.sum():
        print(f"    [{lo:.1f}, {hi:.1f}){'':4}{int(m.sum()):>5}{np.median(R[m]):>13.4f}"
              f"{np.median(C[m]):>9.4f}{np.median(M[m]):>12.4f}{np.median(C[m] - M[m]):>14.4f}")
print()
print("GEARS_EVAL_DONE")
