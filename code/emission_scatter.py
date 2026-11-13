"""What the official scorer's #348 warning sees, and the question behind it.

`cell_eval2` warns that our submission's "per-cell scatter largely CANCELS in the
pseudobulk, which is not what an i.i.d. sample of a predicted cell population
looks like". Its guard computes `ratio = budget / claim` -- across-perturbation
spread over the claimed sampling correction -- multiplies the correction by that
ratio whenever it is under 1, and warns under 0.7. So the scorer already
withholds the correction: at the 0.1742 measured on h1 it granted 17% of what
was claimed. The flag costs us; it does not pay us.

Two things a reviewer asks next, measured here.

1. The mechanism. `emit_counts` draws control cells with replacement and, at or
   below `ALIGN_MAX_CELLS`, rescales the draw so its mean is the control mean.
   Per-cell scatter stays real while the aggregate's own error is removed by
   construction -- the non-exchangeability the guard detects. Measured as the
   emitted pseudobulk's distance from its intended mean over the standard error
   an i.i.d. sample of the same cells would have. An honest sample gives ~1.

2. The larger question. The emitted cells are control cells rescaled, so their
   per-gene variance is the control's times a fold change, not the perturbed
   population's. If a knockdown changes how heterogeneous the cells are, nothing
   in the emission reproduces it. Measured as emitted over real per-gene
   variance; 1 means as heterogeneous as the real cells.

The emission block is copied verbatim from `vcell.evaluate.emit_counts`, which
cannot be imported here: it pulls in `cell_eval2`, the official scorer, which is
not installed on this host.

Usage: emission_scatter.py <dataset> [n_targets] [out.json]
"""

import json
import pathlib
import pickle
import sys

import anndata as ad
import numpy as np
import scipy.sparse as sp

import pathlib as _p, sys as _s
_h = _p.Path(__file__).resolve().parent
_s.path[:0] = [str(_h), str(_h / "lib")]

from vcell.config import Paths
from vcell.data.dataset import TARGET_COL, PerturbationDataset

ALIGN_MAX_CELLS = 256  # as in vcell.evaluate
SPLIT = "test"
NAME = sys.argv[1] if len(sys.argv) > 1 else "k562"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
OUT = pathlib.Path(sys.argv[3] if len(sys.argv) > 3 else f"emission_{NAME}.json")

paths = Paths.for_dataset(NAME)
data = PerturbationDataset.load(paths)
model = pickle.loads((paths.models_dir / "ridge.pkl").read_bytes())
targets = list(data.targets(SPLIT))
real = ad.read_h5ad(paths.eval_dir / "_real" / SPLIT / "real.h5ad")
deltas = model.predict(data, targets)
print(f"  {NAME}: {len(targets)} test knockdowns, {real.n_obs:,} cells, "
      f"ALIGN_MAX_CELLS={ALIGN_MAX_CELLS}", flush=True)

# --- verbatim from emit_counts, with the per-target measurement folded in ---
rng = np.random.default_rng(0)
labels = real.obs[TARGET_COL].to_numpy()
control_rows = np.flatnonzero(labels == data.control_label)
control = sp.csr_matrix(real.X[control_rows], dtype=np.float64)
before = np.expm1(data.control_mean.astype(np.float64))
after = np.clip(np.expm1(data.control_mean + deltas.astype(np.float64)), 0.0, None)
scale = np.divide(after, before, out=np.ones_like(after), where=before > 0)
control_mean = np.asarray(control.mean(axis=0)).ravel()

want = set(np.random.default_rng(0).choice(len(targets), size=min(N, len(targets)), replace=False))
rows = []
for ti, (target, gene_scale) in enumerate(zip(targets, scale, strict=True)):
    own = np.flatnonzero(labels == target)
    drawn = control[rng.integers(0, control.shape[0], len(own))]
    if len(own) <= ALIGN_MAX_CELLS:
        drawn_mean = np.asarray(drawn.mean(axis=0)).ravel()
        to_mean = np.divide(control_mean, drawn_mean, out=np.ones_like(drawn_mean), where=drawn_mean > 0)
    else:
        to_mean = np.ones_like(control_mean)
    cells = drawn.multiply((gene_scale * to_mean)[None, :]).tocsr()
    floor = np.floor(cells.data)
    cells.data = floor + (rng.random(cells.data.size) < cells.data - floor)
    cells.eliminate_zeros()
    # --- end verbatim ---
    if ti not in want or len(own) < 10:
        continue
    R = sp.csr_matrix(real.X[own]).astype(np.float64)
    p_mean = np.asarray(cells.mean(axis=0)).ravel()
    r_mean = np.asarray(R.mean(axis=0)).ravel()
    p_var = np.asarray(cells.multiply(cells).mean(axis=0)).ravel() - p_mean ** 2
    r_var = np.asarray(R.multiply(R).mean(axis=0)).ravel() - r_mean ** 2
    intended = control_mean * gene_scale
    iid_se = np.sqrt(np.maximum(p_var, 0) / len(own))
    agg_err = np.abs(p_mean - intended)
    ok = iid_se > 1e-9
    vok = r_var > 1e-9
    rows.append({"target": str(target), "n_cells": int(len(own)),
                 "aligned": bool(len(own) <= ALIGN_MAX_CELLS),
                 "agg_err_over_iid_se": float(np.median(agg_err[ok] / iid_se[ok])),
                 "var_ratio": float(np.median(p_var[vok] / r_var[vok]))})

al = [r for r in rows if r["aligned"]]
un = [r for r in rows if not r["aligned"]]
out = {"dataset": NAME, "align_max_cells": ALIGN_MAX_CELLS, "n_scored": len(rows),
       "n_aligned": len(al), "n_unaligned": len(un), "rows": rows}
print()
print(f"  {'group':12}{'n':>4}{'median cells':>14}{'agg err / iid SE':>19}{'pred var / real var':>22}")
for lab, grp in (("aligned", al), ("unaligned", un)):
    if grp:
        a = float(np.median([r["agg_err_over_iid_se"] for r in grp]))
        v = float(np.median([r["var_ratio"] for r in grp]))
        print(f"  {lab:12}{len(grp):>4}{int(np.median([r['n_cells'] for r in grp])):>14}{a:>19.4f}{v:>22.4f}")
        out[lab] = {"agg_err_over_iid_se": a, "var_ratio": v}
print()
print("  agg err / iid SE -- 1.0 is an i.i.d. sample; below 1 is what #348 detects")
print("  pred var / real var -- 1.0 is as heterogeneous as the real perturbed cells")
OUT.write_text(json.dumps(out, indent=1))
print(f"  wrote {OUT}")
print("EMISSION_SCATTER_DONE")
