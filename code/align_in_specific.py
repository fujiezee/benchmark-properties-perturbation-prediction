"""What the alignment choice is worth on the same scale the models are compared on.

The model-separation collapse is measured in `specific` (models.py's noise-
corrected cosine). The alignment effect was measured in leaderboard composite
units. Those are different scales, so the two numbers cannot be set side by side
until one is converted. This converts the alignment effect.

In delta space the choice has an exact form. `emit_counts` emits cells whose mean
is `draw_mean * gene_scale`; aligning replaces `draw_mean` with the control mean.
The delta recovered from the emitted cells is the log-ratio of that mean to the
control mean, so aligned it is

    log(control_mean * gene_scale) - log(control_mean) = log(gene_scale) = delta

the model's own delta, and unaligned it is `delta + eps` with

    eps = log1p(draw_mean) - log1p(control_mean)

the log-ratio of a bootstrap draw's mean to the control mean. Both terms are
count means, so `eps` stays inside one space -- taking it against `control_mean`
in log-normalised space instead mixes two and adds a large offset rather than
noise. `gene_scale` cancels and never enters. So the alignment choice is exactly
whether the submitted delta carries `eps`, and its value in `specific` units is
the difference between scoring `pred` and `pred + eps`. No cells are emitted.

`eps` is drawn the way `emit_counts` draws it: for each target, `n_i` control
cells with replacement, `n_i` being that target's real cell count.

`score` is copied verbatim from scripts039/models.py.

Usage: align_in_specific.py <dataset> [repeats] [out.json]
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

import ctxlib
from vcell.config import Paths
from vcell.data.dataset import TARGET_COL, PerturbationDataset

SPLIT = "test"
NAME = sys.argv[1] if len(sys.argv) > 1 else "k562"
R = int(sys.argv[2]) if len(sys.argv) > 2 else 20
OUT = pathlib.Path(sys.argv[3] if len(sys.argv) > 3 else f"align_specific_{NAME}.json")


def score(pred: np.ndarray, truth: list[np.ndarray], center: np.ndarray | None) -> float:
    """Verbatim from scripts039/models.py."""
    d, h1, h2 = truth
    if center is not None:
        pred, d, h1, h2 = pred - center, d - center, h1 - center, h2 - center
    pp, tt = (pred * pred).sum(), (h1 * h2).sum()
    return float((pred * d).sum() / np.sqrt(pp * tt)) if pp > 1e-12 and tt > 0 else 0.0


paths = Paths.for_dataset(NAME)
data = PerturbationDataset.load(paths)
ctx = ctxlib.from_dataset(NAME)
model = pickle.loads((paths.models_dir / "ridge.pkl").read_bytes())
real = ad.read_h5ad(paths.eval_dir / "_real" / SPLIT / "real.h5ad")

ctx_targets = list(ctx.targets)
targets = [t for t in data.targets(SPLIT) if t in ctx_targets]
deltas = model.predict(data, targets).astype(np.float64)
cm = data.control_mean.astype(np.float64)
if cm.shape[0] != ctx.delta.shape[1]:
    sys.exit(f"gene spaces differ: control_mean {cm.shape[0]} vs ctx {ctx.delta.shape[1]}")

labels = real.obs[TARGET_COL].to_numpy()
control = sp.csr_matrix(real.X[np.flatnonzero(labels == data.control_label)], dtype=np.float64)
control_mean = np.asarray(control.mean(axis=0)).ravel()
n_cells = np.array([int((labels == t).sum()) for t in targets])

rows = np.array([ctx_targets.index(t) for t in targets])
truth = [M[rows].astype(np.float64) for M in (ctx.delta, ctx.half1, ctx.half2)]
held = set(data.targets(SPLIT)) | set(data.targets("val"))
center = ctx.delta[[i for i, t in enumerate(ctx_targets) if t not in held]].astype(np.float64).mean(0)

print(f"  {NAME}: {len(targets)} test knockdowns, {ctx.delta.shape[1]} genes, "
      f"median {int(np.median(n_cells))} cells per knockdown", flush=True)

base = {"overall": score(deltas, truth, None), "specific": score(deltas, truth, center)}
rng = np.random.default_rng(0)
noisy = {"overall": [], "specific": []}
eps_scale = []
for _ in range(R):
    eps = np.empty_like(deltas)
    for i, n in enumerate(n_cells):
        drawn = control[rng.integers(0, control.shape[0], n)]
        draw_mean = np.asarray(drawn.mean(axis=0)).ravel()
        eps[i] = np.log1p(draw_mean) - np.log1p(control_mean)
    noisy["overall"].append(score(deltas + eps, truth, None))
    noisy["specific"].append(score(deltas + eps, truth, center))
    eps_scale.append((float(np.median(np.abs(eps))), float(eps.mean())))

out = {"dataset": NAME, "n_knockdowns": len(targets), "repeats": R,
       "median_cells": int(np.median(n_cells)), "aligned": base, "unaligned": {}}
# eps must look like sampling noise: centred near zero and small against the
# signal it is added to. Printed so the result can be checked rather than trusted.
me = float(np.mean([e[0] for e in eps_scale]))
mu = float(np.mean([e[1] for e in eps_scale]))
md = float(np.median(np.abs(deltas)))
out["eps"] = {"median_abs": me, "mean": mu, "median_abs_delta": md, "ratio": me / md}
print()
print(f"  eps: median |eps| {me:.4f}, mean {mu:+.5f}; median |delta| {md:.4f}; "
      f"ratio {me / md:.3f}")
print()
print(f"  {'scale':12}{'aligned':>10}{'unaligned':>22}{'alignment is worth':>22}")
for which in ("overall", "specific"):
    m, sd = float(np.mean(noisy[which])), float(np.std(noisy[which]))
    out["unaligned"][which] = {"mean": m, "sd": sd}
    out.setdefault("alignment_worth", {})[which] = base[which] - m
    print(f"  {which:12}{base[which]:>10.4f}{m:>14.4f} ± {sd:.4f}{base[which] - m:>22.4f}")
OUT.write_text(json.dumps(out, indent=1))
print(f"  wrote {OUT}")
print("ALIGN_SPECIFIC_DONE")
