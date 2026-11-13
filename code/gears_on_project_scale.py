"""Score GEARS on the project's own scale, against ridge, on knockdowns neither saw.

GEARS emits a mean delta, not a cell population, so it cannot go through
`evaluate.py`: the leaderboard metric scores emitted cells, and inventing a noise
model to turn GEARS's mean into cells would score my invention, not GEARS. What
GEARS can be scored on is the other scale this project already uses, from
`scripts039/models.py`:

    overall  = sum <pred, truth> / sqrt(sum |pred|^2 * sum <half1, half2>)
    specific = the same after subtracting a centre from pred and truth

The denominator takes the split-half inner product in place of |truth|^2, so
target noise is corrected rather than charged to the model. `specific` removes
the average response, leaving what a model says about *this* knockdown. The
`score` function below is copied verbatim from models.py so both sit on one
definition.

The comparison set is the intersection of GEARS's test split and the project's
test split: knockdowns held out from both models. Columns are GEARS's 2,000
genes, all of which exist in the project's 8,563, so both models are scored on
identical columns. The centre is the mean delta over knockdowns in neither test
split, so it is built from material neither model is scored on.

Usage: gears_on_project_scale.py <gears_pred.npz> [out.json]
"""

import json
import pathlib
import pickle
import sys

import numpy as np

import pathlib as _p, sys as _s
_h = _p.Path(__file__).resolve().parent
_s.path[:0] = [str(_h), str(_h / "lib")]

import ctxlib
from vcell.config import Paths
from vcell.data.dataset import PerturbationDataset

PRED = pathlib.Path(sys.argv[1])
OUT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "gears_on_project_scale.json")
DRAWS = 2000


def score(pred: np.ndarray, truth: list[np.ndarray], center: np.ndarray | None) -> float:
    """Verbatim from scripts039/models.py."""
    d, h1, h2 = truth
    if center is not None:
        pred, d, h1, h2 = pred - center, d - center, h1 - center, h2 - center
    pp, tt = (pred * pred).sum(), (h1 * h2).sum()
    return float((pred * d).sum() / np.sqrt(pp * tt)) if pp > 1e-12 and tt > 0 else 0.0


ctx = ctxlib.from_dataset("k562")
paths = Paths.for_dataset("k562")
data = PerturbationDataset.load(paths)
ridge = pickle.loads((paths.models_dir / "ridge.pkl").read_bytes())

with np.load(PRED, allow_pickle=False) as z:
    g_delta, g_targets, g_genes = z["delta"], list(z["targets"]), list(z["genes"])

ctx_targets = list(ctx.targets)
proj_test = set(data.targets("test"))
proj_val = set(data.targets("val"))
common = sorted(set(g_targets) & proj_test & set(ctx_targets))
print(f"  project k562: {len(ctx_targets)} knockdowns, {len(ctx.genes)} genes")
print(f"  GEARS: {len(g_targets)} test knockdowns, {len(g_genes)} genes")
print(f"  held out from both: {len(common)} knockdowns")
if len(common) < 8:
    sys.exit("too few shared knockdowns to compare")

gene_pos = {g: i for i, g in enumerate(ctx.genes)}
cols = np.array([gene_pos[g] for g in g_genes])
rows = np.array([ctx_targets.index(t) for t in common])
truth = [M[np.ix_(rows, cols)].astype(np.float64) for M in (ctx.delta, ctx.half1, ctx.half2)]

# Centre from knockdowns in neither test split, so it is built from material
# neither model is scored on.
held = proj_test | proj_val | set(g_targets)
train_rows = np.array([i for i, t in enumerate(ctx_targets) if t not in held])
center = ctx.delta[np.ix_(train_rows, cols)].astype(np.float64).mean(0)
print(f"  centre: mean delta over {len(train_rows)} knockdowns in neither test split")

g_pos = {t: i for i, t in enumerate(g_targets)}
preds = {
    "GEARS": g_delta[[g_pos[t] for t in common]].astype(np.float64),
    "ridge": ridge.predict(data, common)[:, cols].astype(np.float64),
    "mean delta": np.repeat(center[None], len(common), 0),
}

rng = np.random.default_rng(0)
boots = {k: {"overall": [], "specific": []} for k in preds}
for _ in range(DRAWS):
    pick = rng.integers(0, len(common), len(common))
    t_b = [M[pick] for M in truth]
    for k, P in preds.items():
        boots[k]["overall"].append(score(P[pick], t_b, None))
        boots[k]["specific"].append(score(P[pick], t_b, center))

out = {"n_knockdowns": len(common), "n_genes": len(cols), "draws": DRAWS,
       "knockdowns": common, "models": {}}
print()
print(f"  {'model':14}{'overall':>10}{'95% CI':>22}{'specific':>11}{'95% CI':>22}")
for k, P in preds.items():
    row = {}
    for which, center_arg in (("overall", None), ("specific", center)):
        point = score(P, truth, center_arg)
        lo, hi = np.percentile(boots[k][which], [2.5, 97.5])
        row[which] = point
        row[which + "_ci"] = [float(lo), float(hi)]
    out["models"][k] = row
    print(f"  {k:14}{row['overall']:>10.4f}  [{row['overall_ci'][0]:+.4f}, {row['overall_ci'][1]:+.4f}]"
          f"{row['specific']:>11.4f}  [{row['specific_ci'][0]:+.4f}, {row['specific_ci'][1]:+.4f}]")

d_boot = np.array(boots["ridge"]["specific"]) - np.array(boots["GEARS"]["specific"])
lo, hi = np.percentile(d_boot, [2.5, 97.5])
out["ridge_minus_gears_specific"] = {"point": out["models"]["ridge"]["specific"] - out["models"]["GEARS"]["specific"],
                                     "ci": [float(lo), float(hi)]}
print()
print(f"  ridge - GEARS on specific: {out['ridge_minus_gears_specific']['point']:+.4f} "
      f"[{lo:+.4f}, {hi:+.4f}]")
OUT.write_text(json.dumps(out, indent=1))
print(f"  wrote {OUT}")
print("GEARS_SCALE_DONE")
