"""One table: GEARS, both CPA seeds, ridge and mean delta on the same knockdowns.

Claim 5 currently reports GEARS against ridge and a mean-delta floor on the 40
knockdowns held out from both. CPA was trained on the same split but has never
been placed in that table, so the deep models cannot be compared with each other
or against the baselines on a common footing.

This extends `gears_on_project_scale.py` rather than replacing it: the same
score, the same centre -- the mean delta over knockdowns in neither test split,
so it is built from material no model is scored on -- and the same paired
bootstrap over knockdowns, with every model resampled on the same draw so the
differences are paired.

A model is included only if its prediction file exists; a missing CPA seed is
reported as missing rather than silently dropped, since an absent row and a
failed row look the same in a finished table.

Usage: deep_table.py OUT.json PRED_SPEC [PRED_SPEC ...]
  where PRED_SPEC is LABEL=/path/to/pred.npz
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

DRAWS = 2000
out_p, specs = sys.argv[1], sys.argv[2:]


def score(pred, truth, center):
    d, h1, h2 = truth
    if center is not None:
        pred, d, h1, h2 = pred - center, d - center, h1 - center, h2 - center
    pp, tt = (pred * pred).sum(), (h1 * h2).sum()
    return float((pred * d).sum() / np.sqrt(pp * tt)) if pp > 1e-12 and tt > 0 else 0.0


ctx = ctxlib.from_dataset("k562")
paths = Paths.for_dataset("k562")
data = PerturbationDataset.load(paths)
ridge = pickle.loads((paths.models_dir / "ridge.pkl").read_bytes())
ctx_targets = list(ctx.targets)
proj_test, proj_val = set(data.targets("test")), set(data.targets("val"))

found, missing = {}, []
for spec in specs:
    label, _, path = spec.partition("=")
    if pathlib.Path(path).exists():
        with np.load(path, allow_pickle=False) as z:
            found[label] = (z["delta"], [str(t) for t in z["targets"]], [str(g) for g in z["genes"]])
    else:
        missing.append((label, path))
if not found:
    sys.exit("no prediction file exists yet")
for label, path in missing:
    print(f"  MISSING {label}: {path}", flush=True)

# Score every model on the knockdowns all of them hold out, so one model with a
# smaller test set does not get an easier or merely different set.
common = sorted(set.intersection(*(set(t) for _, t, _ in found.values())) & proj_test & set(ctx_targets))
genes = [g for g in next(iter(found.values()))[2]]
for _, _, g in found.values():
    assert g == genes, "prediction files disagree on gene order"
print(f"  {len(found)} models, {len(common)} knockdowns held out from all, {len(genes)} genes", flush=True)
if len(common) < 8:
    sys.exit("too few shared knockdowns to compare")

gene_pos = {g: i for i, g in enumerate(ctx.genes)}
cols = np.array([gene_pos[g] for g in genes])
rows = np.array([ctx_targets.index(t) for t in common])
truth = [M[np.ix_(rows, cols)].astype(np.float64) for M in (ctx.delta, ctx.half1, ctx.half2)]

held = proj_test | proj_val | set().union(*(set(t) for _, t, _ in found.values()))
train_rows = np.array([i for i, t in enumerate(ctx_targets) if t not in held])
center = ctx.delta[np.ix_(train_rows, cols)].astype(np.float64).mean(0)
print(f"  centre: mean delta over {len(train_rows)} knockdowns in no test split", flush=True)

preds = {lab: d[[t.index(k) for k in common]].astype(np.float64) for lab, (d, t, _) in found.items()}
preds["ridge"] = ridge.predict(data, common)[:, cols].astype(np.float64)
preds["mean delta"] = np.repeat(center[None], len(common), 0)

rng = np.random.default_rng(0)
boots = {k: {"overall": [], "specific": []} for k in preds}
for _ in range(DRAWS):
    pick = rng.integers(0, len(common), len(common))
    t_b = [M[pick] for M in truth]
    for k, P in preds.items():
        boots[k]["overall"].append(score(P[pick], t_b, None))
        boots[k]["specific"].append(score(P[pick], t_b, center))

models = {}
for k, P in preds.items():
    models[k] = {m: score(P, truth, center if m == "specific" else None) for m in ("overall", "specific")}
    for m in ("overall", "specific"):
        models[k][f"{m}_ci"] = np.percentile(boots[k][m], [2.5, 97.5]).tolist()

print()
print(f"  {'model':14}{'overall':>10}{'95% CI':>20}{'specific':>10}{'95% CI':>20}")
for k in sorted(models, key=lambda x: -models[x]["specific"]):
    v = models[k]
    o_ci = f"[{v['overall_ci'][0]:+.3f}, {v['overall_ci'][1]:+.3f}]"
    s_ci = f"[{v['specific_ci'][0]:+.3f}, {v['specific_ci'][1]:+.3f}]"
    zero = " *" if not (v["specific_ci"][0] < 0 < v["specific_ci"][1]) else "  "
    print(f"  {k:14}{v['overall']:>+10.4f}{o_ci:>20}{v['specific']:>+10.4f}{s_ci:>20}{zero}")
print("  * specific CI excludes zero")

# Paired differences that the claim rests on, plus seed-to-seed spread.
pairs = {}
for a, b in [(x, y) for x in preds for y in preds if x < y]:
    d = np.array(boots[a]["specific"]) - np.array(boots[b]["specific"])
    pairs[f"{a} - {b}"] = {"point": models[a]["specific"] - models[b]["specific"],
                           "ci": np.percentile(d, [2.5, 97.5]).tolist()}
print()
print("  paired differences on specific (95% bootstrap):")
for k, v in sorted(pairs.items(), key=lambda kv: -abs(kv[1]["point"]))[:8]:
    star = "" if v["ci"][0] < 0 < v["ci"][1] else "  excludes zero"
    print(f"    {k:28}{v['point']:+.4f}  [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]{star}")

seeds = sorted(k for k in models if k.lower().startswith("cpa"))
if len(seeds) > 1:
    sp = abs(models[seeds[0]]["specific"] - models[seeds[1]]["specific"])
    print(f"\n  CPA seed-to-seed spread on specific: {sp:.4f}")

json.dump({"n_knockdowns": len(common), "n_genes": len(cols), "draws": DRAWS,
           "knockdowns": common, "models": models, "pairs": pairs,
           "missing": [l for l, _ in missing]}, open(out_p, "w"), indent=1)
print(f"\n  wrote {out_p}")
print("DEEP_TABLE_DONE")
