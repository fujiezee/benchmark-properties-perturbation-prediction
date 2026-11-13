"""Is h1's method-4-to-5 gain real, on h1's 23 perturbations?

The leaderboard moved h1 ridge from -0.0307 to +0.0269. Against the cached
aggregate spreads, eight of nine metrics had a difference-to-SE ratio below 1.96
and `pds_cosine` sat at 0.30 -- but those SEs are unpaired, and the two methods
score the same 23 perturbations, so the paired interval is the one that decides.

Both methods are the same code at a different `ALIGN_MAX_CELLS`: method 5 aligns
a target's emitted draw to the control mean only when the draw is at most 256
cells, method 4 aligns every draw. So this evaluates the one fitted ridge twice,
once per threshold, and pairs the per-target rows.

A paired bootstrap over perturbations, not a t-test: n is 23 and the per-target
metrics are bounded and skewed.

Usage: h1_paired.py [out.json]
"""

import pathlib as _p, sys as _s
_h = _p.Path(__file__).resolve().parent
_s.path[:0] = [str(_h), str(_h / "lib")]

import json
import pathlib
import sys

import pickle

import numpy as np
import polars as pl

from vcell import evaluate as ev
from vcell.config import Paths
from vcell.data.dataset import PerturbationDataset

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "h1_paired.json")
DRAWS = 2000
paths = Paths.for_dataset("h1")
data = PerturbationDataset.load(paths)
model = pickle.loads((paths.models_dir / "ridge.pkl").read_bytes())

targets = list(data.targets(ev.SPLIT))
print(f"  h1 {ev.SPLIT}: {len(targets)} perturbations", flush=True)

# `evaluate_model` writes to paths.eval_dir / model.name, so each threshold runs
# under its own name and neither overwrites the shipped `ridge` directory.
original_name = model.name
per, scores = {}, {}
for label, cap in (("align_all", 10**9), ("align_small", 256)):
    ev.ALIGN_MAX_CELLS = cap
    model.name = f"paired-{label}"
    scores[label] = ev.evaluate_model(data, model)
    per[label] = pl.read_csv(data.paths.eval_dir / model.name / "per_target.csv")
    print(f"  {label}: ALIGN_MAX_CELLS={cap}, score {scores[label]['score']:+.4f}", flush=True)
model.name = original_name

a, b = per["align_all"], per["align_small"]
# per_target.csv is long, one row per (perturbation, metric), so the metric has
# to be pivoted out. Pooling the raw `value` column across metrics averages
# quantities on different scales and produces a number that means nothing.
num = [c for c in a.columns if a[c].dtype in (pl.Float64, pl.Float32)]
txt = [c for c in a.columns if c not in num]
val = "value" if "value" in num else num[0]
met = "metric" if "metric" in txt else min(txt, key=lambda c: a[c].n_unique())
key = next(c for c in txt if c != met)
print(f"  per_target.csv schema: id={key!r} metric={met!r} value={val!r}, "
      f"{len(a)} rows, {a[met].n_unique()} metrics", flush=True)

rng = np.random.default_rng(0)
rows = []
for m in sorted(a[met].unique().to_list()):
    xa = a.filter(pl.col(met) == m).select([key, val]).sort(key)
    xb = b.filter(pl.col(met) == m).select([key, val]).sort(key)
    shared = sorted(set(xa[key].to_list()) & set(xb[key].to_list()))
    xa = xa.filter(pl.col(key).is_in(shared)).sort(key)
    xb = xb.filter(pl.col(key).is_in(shared)).sort(key)
    assert xa[key].to_list() == xb[key].to_list(), f"{m}: rows do not line up"
    x = np.asarray(xa[val], float)
    y = np.asarray(xb[val], float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 4:
        continue
    d = y - x
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(DRAWS)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    rows.append({"metric": m, "n": int(len(x)), "align_all": float(x.mean()),
                 "align_small": float(y.mean()), "difference": float(d.mean()),
                 "ci_low": float(lo), "ci_high": float(hi),
                 "excludes_zero": bool(lo > 0 or hi < 0),
                 "positive_fraction": float((boots > 0).mean())})

rows.sort(key=lambda r: -abs(r["difference"] / max(r["ci_high"] - r["ci_low"], 1e-12)))
print(f"\n  paired bootstrap, {DRAWS} draws over {rows[0]['n']} perturbations\n")
print(f"  {'metric':40}{'all':>9}{'<=256':>9}{'diff':>9}{'95% CI':>22}")
for r in rows:
    flag = "excludes 0" if r["excludes_zero"] else ""
    print(f"  {r['metric']:40}{r['align_all']:>9.4f}{r['align_small']:>9.4f}{r['difference']:>+9.4f}"
          f"  [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  {flag}")
n_sig = sum(r["excludes_zero"] for r in rows)
print(f"\n  {n_sig} of {len(rows)} metrics have an interval excluding zero")
OUT.write_text(json.dumps({"draws": DRAWS, "composite": {k: v["score"] for k, v in scores.items()},
                           "metrics": rows}, indent=1))
print(f"  wrote {OUT}")
print("H1_PAIRED_DONE")
