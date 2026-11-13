"""Same lab, other screen: how far apart are two K562 screens from one lab?

Against one K562 reference, on the same knockdowns and genes, pooled
similarity (ctxlib) of
  batch  the same lab's other K562 screen
  lab    another lab's K562 screen
  line   the same lab's other lines (pooled)
Knockdowns are those in the reference, both K562 comparisons, and at least
one other line; bins by the reference's split-half reliability. Bootstrap over
knockdowns gives intervals for batch - line and lab - batch.

Usage: python baseline.py OUT_JSON REF BATCH LAB LINE [LINE ...]   (npz files)
"""

import json
import sys

import numpy as np

import pathlib as _p, sys as _s
_h = _p.Path(__file__).resolve().parent
_s.path[:0] = [str(_h), str(_h / "lib")]
import ctxlib

out, ref_p, batch_p, lab_p, *line_p = sys.argv[1:]
BINS = [(-1.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]
BOOT = 1000

ref = ctxlib.from_npz("ref", ref_p)
batch = ctxlib.from_npz("batch", batch_p)
lab = ctxlib.from_npz("lab", lab_p)
lines = [ctxlib.from_npz(p, p) for p in line_p]
genes = ctxlib.shared_genes([ref, batch, lab, *lines])
ref, batch, lab = (ctxlib.restrict(c, genes) for c in (ref, batch, lab))
lines = [ctxlib.restrict(c, genes) for c in lines]
kds = sorted(set(ref.targets) & set(batch.targets) & set(lab.targets) & set.union(*(set(c.targets) for c in lines)))
print(f"{len(genes)} genes, {len(kds)} knockdowns", flush=True)

ir = {t: i for i, t in enumerate(ref.targets)}
h1, h2 = ref.half1[[ir[t] for t in kds]], ref.half2[[ir[t] for t in kds]]
c = np.clip((h1 * h2).sum(1) / (np.linalg.norm(h1, axis=1) * np.linalg.norm(h2, axis=1) + 1e-12), 0, None)
rel = 2 * c / (1 + c)
pos = {t: i for i, t in enumerate(kds)}


def rows(other: ctxlib.Context, sub: list[str]) -> np.ndarray:
    r = ctxlib.per_knockdown(ref, other, sub)
    return np.c_[[pos[t] for t in sub], r["dot"], r["power_a"], r["power_b"]]


R = {"batch": rows(batch, kds), "lab": rows(lab, kds)}
R["line"] = np.vstack([rows(l, [t for t in kds if t in set(l.targets)]) for l in lines])


# Per knockdown sums (lines pooled), so a resample is a weighted sum.
S = {n: np.stack([np.bincount(m[:, 0].astype(int), weights=m[:, j], minlength=len(kds)) for j in (1, 2, 3)]) for n, m in R.items()}


def score(name: str, members: np.ndarray) -> float:
    dot, pa, pb = S[name][:, members].sum(1)
    return float(dot / np.sqrt(pa * pb)) if pa > 0 and pb > 0 else float("nan")


rng = np.random.default_rng(0)
report = {"genes": len(genes), "knockdowns": len(kds), "bins": []}
for lo, hi in [*BINS, (-1.0, 1.01)]:
    members = np.flatnonzero((rel > lo) & (rel <= hi))
    if len(members) < 10:
        continue
    s = {n: score(n, members) for n in R}
    # line - lab added by 040: 039 observed that the same lab's other lines
    # beat another lab's same line, but reported no interval on that
    # contrast, so it could not be claimed. The bootstrap already draws it.
    boot = {"batch - line": [], "lab - batch": [], "line - lab": []}
    for _ in range(BOOT):
        b = rng.choice(members, len(members))
        v = {n: score(n, b) for n in R}
        boot["batch - line"].append(v["batch"] - v["line"])
        boot["lab - batch"].append(v["lab"] - v["batch"])
        boot["line - lab"].append(v["line"] - v["lab"])
    row = {"bin": "all" if (lo, hi) == (-1.0, 1.01) else f"{max(lo, 0):.1f}-{min(hi, 1):.1f}", "knockdowns": len(members),
           **{n: round(v, 3) for n, v in s.items()},
           **{f"{k} 95%": [round(x, 3) for x in np.percentile(v, [2.5, 97.5])] for k, v in boot.items()}}
    report["bins"].append(row)
    print(row, flush=True)
json.dump(report, open(out, "w"), indent=1)
