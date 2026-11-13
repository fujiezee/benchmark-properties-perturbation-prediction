"""Treat each knockdown as a test item and each model as a respondent.

Benchmark analysis in NLP routinely fits an item-response model to a leaderboard:
every test item gets a difficulty and a discrimination, items with no
discrimination contribute nothing to the ranking, and items with *negative*
discrimination rank respondents backwards and usually mark a broken item. On
HellaSwag, 42 of 5,608 items reproduce the full-benchmark estimate. No published
study does this for a perturbation benchmark, where the items are knockdowns.

Classic 2PL needs binary responses; our scores are continuous, so this fits the
continuous analogue, a one-factor model per pair:

  ability_m    = the model's mean score over all knockdowns in the pair
  discrimination_k = slope of score_{m,k} on ability_m across models
  difficulty_k     = minus the mean score over models (high = hard)

Discrimination near zero means every model does the same on that knockdown, so
it cannot separate them whatever its difficulty. Negative means better models do
worse on it, which in a benchmark of language is a labelling error and here can
only be a measurement artefact.

Scores are per knockdown, which the aggregate score decomposes into exactly: the
same ratio summed over genes for one knockdown instead of over the whole test
set.

Usage: items.py OUT.json PAIR_INDEX [PAIR_INDEX ...]
       with pair indices into models.py's PAIRS.
"""

import json
import os
import sys

import numpy as np

os.environ.setdefault("PYTHONHASHSEED", "0")
import pathlib as _p, sys as _s
_h = _p.Path(__file__).resolve().parent
_s.path[:0] = [str(_h), str(_h / "lib")]
_real = sys.argv[:]
sys.argv = [_real[0], "/dev/null", "1"]
import ctxlib
import models                                    # noqa: E402  guarded by __main__
sys.argv = _real

out_p, idx = sys.argv[1], [int(a) for a in sys.argv[2:]]
PAIRS = [models.PAIRS[i] for i in idx] if idx else models.PAIRS
ABL = [m for m in models.MODELS if m != "mean"]


def per_knockdown(pred, truth, centre):
    """The aggregate score's own ratio, evaluated one knockdown at a time."""
    d, h1, h2 = truth
    pred, d, h1, h2 = (x - centre for x in (pred, d, h1, h2))
    pp = (pred * pred).sum(1)
    tt = (h1 * h2).sum(1)
    ok = (pp > 1e-12) & (tt > 0)
    out = np.full(len(pp), np.nan)
    out[ok] = (pred * d).sum(1)[ok] / np.sqrt(pp[ok] * tt[ok])
    return out


results = []
for src, tgt, kind in PAIRS:
    ctx = {k: (ctxlib.from_dataset(models.SRC[k].split(":", 1)[1])
               if models.SRC[k].startswith("dataset:") else ctxlib.from_npz(k, models.SRC[k]))
           for k in (src, tgt)}
    genes = ctxlib.shared_genes(list(ctx.values()))
    s, t = (ctxlib.restrict(ctx[k], genes) for k in (src, tgt))
    kds = sorted(set(s.targets) & set(t.targets))
    S = s.delta[[s.targets.index(k) for k in kds]]
    ri = [t.targets.index(k) for k in kds]
    T = [M[ri] for M in (t.delta, t.half1, t.half2)]
    Us = models.basis(s.delta, models.K)
    nbr = models.load_string()
    n = min(models.ANCHORS, len(kds) // 2)
    rng = np.random.default_rng(0)               # fixed, unlike models.py:151
    print(f"  {src} -> {tgt}: {len(kds)} knockdowns, {len(genes)} genes", flush=True)

    acc = {m: {} for m in ABL}                   # model -> knockdown index -> [scores]
    for rep in range(models.REPEATS):
        perm = rng.permutation(len(kds))
        anc, test = perm[:n], perm[n:]
        T_anc = T[0][anc]
        truth = [M[test] for M in T]
        centre = T_anc.mean(0)
        for m in ABL:
            pred = models.predict(m, S, T_anc, anc, test, Us, kds, nbr)
            v = per_knockdown(pred, truth, centre)
            for j, ti in enumerate(test):
                acc[m].setdefault(int(ti), []).append(float(v[j]))

    # A knockdown counts only where every model was scored on it the same number
    # of times, so discrimination is not a function of who happened to see it.
    common = set.intersection(*(set(acc[m]) for m in ABL))
    common = sorted(i for i in common
                    if len({len(acc[m][i]) for m in ABL}) == 1
                    and np.isfinite([np.mean(acc[m][i]) for m in ABL]).all())
    M = np.array([[np.mean(acc[m][i]) for i in common] for m in ABL])   # models x items
    ability = M.mean(1)
    a_c = ability - ability.mean()
    denom = (a_c * a_c).sum()
    disc = ((M - M.mean(0)) * a_c[:, None]).sum(0) / denom if denom > 0 else np.zeros(M.shape[1])
    diff = -M.mean(0)

    neg = int((disc < 0).sum())
    print(f"    {len(common)} scored by all {len(ABL)} models", flush=True)
    print(f"    discrimination: median {np.median(disc):+.3f}, "
          f"IQR [{np.percentile(disc,25):+.3f}, {np.percentile(disc,75):+.3f}], "
          f"negative for {neg} ({neg/len(common):.1%})", flush=True)
    order = np.argsort(-np.abs(disc))
    for frac in (0.05, 0.10, 0.25, 0.50):
        k = max(1, int(frac * len(common)))
        top = order[:k]
        rank_full = np.argsort(-ability)
        rank_sub = np.argsort(-M[:, top].mean(1))
        agree = float(np.mean(rank_full == rank_sub))
        print(f"      top {frac:.0%} by |discrimination| ({k:>4}): "
              f"model order reproduced {agree:.0%}", flush=True)

    results.append({"source": src, "target": tgt, "kind": kind,
                    "items": len(common), "models": ABL,
                    "ability": ability.tolist(),
                    "discrimination": disc.tolist(), "difficulty": diff.tolist(),
                    "negative_frac": neg / len(common)})

json.dump(results, open(out_p, "w"))
print(f"\n  wrote {out_p}")
print("ITEMS_DONE")
