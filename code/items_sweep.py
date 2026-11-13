"""A respondent pool large enough to estimate item parameters.

The first probe fitted discrimination from six models. Simulating six models at
the observed ability spread with no true item signal gives 50% negative
discrimination against the 30-33% observed, so the signal was real but the
margin small, and compact-benchmark selection failed. Item-response theory needs
respondents spread in ability, not respondents that are separately publishable,
and the existing models have hyperparameters nobody fixed for a scientific
reason.

So the pool is the same seven models swept over K, the ridge penalty, the
neighbour count and the STRING threshold: about thirty respondents from code
already written. `models.predict` is reused unchanged -- K enters as a slice of
one SVD, the penalty and neighbour count by rebinding the module globals the
function reads, the threshold as a different neighbour map -- so a respondent
cannot drift from the model it is a variant of.

Usage: items_sweep.py OUT.json PAIR_INDEX [PAIR_INDEX ...]
"""

import itertools
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
KMAX = 100
THRESHOLDS = (400, 700, 900)

# (label, model, K, lambda, knn, string threshold). None = not used by that model.
POOL = [("copy", "copy", None, None, None, None)]
POOL += [(f"copy_pc.K{k}", "copy_pc", k, None, None, None) for k in (5, 10, 20, 50, 100)]
POOL += [(f"linear.K{k}.L{l:g}", "linear", k, l, None, None)
         for k, l in itertools.product((5, 20, 50, 100), (0.1, 1.0, 10.0))]
POOL += [(f"calibrated.K{k}.L{l:g}", "calibrated", k, l, None, None)
         for k, l in itertools.product((10, 50), (0.1, 10.0))]
POOL += [(f"knn.n{n}", "knn", None, None, n, None) for n in (1, 3, 10, 25, 50)]
POOL += [(f"network.s{s}", "network", None, None, None, s) for s in THRESHOLDS]
print(f"  {len(POOL)} respondents over {len(PAIRS)} pairs", flush=True)

NBR = {s: models.load_string(s) for s in THRESHOLDS}


def per_knockdown(pred, truth, centre):
    """The aggregate score's own ratio, one knockdown at a time."""
    d, h1, h2 = truth
    pred, d, h1, h2 = (x - centre for x in (pred, d, h1, h2))
    pp, tt = (pred * pred).sum(1), (h1 * h2).sum(1)
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
    # One SVD; every K is a slice of it, since basis() returns the leading
    # columns in order.
    Umax = models.basis(s.delta, KMAX)
    n = min(models.ANCHORS, len(kds) // 2)
    rng = np.random.default_rng(0)
    print(f"  {src} -> {tgt}: {len(kds)} knockdowns, {len(genes)} genes", flush=True)

    acc = {lab: {} for lab, *_ in POOL}
    for rep in range(models.REPEATS):
        perm = rng.permutation(len(kds))
        anc, test = perm[:n], perm[n:]
        T_anc = T[0][anc]
        truth = [M[test] for M in T]
        centre = T_anc.mean(0)
        for lab, m, k, lam, nn, thr in POOL:
            if lam is not None:
                models.LAMBDA = lam
            if nn is not None:
                models.KNN = nn
            Us = Umax[:, :k] if k else Umax
            pred = models.predict(m, S, T_anc, anc, test, Us, kds,
                                  NBR[thr] if thr else NBR[700])
            v = per_knockdown(pred, truth, centre)
            for j, ti in enumerate(test):
                acc[lab].setdefault(int(ti), []).append(float(v[j]))
    models.LAMBDA, models.KNN = 1.0, 10          # leave the module as found

    labs = [lab for lab, *_ in POOL]
    common = sorted(set.intersection(*(set(acc[l]) for l in labs)))
    common = [i for i in common if np.isfinite([np.mean(acc[l][i]) for l in labs]).all()]
    M = np.array([[np.mean(acc[l][i]) for i in common] for l in labs])
    ability = M.mean(1)
    a_c = ability - ability.mean()
    disc = ((M - M.mean(0)) * a_c[:, None]).sum(0) / (a_c * a_c).sum()
    neg = int((disc < 0).sum())

    # What a pool of this size gives with no item signal at all, as the floor to
    # read the negative fraction against.
    null = np.random.default_rng(1).normal(0, M.std(), (len(labs), 400))
    nd = ((null - null.mean(0)) * a_c[:, None]).sum(0) / (a_c * a_c).sum()
    print(f"    {len(common)} items scored by all {len(labs)}", flush=True)
    print(f"    ability {ability.min():.3f} to {ability.max():.3f}", flush=True)
    print(f"    discrimination median {np.median(disc):+.3f}, "
          f"negative {neg} ({neg / len(common):.1%}) vs null {(nd < 0).mean():.1%}", flush=True)

    # Selecting by |discrimination| is only worth anything if it beats drawing the
    # same number of items at random, which is the comparison the NLP studies
    # make and the one that decides whether this machinery earns its place.
    rank = lambda v: np.argsort(np.argsort(-v))
    full_r = rank(ability)
    rng_sel = np.random.default_rng(2)
    sel = []
    for frac in (0.02, 0.05, 0.10, 0.25, 0.50):
        kk = max(2, int(frac * len(common)))
        top = np.argsort(-np.abs(disc))[:kk]
        rho = np.corrcoef(full_r, rank(M[:, top].mean(1)))[0, 1]
        rnd = [np.corrcoef(full_r, rank(M[:, rng_sel.choice(len(common), kk, replace=False)]
                                        .mean(1)))[0, 1] for _ in range(200)]
        beat = float(np.mean(np.array(rnd) < rho))
        sel.append({"frac": frac, "k": kk, "rho": float(rho),
                    "random_median": float(np.median(rnd)),
                    "random_p90": float(np.percentile(rnd, 90)), "beats_random": beat})
        print(f"      top {frac:.0%} ({kk:>4}): selected {rho:+.3f}  "
              f"random {np.median(rnd):+.3f} (p90 {np.percentile(rnd, 90):+.3f})  "
              f"better than {beat:.0%} of random draws", flush=True)

    results.append({"source": src, "target": tgt, "kind": kind, "items": len(common),
                    "respondents": labs, "ability": ability.tolist(),
                    "discrimination": disc.tolist(),
                    "difficulty": (-M.mean(0)).tolist(),
                    "negative_frac": neg / len(common),
                    "null_negative_frac": float((nd < 0).mean()),
                    "selection": sel})

json.dump(results, open(out_p, "w"))
print(f"\n  wrote {out_p}")
print("SWEEP_DONE")
