"""Put a deep model in the cross-lab table, zero-shot.

Claim 2 says a linear map beats every other way of using a hundred measurements
from a new context. GEARS and CPA cannot enter that protocol -- neither has an
input for measured responses in the target context -- so they are scored on a
held-out split inside K562 while the claim is made across labs. The deep models
are evaluated in the easy regime and the claim is made in the hard one.

This closes that without changing the protocol, by asking a harsher question of
the deep model and an easier one of the baseline:

  the deep model  trains on an ENTIRE source screen and predicts the target
                  context with no target data at all
  `linear`        gets 100 measured responses from the target context

Different information, deliberately. If the linear map still wins, the sentence
is that a published deep model given a whole screen transfers worse than a
linear map fitted to a hundred measurements. If it loses, Claim 2 needs
qualifying, which is better found here than by a reviewer.

The deep model's prediction is its learned source response for a knockdown,
scored against that knockdown's measured response in the target. That is the
same quantity `copy` produces from the raw source delta, so `copy` is the
baseline that isolates what the model's representation adds.

Scoring reuses `models.py` rather than reimplementing it, so the numbers sit in
the same table: same `score`, same `specific` centring on the anchors' mean,
same anchor count and repeat count, same PC basis.

Usage: zeroshot.py OUT.json PRED.npz SOURCE.npz TARGET.npz MODEL_LABEL [SCOPE]
"""

import json
import pathlib
import sys

import numpy as np

import pathlib as _p, sys as _s
_h = _p.Path(__file__).resolve().parent
_s.path[:0] = [str(_h), str(_h / "lib")]

# models.py parses argv at import (out path, worker count), so it is imported
# under a dummy argv and the real one restored afterwards. Importing it rather
# than copying score/basis/predict is deliberate: these numbers have to sit in
# the same table as the 21 pairs, and a copy would be free to drift from it.
_real = sys.argv[:]
sys.argv = [_real[0], "/dev/null", "1"]

import ctxlib
import models                     # noqa: E402  -- reused for score/basis/predict

sys.argv = _real
out_p, pred_p, src_p, tgt_p, label = sys.argv[1:6]
# Which knockdowns of the prediction file to score. "train" is the primary
# experiment: knockdowns the deep model trained on, where it and `copy` hold the
# same source information. "all" or "val" are diagnostics.
SCOPE = sys.argv[6] if len(sys.argv) > 6 else "train"
REPEATS, ANCHORS = models.REPEATS, models.ANCHORS
BASELINES = ["mean", "copy", "copy_pc", "linear"]

z = np.load(pred_p, allow_pickle=False)
pred_genes = [str(g) for g in z["genes"]]
pred_kds = [str(t) for t in z["targets"]]
P_raw = np.asarray(z["delta"], dtype=np.float64)
print(f"  prediction: {len(pred_kds)} knockdowns x {len(pred_genes)} genes", flush=True)

if SCOPE != "all":
    if "split" not in z.files:
        sys.exit(f"prediction file has no `split` array, so scope {SCOPE!r} cannot be applied; "
                 f"regenerate it with gears_train_transfer.py or cpa_predict_all.py, or pass 'all' "
                 f"and state in the write-up that the knockdowns were held out from the deep model")
    sp_arr = z["split"].astype(str)
    keep = {t for t, g in zip(pred_kds, sp_arr) if g == SCOPE}
    print(f"  scope {SCOPE!r}: {len(keep)} of {len(pred_kds)} knockdowns", flush=True)
    pred_kds = [t for t in pred_kds if t in keep]
    P_raw = P_raw[[i for i, t in enumerate(z["targets"].astype(str)) if t in keep]]

src = ctxlib.from_npz("source", src_p)
tgt = ctxlib.from_npz("target", tgt_p)
genes = [g for g in ctxlib.shared_genes([src, tgt]) if g in set(pred_genes)]
src, tgt = (ctxlib.restrict(c, genes) for c in (src, tgt))
gi = {g: i for i, g in enumerate(pred_genes)}
P_cols = [gi[g] for g in genes]
print(f"  shared genes: {len(genes)}", flush=True)

kds = sorted(set(src.targets) & set(tgt.targets) & set(pred_kds))
if len(kds) < 4 * ANCHORS:
    print(f"  WARNING only {len(kds)} shared knockdowns", flush=True)
print(f"  shared knockdowns: {len(kds)}", flush=True)

si = {t: i for i, t in enumerate(src.targets)}
ti = {t: i for i, t in enumerate(tgt.targets)}
pi = {t: i for i, t in enumerate(pred_kds)}
S = src.delta[[si[k] for k in kds]]
T = [M[[ti[k] for k in kds]] for M in (tgt.delta, tgt.half1, tgt.half2)]
P = P_raw[[pi[k] for k in kds]][:, P_cols]

Us = models.basis(src.delta, models.K)
nbr = models.load_string()
n = min(ANCHORS, len(kds) // 2)
rng = np.random.default_rng(0)        # fixed: see the note on PYTHONHASHSEED below

rows = []
for rep in range(REPEATS):
    perm = rng.permutation(len(kds))
    anc, test = perm[:n], perm[n:]
    T_anc = T[0][anc]
    truth = [M[test] for M in T]
    centre = T_anc.mean(0)
    for m in BASELINES:
        p = models.predict(m, S, T_anc, anc, test, Us, kds, nbr)
        rows.append({"model": m, "repeat": rep, "anchors": int(n), "test": int(len(test)),
                     "overall": models.score(p, truth, None),
                     "specific": models.score(p, truth, centre)})
    # the deep model uses no anchors; its prediction is the same for every repeat
    rows.append({"model": label, "repeat": rep, "anchors": 0, "test": int(len(test)),
                 "overall": models.score(P[test], truth, None),
                 "specific": models.score(P[test], truth, centre)})

summary = []
for m in BASELINES + [label]:
    sel = [r for r in rows if r["model"] == m]
    summary.append({"model": m, "anchors": sel[0]["anchors"], "test": sel[0]["test"],
                    **{k: float(np.mean([r[k] for r in sel])) for k in ("overall", "specific")},
                    # ddof=1, unlike models.py:174; see tool/README.md
                    **{k + " sd": float(np.std([r[k] for r in sel], ddof=1))
                       for k in ("overall", "specific")}})

print()
print(f"  {'model':14}{'anchors':>8}{'overall':>10}{'sd':>8}{'specific':>10}{'sd':>8}")
for r in sorted(summary, key=lambda x: -x["specific"]):
    print(f"  {r['model']:14}{r['anchors']:>8}{r['overall']:>10.4f}{r['overall sd']:>8.4f}"
          f"{r['specific']:>10.4f}{r['specific sd']:>8.4f}")

lin = next(r for r in summary if r["model"] == "linear")
dm = next(r for r in summary if r["model"] == label)
d = [a["specific"] - b["specific"] for a, b in
     zip([r for r in rows if r["model"] == label], [r for r in rows if r["model"] == "linear"])]
sd = float(np.std(d, ddof=1))
half = 1.96 * sd / np.sqrt(len(d))
print()
print(f"  {label} - linear on specific: {np.mean(d):+.4f} "
      f"[{np.mean(d)-half:+.4f}, {np.mean(d)+half:+.4f}] (paired over {len(d)} draws)")
cp = next(r for r in summary if r["model"] == "copy")
print(f"  {label} - copy   on specific: {dm['specific']-cp['specific']:+.4f} "
      f"(what the model's representation adds over the raw source delta)")

pathlib.Path(out_p).write_text(json.dumps(
    {"label": label, "scope": SCOPE, "genes": len(genes), "knockdowns": len(kds),
     "summary": summary, "rows": rows}, indent=1))
print(f"\n  wrote {out_p}")
print("ZEROSHOT_DONE")
