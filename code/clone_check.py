"""Bound how much random-half splitting inflates our ceiling.

arXiv:2610.32962 reports that cells carrying one guide come from few transduced
clones, so two random halves of a perturbation's cells share clones and agree
more than independent transductions do. Our `half1`/`half2` are random cell
splits, so an inflated `<half1, half2>` would deflate every transfer score we
report and inflate the headroom -- the direction that would manufacture the
finding we are considering building on.

The replicate pair bounds it with data in hand: Replogle's K562 essential and
K562 genome-wide are two separate screens of one cell line from one lab. For a
knockdown in both,

  within-screen  = corr(half1, half2), the quantity our denominator rests on,
                   raised to a whole-screen reliability by Spearman-Brown;
  between-screen = corr(screen A delta, screen B delta).

Under sampling noise alone the two should agree. Within-screen running above
between-screen is excess agreement, which the clone effect and any batch effect
share; this therefore bounds the clone inflation from above rather than
isolating it. Reported per knockdown, so the spread is visible and not just a
pooled number.

Usage: clone_check.py OUT.json A.npz B.npz [LABEL_A LABEL_B]
"""

import json
import pathlib
import sys

import numpy as np

import pathlib as _p, sys as _s
_h = _p.Path(__file__).resolve().parent
_s.path[:0] = [str(_h), str(_h / "lib")]
import ctxlib

out_p, a_p, b_p = sys.argv[1:4]
LA = sys.argv[4] if len(sys.argv) > 4 else "A"
LB = sys.argv[5] if len(sys.argv) > 5 else "B"

A = ctxlib.from_npz(LA, a_p)
B = ctxlib.from_npz(LB, b_p)
genes = ctxlib.shared_genes([A, B])
A, B = ctxlib.restrict(A, genes), ctxlib.restrict(B, genes)
kds = sorted(set(A.targets) & set(B.targets))
print(f"  {len(kds)} knockdowns in both, {len(genes)} shared genes", flush=True)

ai = {t: i for i, t in enumerate(A.targets)}
bi = {t: i for i, t in enumerate(B.targets)}
ra = [ai[k] for k in kds]
rb = [bi[k] for k in kds]


def rowcorr(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Pearson r per row, each row centred on its own mean."""
    Xc = X - X.mean(1, keepdims=True)
    Yc = Y - Y.mean(1, keepdims=True)
    num = (Xc * Yc).sum(1)
    den = np.sqrt((Xc * Xc).sum(1) * (Yc * Yc).sum(1))
    return np.where(den > 1e-12, num / np.maximum(den, 1e-12), np.nan)


wa = rowcorr(A.half1[ra], A.half2[ra])
wb = rowcorr(B.half1[rb], B.half2[rb])
bt = rowcorr(A.delta[ra], B.delta[rb])

# Spearman-Brown: a split-half r measures half the cells; the whole screen's
# reliability is 2r/(1+r). This is the correction cell-eval applies, and without
# it the within-screen number is not comparable with the between-screen one.
sb = lambda r: np.where(np.isfinite(r), 2 * r / (1 + r), np.nan)
wa_f, wb_f = sb(wa), sb(wb)

ok = np.isfinite(wa_f) & np.isfinite(wb_f) & np.isfinite(bt)
print(f"  {ok.sum()} knockdowns with all three finite", flush=True)
within = np.nanmean([wa_f[ok], wb_f[ok]], axis=0)
excess = within - bt[ok]

q = lambda v: np.percentile(v, [25, 50, 75])
print()
print(f"  {'quantity':34}{'median':>9}{'IQR':>22}")
for name, v in (("within-screen split-half (raw)", np.nanmean([wa[ok], wb[ok]], axis=0)),
                ("within-screen, Spearman-Brown", within),
                ("between-screen (two screens)", bt[ok]),
                ("excess = within - between", excess)):
    lo, md, hi = q(v)
    print(f"  {name:34}{md:>9.3f}{f'[{lo:.3f}, {hi:.3f}]':>22}")

frac = float(np.mean(excess > 0))
ratio = float(np.median(within) / max(np.median(bt[ok]), 1e-9))
print()
print(f"  within exceeds between for {frac:.1%} of knockdowns")
print(f"  median ratio within / between = {ratio:.2f}")
print(f"  implied inflation of our denominator <h1,h2>: up to {ratio:.2f}x,")
print(f"  which would deflate a transfer score by up to sqrt({ratio:.2f}) = {np.sqrt(ratio):.2f}x")

# Stratify by reliability: the clone effect should not vanish with more cells,
# a batch effect need not behave that way.
print()
print(f"  {'within-screen bin':22}{'n':>6}{'within':>9}{'between':>9}{'excess':>9}")
for lo, hi in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)):
    m = (within >= lo) & (within < hi)
    if m.sum() < 10:
        continue
    print(f"  {f'{lo:.1f}-{hi:.1f}':22}{m.sum():>6}{within[m].mean():>9.3f}"
          f"{bt[ok][m].mean():>9.3f}{excess[m].mean():>9.3f}")

json.dump({"a": LA, "b": LB, "knockdowns": int(ok.sum()), "genes": len(genes),
           "within_sb_median": float(np.median(within)),
           "between_median": float(np.median(bt[ok])),
           "excess_median": float(np.median(excess)),
           "frac_within_exceeds": frac, "ratio": ratio},
          open(out_p, "w"), indent=1)
print(f"\n  wrote {out_p}")
print("CLONE_CHECK_DONE")
