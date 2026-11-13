"""Which knockdowns transfer, once you control for how well they were measured?

Claim 1 shows transfer falls with context distance. The obvious next question is
biological: is what survives the move to another lab or cell type a particular
kind of gene? The confound is that strongly-responding genes are both better
measured and more likely to be core-essential, so any raw association between
essentiality and transfer is partly a measurement artefact. Reliability is
measurable per knockdown from the reference context's split halves, so it can be
conditioned on rather than argued about.

Per knockdown this computes:
  rel       2c/(1+c) over the reference's halves, as in baseline.py
  transfer  <ref, other> / sqrt(|ref|^2 * <other_h1, other_h2>) per context,
            the per-knockdown form of the pooled score baseline.py reports

and joins DepMap 24Q4:
  essential   membership in CRISPRInferredCommonEssentials (genes whose knockout
              kills nearly every line)
  selectivity sd of CRISPRGeneEffect across cell lines -- low means the knockout
              does the same thing everywhere, high means it depends on context
  effect      mean CRISPRGeneEffect across lines

The test is run within reliability strata, so "essential genes transfer better"
cannot be satisfied by "essential genes are measured better".

Usage: transfer_biology.py <out.json> <ref> <batch> <lab> <far> <line...>
"""

import json
import pathlib
import sys

import numpy as np

import pathlib as _p, sys as _s
_h = _p.Path(__file__).resolve().parent
_s.path[:0] = [str(_h), str(_h / "lib")]

import ctxlib

DEPMAP = pathlib.Path("/data/vcell/raw/depmap24q4")
BINS = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]
DRAWS = 2000

out_p, ref_p, batch_p, lab_p, far_p, *line_p = sys.argv[1:]

ref = ctxlib.from_npz("ref", ref_p)
batch = ctxlib.from_npz("batch", batch_p)
lab = ctxlib.from_npz("lab", lab_p)
far = ctxlib.from_npz("far", far_p)
lines = [ctxlib.from_npz(p, p) for p in line_p]
genes = ctxlib.shared_genes([ref, batch, lab, far, *lines])
ref, batch, lab, far = (ctxlib.restrict(c, genes) for c in (ref, batch, lab, far))
lines = [ctxlib.restrict(c, genes) for c in lines]
# Each context is scored on its own intersection with the reference, not on one
# set shared by all four. Forcing a common set would drag every context down to
# the smallest, and -- because the shared set is pinned by an essentiality-focused
# screen -- would also narrow the very variable this analysis tests.
kds = sorted(set(ref.targets) & (set(batch.targets) | set(lab.targets)
             | set(far.targets) | set.union(*(set(c.targets) for c in lines))))
print(f"  {len(genes)} genes, {len(kds)} knockdowns", flush=True)

ir = {t: i for i, t in enumerate(ref.targets)}
h1, h2 = ref.half1[[ir[t] for t in kds]], ref.half2[[ir[t] for t in kds]]
c = np.clip((h1 * h2).sum(1) / (np.linalg.norm(h1, axis=1) * np.linalg.norm(h2, axis=1) + 1e-12), 0, None)
rel = 2 * c / (1 + c)

CTX = {"batch": batch, "lab": lab, "far": far}
transfer: dict[str, np.ndarray] = {}
for name, ctx in CTX.items():
    sub = [t for t in kds if t in set(ctx.targets)]
    r = ctxlib.per_knockdown(ref, ctx, sub)
    v = np.full(len(kds), np.nan)
    pos = {t: i for i, t in enumerate(kds)}
    denom = np.sqrt(np.maximum(r["power_a"] * r["power_b"], 0))
    v[[pos[t] for t in sub]] = np.where(denom > 1e-12, r["dot"] / np.maximum(denom, 1e-12), np.nan)
    transfer[name] = v
    print(f"  {name}: {np.isfinite(v).sum()} knockdowns scored", flush=True)

# lines pooled: a knockdown's transfer is the mean over the lines that carry it
acc, cnt = np.zeros(len(kds)), np.zeros(len(kds))
pos = {t: i for i, t in enumerate(kds)}
for ln in lines:
    sub = [t for t in kds if t in set(ln.targets)]
    r = ctxlib.per_knockdown(ref, ln, sub)
    denom = np.sqrt(np.maximum(r["power_a"] * r["power_b"], 0))
    ok = denom > 1e-12
    idx = np.array([pos[t] for t in sub])[ok]
    acc[idx] += r["dot"][ok] / denom[ok]
    cnt[idx] += 1
transfer["line"] = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
print(f"  line: {np.isfinite(transfer['line']).sum()} knockdowns scored", flush=True)

# --- DepMap ---
def symbol(col: str) -> str:
    return col.split(" (")[0].strip()

ess = {symbol(l) for l in (DEPMAP / "CRISPRInferredCommonEssentials.csv").read_text().splitlines()[1:] if l.strip()}
print(f"  DepMap common essentials: {len(ess)}", flush=True)

eff_mean: dict[str, float] = {}
eff_sd: dict[str, float] = {}
with (DEPMAP / "CRISPRGeneEffect.csv").open() as fh:
    cols = [symbol(x) for x in fh.readline().rstrip("\n").split(",")[1:]]
    sums = np.zeros(len(cols)); sqs = np.zeros(len(cols)); ns = np.zeros(len(cols))
    for line in fh:
        parts = line.rstrip("\n").split(",")[1:]
        if len(parts) != len(cols):
            continue
        v = np.array([float(x) if x not in ("", "NA") else np.nan for x in parts])
        ok = np.isfinite(v)
        sums[ok] += v[ok]; sqs[ok] += v[ok] ** 2; ns[ok] += 1
m = ns > 10
mu = np.where(m, sums / np.maximum(ns, 1), np.nan)
sd = np.sqrt(np.maximum(np.where(m, sqs / np.maximum(ns, 1), np.nan) - mu ** 2, 0))
for i, g in enumerate(cols):
    if m[i]:
        eff_mean[g], eff_sd[g] = float(mu[i]), float(sd[i])
print(f"  DepMap gene effect: {len(eff_mean)} genes over up to {int(ns.max())} lines", flush=True)

is_ess = np.array([t in ess for t in kds])
sel = np.array([eff_sd.get(t, np.nan) for t in kds])
eff = np.array([eff_mean.get(t, np.nan) for t in kds])
print(f"  of {len(kds)} knockdowns: {is_ess.sum()} common-essential, "
      f"{np.isfinite(sel).sum()} with a gene-effect profile", flush=True)

rng = np.random.default_rng(0)
res = {"genes": len(genes), "knockdowns": len(kds), "draws": DRAWS,
       "common_essential": int(is_ess.sum()), "bins": []}
print()
print(f"  {'bin':10}{'n':>5}{'ess':>5}  " + "".join(f"{c:>26}" for c in ("batch", "line", "lab", "far")))
for lo, hi in BINS:
    m_bin = (rel >= lo) & (rel < hi)
    row = {"bin": f"{lo}-{min(hi, 1.0)}", "knockdowns": int(m_bin.sum()),
           "essential": int((m_bin & is_ess).sum())}
    cells = []
    for ctx in ("batch", "line", "lab", "far"):
        v = transfer[ctx]
        a = m_bin & is_ess & np.isfinite(v)
        b = m_bin & ~is_ess & np.isfinite(v)
        if a.sum() < 10 or b.sum() < 10:
            cells.append(f"{'-':>26}"); continue
        d = float(v[a].mean() - v[b].mean())
        ia, ib = np.flatnonzero(a), np.flatnonzero(b)
        bs = [v[rng.choice(ia, len(ia))].mean() - v[rng.choice(ib, len(ib))].mean() for _ in range(DRAWS)]
        loq, hiq = np.percentile(bs, [2.5, 97.5])
        row[ctx] = {"essential": float(v[a].mean()), "other": float(v[b].mean()),
                    "diff": d, "ci": [float(loq), float(hiq)]}
        star = "*" if (loq > 0) or (hiq < 0) else " "
        cells.append(f"{d:+.3f} [{loq:+.3f},{hiq:+.3f}]{star}".rjust(26))
    print(f"  {row['bin']:10}{row['knockdowns']:>5}{row['essential']:>5}  " + "".join(cells))
    res["bins"].append(row)

# Is it essentiality, or just that essential genes respond more strongly?
# Reliability already conditions on signal against noise, but magnitude is the
# objection a reader will raise, so it is controlled explicitly: a partial
# Spearman of essentiality against transfer with reliability AND the reference
# response magnitude both ranked and regressed out.
mag = np.linalg.norm(ref.delta[[ir[t] for t in kds]], axis=1)


def partial_spearman(x, y, covars):
    """Spearman of x and y with covars removed, all on ranks."""
    from scipy.stats import rankdata
    X = np.column_stack([rankdata(c) for c in covars] + [np.ones(len(x))])
    rx, ry = rankdata(x), rankdata(y)
    bx = np.linalg.lstsq(X, rx, rcond=None)[0]
    by = np.linalg.lstsq(X, ry, rcond=None)[0]
    ex, ey = rx - X @ bx, ry - X @ by
    n = len(x)
    if n < 20 or ex.std() < 1e-9 or ey.std() < 1e-9:
        return float("nan"), float("nan")
    r = float(np.corrcoef(ex, ey)[0, 1])
    t = r * np.sqrt(max(n - 2 - len(covars), 1) / max(1 - r * r, 1e-12))
    from scipy.stats import t as tdist
    return r, float(2 * tdist.sf(abs(t), max(n - 2 - len(covars), 1)))


print()
print("  essentiality vs transfer, partial on reliability AND response magnitude:")
res["partial"] = {}
for ctx in ("batch", "line", "lab", "far"):
    v = transfer[ctx]
    ok = np.isfinite(v) & np.isfinite(mag)
    r, pv = partial_spearman(is_ess[ok].astype(float), v[ok], [rel[ok], np.log1p(mag[ok])])
    res["partial"][ctx] = {"rho": r, "p": pv, "n": int(ok.sum())}
    print(f"    {ctx:8} rho {r:+.3f}  p {pv:.2e}  n {ok.sum()}")

print()
print("  magnitude vs transfer, partial on reliability (for comparison):")
res["partial_mag"] = {}
for ctx in ("batch", "line", "lab", "far"):
    v = transfer[ctx]
    ok = np.isfinite(v) & np.isfinite(mag)
    r, pv = partial_spearman(np.log1p(mag[ok]), v[ok], [rel[ok]])
    res["partial_mag"][ctx] = {"rho": r, "p": pv, "n": int(ok.sum())}
    print(f"    {ctx:8} rho {r:+.3f}  p {pv:.2e}  n {ok.sum()}")

# selectivity, conditioned on reliability the same way
print()
print("  transfer vs gene-effect selectivity (Spearman), within reliability bins:")
from scipy.stats import spearmanr
res["selectivity"] = []
for lo, hi in BINS:
    m_bin = (rel >= lo) & (rel < hi) & np.isfinite(sel)
    cells = []
    row = {"bin": f"{lo}-{min(hi, 1.0)}", "n": int(m_bin.sum())}
    for ctx in ("batch", "line", "lab", "far"):
        v = transfer[ctx]
        ok = m_bin & np.isfinite(v)
        if ok.sum() < 30:
            cells.append(f"{'-':>14}"); continue
        rho, p = spearmanr(sel[ok], v[ok])
        row[ctx] = {"rho": float(rho), "p": float(p), "n": int(ok.sum())}
        cells.append(f"{rho:+.3f}{'*' if p < 0.05 else ' '}".rjust(14))
    print(f"  {row['bin']:10}{row['n']:>5}  " + "".join(cells))
    res["selectivity"].append(row)

pathlib.Path(out_p).write_text(json.dumps(res, indent=1))
print(f"\n  wrote {out_p}")
print("TRANSFER_BIOLOGY_DONE")
