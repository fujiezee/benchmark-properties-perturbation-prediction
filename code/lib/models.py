"""Does the ranking of perturbation-response models depend on which lab the target comes from?

Task: in a target context, n anchor knockdowns are measured; predict the
others. Six models, each given the source context (all its knockdowns) and
the target's anchors:
  mean       the anchors' mean response, for every knockdown
  copy       the source's response to the same knockdown
  copy_pc    copy projected on the source's top-K components: denoised, no
             anchors; what a model gains from anchors is measured above this
  calibrated copy plus a ridge correction from the source's top-K components
             to the target-minus-source residual, fit on the anchors (calib.py)
  knn        the anchors most like the knockdown in the source (top 10 by
             cosine), their target responses weighted by similarity
  network    the anchors that are the knockdown's STRING neighbours (combined
             score >= 700), their target responses averaged; mean if none
  linear     ridge from the source's top-K components of a knockdown's
             response to its target response (Ahlmann-Eltze et al. 2024's
             linear baseline with source-derived perturbation embeddings)
Scores on held-out knockdowns, target noise corrected with its halves:
  overall    sum <pred, truth> / sqrt(sum |pred|^2 * sum <half1, half2>)
  specific   the same after subtracting the anchors' mean response from both,
             i.e. what a model says about this knockdown beyond the average
Random anchor draws (REPEATS), run in parallel.

Usage: python models.py OUT_JSON [WORKERS]
"""

import pathlib as _p, sys as _s
_h = _p.Path(__file__).resolve().parent
_s.path[:0] = [str(_h), str(_h / "lib")]

import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import ctxlib

out = sys.argv[1]
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
P = "/data/vcell/pilot039"
ANCHORS = 100
REPEATS = 10
K = 30
LAMBDA = 1.0
KNN = 10
SRC = {
    "Weissman K562 essential": f"{P}/replogle/k562_essential.npz",
    "Weissman K562 genome-wide": f"{P}/replogle/k562_gwps.npz",
    "Weissman RPE1": f"{P}/replogle/rpe1.npz",
    "Weissman HepG2": f"{P}/replogle/hepg2.npz",
    "Weissman Jurkat": f"{P}/replogle/jurkat.npz",
    "Satija K562": f"{P}/viperturb/k562_viperturb.npz",
    "Shendure K562": f"{P}/gasperini/k562_gasperini.npz",
    "Arc H1": "dataset:h1",
}
PAIRS = [
    ("Weissman K562 essential", "Weissman K562 genome-wide", "replicate"),
    ("Weissman K562 essential", "Weissman RPE1", "same lab, other type"),
    ("Weissman K562 essential", "Weissman HepG2", "same lab, other type"),
    ("Weissman K562 essential", "Weissman Jurkat", "same lab, other type"),
    ("Weissman K562 essential", "Satija K562", "other lab, same type"),
    ("Weissman K562 genome-wide", "Satija K562", "other lab, same type"),
    ("Weissman K562 genome-wide", "Shendure K562", "other lab, same type"),
    ("Weissman K562 essential", "Shendure K562", "other lab, same type"),
    ("Weissman K562 genome-wide", "Arc H1", "other lab, other type (VCC)"),
]
MODELS = ["mean", "copy", "copy_pc", "calibrated", "knn", "network", "linear"]


def load_string(min_score: int = 700) -> dict[str, set[str]]:
    name = {}
    with gzip.open("/data/vcell/raw/9606.protein.info.v12.0.txt.gz", "rt") as f:
        next(f)
        for line in f:
            pid, gene = line.split("\t", 2)[:2]
            name[pid] = gene
    nbr: dict[str, set[str]] = {}
    with gzip.open("/data/vcell/raw/9606.protein.links.v12.0.txt.gz", "rt") as f:
        next(f)
        for line in f:
            a, b, s = line.split()
            if int(s) >= min_score and a in name and b in name:
                nbr.setdefault(name[a], set()).add(name[b])
    return nbr


def basis(M: np.ndarray, k: int) -> np.ndarray:
    _, _, vt = np.linalg.svd(M - M.mean(0), full_matrices=False)
    return vt[:k].T


def ridge(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    G = X.T @ X
    return np.linalg.solve(G + LAMBDA * np.eye(G.shape[0]) * np.trace(G) / G.shape[0], X.T @ Y)


def predict(model: str, S: np.ndarray, T_anc: np.ndarray, anc: np.ndarray, test: np.ndarray, Us: np.ndarray,
            kds: list[str], nbr: dict[str, set[str]]) -> np.ndarray:
    A, X = S[anc], S[test]
    mu = T_anc.mean(0)
    if model == "mean":
        return np.tile(mu, (len(test), 1))
    if model == "copy":
        return X
    if model == "copy_pc":
        mu_s = S.mean(0)
        return mu_s + ((X - mu_s) @ Us) @ Us.T
    if model == "calibrated":
        return X + (X @ Us) @ ridge(A @ Us, T_anc - A)
    if model == "linear":
        Za, Zx = A @ Us, X @ Us
        za_mu = Za.mean(0)
        return mu + (Zx - za_mu) @ ridge(Za - za_mu, T_anc - mu)
    if model == "knn":
        an = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        sim = xn @ an.T
        top = np.argsort(-sim, axis=1)[:, :KNN]
        w = np.clip(np.take_along_axis(sim, top, 1), 0, None) + 1e-6
        return np.einsum("ij,ijk->ik", w / w.sum(1, keepdims=True), T_anc[top])
    if model == "network":
        anchor_pos = {kds[i]: j for j, i in enumerate(anc)}
        out = np.empty((len(test), T_anc.shape[1]))
        for r, i in enumerate(test):
            hits = [anchor_pos[g] for g in nbr.get(kds[i], ()) if g in anchor_pos]
            out[r] = T_anc[hits].mean(0) if hits else mu
        return out
    raise ValueError(model)


def score(pred: np.ndarray, truth: list[np.ndarray], center: np.ndarray | None) -> float:
    d, h1, h2 = truth
    if center is not None:
        pred, d, h1, h2 = pred - center, d - center, h1 - center, h2 - center
    pp, tt = (pred * pred).sum(), (h1 * h2).sum()
    return float((pred * d).sum() / np.sqrt(pp * tt)) if pp > 1e-12 and tt > 0 else 0.0


def run_pair(job: tuple[str, str, str]) -> list[dict]:
    src, tgt, kind = job
    ctx = {k: (ctxlib.from_dataset(SRC[k].split(":", 1)[1]) if SRC[k].startswith("dataset:") else ctxlib.from_npz(k, SRC[k]))
           for k in (src, tgt)}
    genes = ctxlib.shared_genes(list(ctx.values()))
    s, t = (ctxlib.restrict(ctx[k], genes) for k in (src, tgt))
    kds = sorted(set(s.targets) & set(t.targets))
    S = s.delta[[s.targets.index(k) for k in kds]]
    ri = [t.targets.index(k) for k in kds]
    T = [M[ri] for M in (t.delta, t.half1, t.half2)]
    Us = basis(s.delta, K)
    nbr = load_string()
    n = min(ANCHORS, len(kds) // 2)
    rng = np.random.default_rng(abs(hash((src, tgt))) % 2**32)
    rows = []
    for rep in range(REPEATS):
        perm = rng.permutation(len(kds))
        anc, test = perm[:n], perm[n:]
        T_anc = T[0][anc]
        truth = [M[test] for M in T]
        for m in MODELS:
            pred = predict(m, S, T_anc, anc, test, Us, kds, nbr)
            rows.append({"source": src, "target": tgt, "kind": kind, "repeat": rep, "model": m, "anchors": n, "test": len(test),
                         "overall": score(pred, truth, None), "specific": score(pred, truth, T_anc.mean(0))})
    return rows


if __name__ == "__main__":
    with ProcessPoolExecutor(min(WORKERS, len(PAIRS))) as ex:
        rows = [r for part in ex.map(run_pair, PAIRS) for r in part]
    summary = []
    for src, tgt, kind in PAIRS:
        for m in MODELS:
            sel = [r for r in rows if r["source"] == src and r["target"] == tgt and r["model"] == m]
            summary.append({"source": src, "target": tgt, "kind": kind, "model": m, "anchors": sel[0]["anchors"], "test": sel[0]["test"],
                            **{k: round(float(np.mean([r[k] for r in sel])), 3) for k in ("overall", "specific")},
                            **{k + " sd": round(float(np.std([r[k] for r in sel])), 3) for k in ("overall", "specific")}})
    json.dump({"summary": summary, "rows": rows}, open(out, "w"), indent=1)
    for src, tgt, kind in PAIRS:
        sel = [x for x in summary if x["source"] == src and x["target"] == tgt]
        rank = sorted(sel, key=lambda x: -x["specific"])
        print(f"{kind:28s} {src} -> {tgt} (anchors {sel[0]['anchors']}, test {sel[0]['test']})")
        print("   overall : " + "  ".join(f"{x['model']} {x['overall']:.2f}" for x in sel))
        print("   specific: " + "  ".join(f"{x['model']} {x['specific']:.2f}±{x['specific sd']:.2f}" for x in sel))
        print("   rank by specific: " + " > ".join(x["model"] for x in rank), flush=True)
