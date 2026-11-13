"""Cell-level parts for any source screen, in one fixed gene space.

`gears_data.py` builds `k562_parts.npz` for a single screen and picks its 2,000
genes from that screen's own variance. The zero-shot transfer comparison needs
the same treatment for several source screens, and needs every model to predict
into the *same* gene space -- otherwise a deep model's row is not comparable with
the 21-pair table or with the other sources. So the gene list is read from a file
rather than recomputed, and every source is projected onto it.

Splits differ from the leaderboard's too. There the holdout is a set of
knockdowns; here the holdout is the target *context*, so the source model should
see every knockdown the source measured. This writes `train` = all knockdowns
except a 10% `val` for early stopping, and an empty `test`. Scoring happens in
`zeroshot.py` against a different screen entirely.

A knockdown is kept only if it appears both in the raw cells and in the source's
pseudobulk `.npz`, since `copy` and `linear` need its measured source delta to
compete against the deep model on equal information.

Usage: source_parts.py OUT_DIR RAW.h5ad GENES.txt SOURCE.npz [PER_KD] [N_CTRL]
"""

import json
import pathlib
import sys

import h5py
import numpy as np
import scipy.sparse as sp

import pathlib as _p, sys as _s
_h = _p.Path(__file__).resolve().parent
_s.path[:0] = [str(_h), str(_h / "lib")]
import ctxlib

out = pathlib.Path(sys.argv[1])
raw, genes_txt, src_npz = sys.argv[2:5]
# 60 cells per knockdown and 6,000 controls are gears_data.py's values, which
# produced the 122,707-cell matrix GEARS and CPA were scored on. Keeping them
# means a source screen's model sees the same depth per knockdown as that run.
PER_KD = int(sys.argv[5]) if len(sys.argv) > 5 else 60
N_CTRL = int(sys.argv[6]) if len(sys.argv) > 6 else 6000
VAL_FRAC, SEED, CHUNK = 0.1, 0, 20000
rng = np.random.default_rng(SEED)
out.mkdir(parents=True, exist_ok=True)

genes = np.array([g for g in pathlib.Path(genes_txt).read_text().split() if g], dtype=str)
src = ctxlib.from_npz("source", src_npz)
print(f"  gene space: {len(genes)} genes; source pseudobulk: {len(src.targets)} knockdowns", flush=True)

f = h5py.File(raw, "r")


def column(group: str, name: str) -> np.ndarray:
    """Read an obs/var column, decoding whichever categorical layout anndata used."""
    node = f[group][name]
    if isinstance(node, h5py.Group):
        values = node["categories"][:][node["codes"][:]]
    else:
        values = node[:]
        cats = f[group].get("__categories")
        if cats is not None and name in cats:
            values = cats[name][:][values]
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in values])


target = column("obs", "gene")
umi = f["obs"]["UMI_count"][:].astype(np.float64)
var_names = column("var", "gene_name")

missing = [g for g in genes if not (var_names == g).any()]
if missing:
    print(f"  WARNING {len(missing)} of {len(genes)} genes absent from this screen, "
          f"e.g. {missing[:5]}", flush=True)
keep_g = np.array([bool((var_names == g).any()) for g in genes])
genes, cols = genes[keep_g], np.array([np.flatnonzero(var_names == g)[0] for g in genes[keep_g]])

kds = sorted(set(src.targets) & set(np.unique(target)))
if not kds:
    sys.exit("no knockdown appears in both the raw cells and the source pseudobulk")
val = sorted(rng.choice(kds, max(1, int(VAL_FRAC * len(kds))), replace=False).tolist())
train = sorted(set(kds) - set(val))
print(f"  {len(kds)} knockdowns shared with the pseudobulk: {len(train)} train, {len(val)} val",
      flush=True)

pick = [rng.choice(np.flatnonzero(target == k), min(PER_KD, int((target == k).sum())), replace=False)
        for k in kds]
ctrl = np.flatnonzero(target == "non-targeting")
if len(ctrl) == 0:
    sys.exit("no 'non-targeting' cells; control label differs in this screen")
pick.append(rng.choice(ctrl, min(N_CTRL, len(ctrl)), replace=False))
rows = np.sort(np.concatenate(pick))

X = np.empty((len(rows), len(genes)), np.float32)
for s in range(0, len(rows), CHUNK):
    r = rows[s:s + CHUNK]
    X[s:s + len(r)] = np.log1p(f["X"][r][:, cols] * (1e4 / umi[r])[:, None])
cond = np.where(target[rows] == "non-targeting", "ctrl",
                np.char.add(target[rows].astype(str), "+ctrl"))
X = sp.csr_matrix(X)
np.savez_compressed(out / "parts.npz", data=X.data, indices=X.indices, indptr=X.indptr,
                    shape=np.array(X.shape), condition=cond.astype(str), genes=genes.astype(str))

as_cond = lambda ks: [f"{k}+ctrl" for k in ks]
json.dump({"train": ["ctrl", *as_cond(train)], "val": as_cond(val), "test": []},
          open(out / "split.json", "w"))
(out / "genes.txt").write_text("\n".join(genes))
json.dump({"raw": raw, "source_npz": src_npz, "genes": len(genes),
           "knockdowns": len(kds), "cells": int(X.shape[0]),
           "per_kd": PER_KD, "n_ctrl": N_CTRL, "seed": SEED},
          open(out / "manifest.json", "w"), indent=1)
print(f"  wrote {out}/parts.npz: {X.shape[0]} cells x {X.shape[1]} genes", flush=True)
print("SOURCE_PARTS_DONE")
