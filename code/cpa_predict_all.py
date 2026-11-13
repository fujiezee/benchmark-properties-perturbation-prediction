"""Predict every knockdown from a saved CPA model, counterfactually.

`cpa_train_cf.py` already runs `model.predict` over all cells and then keeps only
the test knockdowns, so the predictions the transfer comparison needs are
computed and discarded. This reloads a saved model and keeps all of them, with a
`split` array so the scorer can select knockdowns the model trained on -- which
is what makes it comparable with `copy`, as `zeroshot.py` explains.

The counterfactual setup is CPA's documented one and the reason the earlier 0.90
was a reconstruction: `X_true` holds the observed expression and `adata.X` is
replaced by control cells, so the decoder sees a control profile plus the
perturbation label and cannot read the answer off the cell.

Usage: cpa_predict_all.py MODEL_DIR PARTS_DIR OUT.npz [SEED]
"""

import json
import pathlib
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

import cpa

model_dir, parts_dir, out_p = sys.argv[1:4]
parts_dir = pathlib.Path(parts_dir)
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0

z = np.load(parts_dir / "parts.npz", allow_pickle=False)
X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
genes = z["genes"].astype(str)
cond = z["condition"].astype(str)
split = json.loads((parts_dir / "split.json").read_text())

pert = np.array([c.split("+")[0] if c != "ctrl" else "ctrl" for c in cond])
in_split = {}
for name in ("train", "val", "test"):
    in_split.update({c.split("+")[0] if c != "ctrl" else "ctrl": name for c in split[name]})
in_split["ctrl"] = "train"
cell_split = np.array([in_split.get(p, "train") for p in pert])

adata = ad.AnnData(X, obs=pd.DataFrame({"condition": cond, "perturbation": pert,
                                        "split": cell_split,
                                        "dose": np.where(pert == "ctrl", 0.0, 1.0),
                                        "cell_type": "source"},
                                       index=[f"c{i}" for i in range(X.shape[0])]),
                   var=pd.DataFrame({"gene_name": genes}, index=genes))
print(f"  {X.shape[0]} cells x {X.shape[1]} genes", flush=True)

model = cpa.CPA.load(dir_path=model_dir, adata=adata, use_gpu=False)
print(f"  loaded {model_dir}", flush=True)

ctrl_rows = np.flatnonzero(pert == "ctrl")
rng = np.random.default_rng(SEED)
Xcf = X.astype(np.float32).toarray()
donor = rng.choice(ctrl_rows, size=Xcf.shape[0], replace=True)
adata.obsm["X_true"] = Xcf.copy()
adata.X = Xcf[donor]                        # every cell now carries control expression
del Xcf

model.predict(adata, batch_size=512)
pred = adata.obsm["CPA_pred"]
pred = np.asarray(pred.todense() if sp.issparse(pred) else pred, dtype=np.float64)

ctrl_mean = pred[ctrl_rows].mean(0)         # control rows kept the "ctrl" label
of_split = {p: in_split[p] for p in np.unique(pert) if p != "ctrl"}
rows, kept = [], []
for t in sorted(of_split):
    m = pert == t
    if m.sum():
        rows.append(pred[m].mean(0) - ctrl_mean)
        kept.append(t)
delta = np.stack(rows)
np.savez_compressed(out_p, genes=genes, targets=np.array(kept, dtype=str),
                    split=np.array([of_split[t] for t in kept], dtype=str),
                    delta=delta.astype(np.float32))
print(f"  wrote {out_p}: {delta.shape}", flush=True)
print("CPA_PREDICT_ALL_DONE")
