"""Train CPA on the same K562 cells, genes and split GEARS was given.

The venue assessment found the model comparison thin: of the seven transfer
models only `linear` carries a citation, and GEARS was the one other published
method. CPA (Lotfollahi et al. 2023) is the other canonical deep model for
perturbation response, so it goes in next.

Comparability is the point, so nothing about the data is re-decided here. This
reads `k562_parts.npz` and `split.json` exactly as `gears_train.py` does -- the
same 122,707 cells, the same 2,000 readout genes, the same 1,603/177/278
knockdown split -- and writes `cpa_pred.npz` with the same keys GEARS writes
(genes, targets, delta), so `gears_on_project_scale.py` scores both without
being touched.

CPA is given its package defaults. Predictions are the mean predicted expression
of a knockdown's cells minus the mean predicted expression of control cells, in
the same log space the deltas are measured in.

Usage: cpa_train.py DATA_DIR [EPOCHS] [OUT] [SEED]
"""

import json
import pathlib
import sys

import numpy as np
import scipy.sparse as sp

import anndata as ad
import cpa

data_dir = pathlib.Path(sys.argv[1])
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 50
OUT = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else data_dir / "cpa_pred.npz"
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0

# CPAModule takes `seed` (default 0) and calls torch.manual_seed with it at
# construction, overriding anything set on scvi.settings beforehand. So the seed
# has to be passed to the model, where `**hyper_params` forwards it. Setting
# scvi.settings.seed instead looks like it works -- the log even says "Global
# seed set to 1" -- and produces a bit-identical duplicate run.

z = np.load(data_dir / "k562_parts.npz", allow_pickle=False)
X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
cond, genes = z["condition"].astype(str), z["genes"].astype(str)
split = json.loads((data_dir / "split.json").read_text())
print(f"  {X.shape[0]:,} cells x {X.shape[1]} genes", flush=True)
print(f"  split: {len(split['train'])} train, {len(split['val'])} val, {len(split['test'])} test", flush=True)

# GEARS names conditions "GENE+ctrl"; CPA wants a plain perturbation label.
pert = np.array([c.split("+")[0] if c != "ctrl" else "ctrl" for c in cond])
in_split = {}
for name in ("train", "val", "test"):
    in_split.update({c.split("+")[0] if c != "ctrl" else "ctrl": name for c in split[name]})
# control cells train with everything
in_split["ctrl"] = "train"
cell_split = np.array([in_split.get(p, "train") for p in pert])

adata = ad.AnnData(X=X.astype(np.float32))
adata.var_names = genes
adata.obs["condition"] = pert
adata.obs["split"] = cell_split
adata.obs["dose"] = np.where(pert == "ctrl", 0.0, 1.0)
adata.obs["cov"] = "K562"
print(f"  cells per split: {dict(zip(*np.unique(cell_split, return_counts=True)))}", flush=True)

# CPA trains on train+val and holds the test knockdowns out entirely.
cpa.CPA.setup_anndata(
    adata,
    perturbation_key="condition",
    control_group="ctrl",
    dosage_key="dose",
    categorical_covariate_keys=["cov"],
    is_count_data=False,
    max_comb_len=1,
)

model = cpa.CPA(adata=adata, split_key="split", train_split="train",
                valid_split="val", test_split="test", seed=SEED)
print("  training...", flush=True)
model.train(max_epochs=EPOCHS, use_gpu=False, batch_size=512,
            early_stopping_patience=10, check_val_every_n_epoch=5)

# Counterfactual prediction, following CPA's own predict() docstring:
#   "To produce counterfactuals, save adata.X to adata.obsm['X_true'] and set it
#    to control cells gene expression. For the case of reconstruction, you can
#    pass original adata without any further modifications."
# Passing the unmodified adata -- which an earlier version of this script did --
# decodes each observed cell's own latent code, so a held-out knockdown's
# "prediction" carries that knockdown's measured expression and scoring it
# against the measurement scores the model against itself. It reached 0.90
# specific that way, against ridge's 0.65, which is what gave it away.
# GEARS predicts from perturbation identity alone, so only the counterfactual is
# comparable: feed control expression, keep the perturbation label.
print("  counterfactual prediction...", flush=True)
model.save("/data/vcell/cpa/model_seed%d" % SEED, overwrite=True)

ctrl_rows = np.flatnonzero(pert == "ctrl")
rng = np.random.default_rng(SEED)
Xcf = X.astype(np.float32).toarray() if sp.issparse(X) else np.array(X, dtype=np.float32)
donor = rng.choice(ctrl_rows, size=Xcf.shape[0], replace=True)
adata.obsm["X_true"] = Xcf.copy()
adata.X = Xcf[donor]          # every cell now carries control expression
del Xcf

model.predict(adata, batch_size=512)
pred = adata.obsm["CPA_pred"]
if sp.issparse(pred):
    pred = np.asarray(pred.todense())
pred = np.asarray(pred, dtype=np.float64)

# control rows kept the "ctrl" label, so their prediction is the control baseline
ctrl_mean = pred[ctrl_rows].mean(0)
targets = sorted({c.split("+")[0] for c in split["test"] if c != "ctrl"})
rows, kept = [], []
for t in targets:
    m = pert == t
    if m.sum() == 0:
        continue
    rows.append(pred[m].mean(0) - ctrl_mean)
    kept.append(t)
delta = np.stack(rows)
print(f"  {len(kept)} of {len(targets)} test knockdowns predicted", flush=True)

np.savez_compressed(OUT, genes=genes, targets=np.array(kept, dtype=str), delta=delta.astype(np.float32))
print(f"  wrote {OUT}: {delta.shape}")
print("CPA_TRAIN_DONE")
