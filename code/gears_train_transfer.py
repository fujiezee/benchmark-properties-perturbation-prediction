"""Train GEARS on a whole source screen and predict every knockdown it measured.

`gears_train.py` trains on K562's train split and writes predictions for the 278
held-out knockdowns only, and it does not save the model. Neither suits the
zero-shot transfer comparison, which needs

  * predictions for knockdowns the model *trained on*, because the baselines it
    competes with (`copy`, `linear`) are handed those knockdowns' measured source
    response, and a model asked to predict unseen knockdowns from the gene graph
    alone is being asked a harder question -- the asymmetry that made the first
    run of `zeroshot.py` uninterpretable;
  * a saved model, so a second target context costs a prediction pass rather than
    a retrain.

Holding out the target context instead of the knockdown is what makes predicting
training knockdowns legitimate here: no target-context measurement reaches the
model, which is the only thing being withheld.

`split` is written alongside `targets` so the scorer can select on it.

Usage: gears_train_transfer.py PARTS_DIR OUT.npz [EPOCHS] [THREADS]
"""

import json
import pathlib
import pickle
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from gears import GEARS, PertData

parts_dir = pathlib.Path(sys.argv[1])
out_p = sys.argv[2]
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 15
torch.set_num_threads(int(sys.argv[4]) if len(sys.argv) > 4 else 16)
torch.manual_seed(0)
np.random.seed(0)

z = np.load(parts_dir / "parts.npz", allow_pickle=False)
X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
genes = z["genes"].astype(str)
cond = z["condition"].astype(str)
adata = ad.AnnData(X, obs=pd.DataFrame({"condition": cond, "cell_type": "source"},
                                       index=[f"c{i}" for i in range(X.shape[0])]),
                   var=pd.DataFrame({"gene_name": genes}, index=genes))
print(f"  {X.shape[0]} cells x {X.shape[1]} genes", flush=True)

work = parts_dir / "gears_work"
pd_ = PertData(str(work))
if not (work / "src" / "data_pyg" / "cell_graphs.pkl").exists():
    pd_.new_data_process(dataset_name="src", adata=adata)
pd_.load(data_path=str(work / "src"))

# GEARS keeps a per-condition gene filter only for conditions it processed fully;
# conditions it dropped break its loss, so they are removed from the split.
full_to_cond = dict(pd_.adata.obs[["condition_name", "condition"]].astype(str).values)
valid = {full_to_cond[k] for k in pd_.adata.uns["non_zeros_gene_idx"] if k in full_to_cond} | {"ctrl"}
split = {k: [c for c in v if c in valid] for k, v in
         json.loads((parts_dir / "split.json").read_text()).items()}
# GEARS requires a non-empty test split; the real holdout is the target context,
# so val is reused here and the predictions are selected by `split` later.
if not split["test"]:
    split["test"] = list(split["val"])
print(f"  conditions kept per split: {({k: len(v) for k, v in split.items()})}", flush=True)
with open(parts_dir / "split.pkl", "wb") as fh:          # GEARS reads a custom split as a pickle
    pickle.dump(split, fh)
pd_.prepare_split(split="custom", seed=0, split_dict_path=str(parts_dir / "split.pkl"))
pd_.get_dataloader(batch_size=32, test_batch_size=128)

model = GEARS(pd_, device="cpu")
model.model_initialize(hidden_size=64)
model.train(epochs=EPOCHS, lr=1e-3)
model.save_model(str(parts_dir / "gears_model"))
print(f"  saved model to {parts_dir / 'gears_model'}", flush=True)

of_split = {c.split("+")[0]: name for name in ("train", "val") for c in split[name] if c != "ctrl"}
known = set(model.pert_list)
want = [t for t in sorted(of_split) if t in known]
print(f"  {len(want)} of {len(of_split)} knockdowns in GEARS' graph", flush=True)

ctrl_mean = np.asarray(adata[adata.obs["condition"] == "ctrl"].X.mean(0)).ravel()
rows, kept = [], []
for s in range(0, len(want), 64):                        # predict() holds all cells in memory
    chunk = want[s:s + 64]
    pred = model.predict([[t] for t in chunk])
    for t in chunk:
        rows.append(np.asarray(pred[t]).ravel() - ctrl_mean)
        kept.append(t)
delta = np.vstack(rows)
np.savez_compressed(out_p, genes=genes, targets=np.array(kept, dtype=str),
                    split=np.array([of_split[t] for t in kept], dtype=str),
                    delta=delta.astype(np.float32))
print(f"  wrote {out_p}: {delta.shape}", flush=True)
print("GEARS_TRANSFER_DONE")
