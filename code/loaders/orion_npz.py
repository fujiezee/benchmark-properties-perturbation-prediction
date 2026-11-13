"""X-Atlas/Orion (Xaira) HCT116 and HEK293T genome-wide screens as response tables.

A second lab with two cell types, which the other contexts cannot give (every
same-lab, other-type pair so far is Weissman's), and a second fixed-cell
chemistry (FiCS) beside Satija's Flex.

126 GB of parquet on Hugging Face, so each batch file is fetched, folded into
running group sums and deleted; only the sums stay in memory (targets x genes,
two halves, float32, preallocated for MAX_TARGETS; growing such an
accumulator by copy, or densifying a per-batch block, runs the box out of
memory, since a genome-wide batch touches nearly every target with about
one cell each. The full sums are the two half sums added, taken at the end
on the kept targets only. At most WORKERS * 2 files are on disk at a time. Cells with pass_guide_filter and one guide pair for one
gene; mean log1p(CP10k) per target minus Non-Targeting cells; halves split
cells at random. CP10k uses total_counts (whole transcriptome). Gene columns
are the dataset's gene_token_id, named from metadata/gene_metadata.parquet.

Usage: python orion_npz.py OUT_DIR SPLIT [WORKERS] [MIN_CELLS]
  SPLIT: HCT116 | HEK293T
"""

import json
import os
import subprocess
import sys
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq

SEED = 0
MAX_TARGETS = 24000  # 18,903 protein-coding genes targeted, plus controls
BASE = "https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Orion/resolve/main"
out_dir, split = sys.argv[1:3]
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
MIN_CELLS = int(sys.argv[4]) if len(sys.argv) > 4 else 30
work = f"{out_dir}/orion_work_{split}"
os.makedirs(work, exist_ok=True)

meta = pq.read_table(f"{out_dir}/gene_metadata.parquet").to_pydict()
n_genes = max(meta["gene_token_id"]) + 1
gene_name = np.array([""] * n_genes, dtype=object)
for tok, nm in zip(meta["gene_token_id"], meta["gene_name"]):
    gene_name[tok] = nm

with urllib.request.urlopen(f"https://huggingface.co/api/datasets/Xaira-Therapeutics/X-Atlas-Orion/tree/main/data?recursive=true") as r:
    files = sorted(f["path"] for f in json.load(r) if f["type"] == "file" and f["path"].split("/")[-1].startswith(split))
print(f"{split}: {len(files)} batch files, {n_genes} gene columns", flush=True)

targets: dict[str, int] = {"Non-Targeting": 0}
sums = [np.zeros((MAX_TARGETS, n_genes), np.float32) for _ in range(2)]  # half1, half2
counts = np.zeros((MAX_TARGETS, 3), np.int64)
print(f"accumulators: {2 * sums[0].nbytes / 1e9:.1f} GB", flush=True)
rng = np.random.default_rng(SEED)


def fetch(path: str) -> str:
    dest = f"{work}/{os.path.basename(path)}"
    if not os.path.exists(dest):
        subprocess.run(["curl", "-sL", "--fail", "--retry", "10", "--retry-delay", "10", "-m", "1800", "-o", dest + ".part", f"{BASE}/{path}"], check=True)
        os.replace(dest + ".part", dest)
    return dest


def fold(path: str) -> int:
    """Add one batch file to the running sums. Returns cells used.

    Values are prepared for the whole batch at once; the cells are then
    added one at a time, which keeps memory flat. Grouping the batch into
    a matrix first does not help: a genome-wide batch has roughly one cell
    per target, so the grouped block is as large as the accumulator.
    """
    t = pq.read_table(path, columns=["gene_token_id", "gene_expression", "gene_target", "num_features",
                                     "pass_guide_filter", "total_counts"])
    tok_col = t.column("gene_token_id").combine_chunks()
    exp_col = t.column("gene_expression").combine_chunks()
    gene_target = np.array(t.column("gene_target").to_pylist(), dtype=object)
    ok = ((t.column("pass_guide_filter").to_numpy() == 1) & (t.column("num_features").to_numpy() == 2)
          & np.array([g is not None and g != "" for g in gene_target]))
    cells = np.flatnonzero(ok)
    if not len(cells):
        os.remove(path)
        return 0
    for g in set(gene_target[cells]):
        if g not in targets:
            if len(targets) >= MAX_TARGETS:
                raise RuntimeError(f"more than {MAX_TARGETS} targets")
            targets[g] = len(targets)
    offs = tok_col.offsets.to_numpy()
    tok_flat = tok_col.values.to_numpy(zero_copy_only=False)
    exp_flat = exp_col.values.to_numpy(zero_copy_only=False)
    total = t.column("total_counts").to_numpy()
    half = rng.integers(0, 2, len(cells))
    row_of = np.array([targets[g] for g in gene_target[cells]])
    scale = (1e4 / total[cells]).astype(np.float64)
    for j, i in enumerate(cells):
        lo, hi = offs[i], offs[i + 1]
        sums[half[j]][row_of[j], tok_flat[lo:hi]] += np.log1p(exp_flat[lo:hi] * scale[j])
    np.add.at(counts[:, 1], row_of[half == 0], 1)
    np.add.at(counts[:, 2], row_of[half == 1], 1)
    os.remove(path)
    return len(cells)


done = 0
with ThreadPoolExecutor(WORKERS) as ex:
    # Bounded prefetch: fetch ahead a few files, fold in order (the sums are not thread-safe).
    queue = deque(ex.submit(fetch, f) for f in files[:2 * WORKERS])
    for i in range(len(files)):
        local = queue.popleft().result()
        if 2 * WORKERS + i < len(files):
            queue.append(ex.submit(fetch, files[2 * WORKERS + i]))
        fold(local)
        done += 1
        if done % 20 == 0 or done == len(files):
            # column 0 is only filled at the end, so count the halves here
            print(f"  {done}/{len(files)} batches, {int(counts[:, 1:].sum())} cells, {len(targets)} targets", flush=True)

names = [t for t, _ in sorted(targets.items(), key=lambda kv: kv[1])]
n_t = len(names)
counts = counts[:n_t]
counts[:, 0] = counts[:, 1] + counts[:, 2]
mean = [np.divide(sums[0][:n_t] + sums[1][:n_t], np.maximum(counts[:, 0], 1)[:, None]),
        np.divide(sums[0][:n_t], np.maximum(counts[:, 1], 1)[:, None]),
        np.divide(sums[1][:n_t], np.maximum(counts[:, 2], 1)[:, None])]
ctrl = [M[0] for M in mean]
rows = [i for i, t in enumerate(names) if i > 0 and counts[i, 0] >= MIN_CELLS and min(counts[i, 1], counts[i, 2]) >= MIN_CELLS // 3]
keep_cols = np.flatnonzero(gene_name != "")
np.savez_compressed(f"{out_dir}/orion_{split.lower()}.npz",
                    genes=gene_name[keep_cols].astype(str),
                    targets=np.array([names[i] for i in rows], dtype=str),
                    delta=np.array([mean[0][i][keep_cols] - ctrl[0][keep_cols] for i in rows], np.float32),
                    half1=np.array([mean[1][i][keep_cols] - ctrl[1][keep_cols] for i in rows], np.float32),
                    half2=np.array([mean[2][i][keep_cols] - ctrl[2][keep_cols] for i in rows], np.float32),
                    cells=counts[rows, 0].astype(np.int32))
print(f"wrote {out_dir}/orion_{split.lower()}.npz: {len(rows)} targets (median {int(np.median(counts[rows, 0]))} cells), "
      f"{len(keep_cols)} genes, {int(counts[0, 0])} control cells", flush=True)
