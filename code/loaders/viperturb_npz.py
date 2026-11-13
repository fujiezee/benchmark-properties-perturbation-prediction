"""Turn viperturb.R's float32 output into a response table like the other contexts.

Usage: python viperturb_npz.py PREFIX
"""

import sys

import numpy as np

prefix = sys.argv[1]
groups = [x for x in open(f"{prefix}.groups.txt").read().split("\n") if x]
cells = np.array([float(x) for x in open(f"{prefix}.cells.txt").read().split()])
genes = [x for x in open(f"{prefix}.genes.txt").read().split("\n") if x]
k = len(groups)
M = np.fromfile(f"{prefix}.f32", dtype=np.float32).reshape(3 * k, len(genes))  # one row per group
full, h1, h2 = M[:k], M[k : 2 * k], M[2 * k :]
np.savez_compressed(
    f"{prefix}.npz", genes=np.array(genes), targets=np.array(groups[1:]),
    delta=full[1:] - full[0], half1=h1[1:] - h1[0], half2=h2[1:] - h2[0], cells=cells[1:k].astype(np.int32))
print(f"wrote {prefix}.npz: {k - 1} targets x {len(genes)} genes, control cells {int(cells[0])}")
