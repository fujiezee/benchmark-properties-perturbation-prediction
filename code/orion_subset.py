"""Restrict Orion HEK293T to the union of what any source pair can use.

The full array is 17,737 x 38,606 in three matrices, 8.2 GB in memory, and each
worker process loads its own copy -- enough to kill a pool on a 60 GB host. Every
pair intersects its source's targets and genes with Orion's anyway, so
restricting Orion to the union over all sources cannot change any pair's result:
each pair's intersection is a subset of that union.

Membership is tested with Python sets rather than np.isin, which on <U18 string
arrays builds intermediates large enough to OOM on its own, and the three
matrices are subset one at a time with the source released in between.
"""

import numpy as np

P = "/data/vcell/pilot039"
SOURCES = ["replogle/k562_gwps", "replogle/rpe1", "replogle/hepg2", "replogle/jurkat",
           "viperturb/k562_viperturb"]
OUT = "/data/vcell/raw/orion/orion_hek293t_subset.npz"

keep_t: set[str] = set()
keep_g: set[str] = set()
for s in SOURCES:
    z = np.load(f"{P}/{s}.npz", allow_pickle=False)
    keep_t |= set(z["targets"].tolist())
    keep_g |= set(z["genes"].tolist())
    z.close()

o = np.load("/data/vcell/raw/orion/orion_hek293t.npz", allow_pickle=False)
ot, og = o["targets"], o["genes"]
ti = np.array([i for i, t in enumerate(ot.tolist()) if t in keep_t])
gi = np.array([i for i, g in enumerate(og.tolist()) if g in keep_g])
print(f"  targets {len(ot)} -> {len(ti)},  genes {len(og)} -> {len(gi)}", flush=True)

out = {"targets": ot[ti], "genes": og[gi], "cells": o["cells"][ti]}
for k in ("delta", "half1", "half2"):
    a = o[k]
    out[k] = a[ti][:, gi]
    del a
    print(f"  {k}: {out[k].shape} {out[k].nbytes / 2**30:.2f} GB", flush=True)
o.close()
np.savez(OUT, **out)
print(f"  wrote {OUT}")
print("ORION_SUBSET_DONE")
