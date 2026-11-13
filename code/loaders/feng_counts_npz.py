"""Feng et al. 2026 (Sanger, iPSC CRISPRi, many donors) from raw counts, per cell line and pooled.

The authors' LFC tables have no split halves, so the pilot's Feng numbers
could not be noise-corrected like the other contexts. This rebuilds them
from the dense count CSVs (genes x cells, figshare 27989294 via the
Boom5426/iPSC-CRISPRi-PerturbSeq mirror, md5 checked): per cell line
(HipSci donor_clone, e.g. kolf_2) and pooled over lines, mean log1p(CP10k)
of cells with one guide for the target minus cells with a non-targeting
guide in the same line; two random halves of cells. The genome-wide screens
have almost no non-targeting cells (36 and 12), so there the reference is
every assigned cell of the line (most knockdowns do nothing; one target is
under 1% of them), as in gasperini_npz.py. CP10k uses the cell's
total over all Gene-Expression rows (pass 1), then group sums (pass 2).
Matrix cell IDs drop the metadata's prefix: "P1_I1_..." is "PC-P1-D3_I1_..."
(targeted screen), "ST-P3-..." is "MP-ST-P3-..." (genome-wide).

Also writes two donor groups: the lines with the most cells dealt
alternately into group A and B, each pooled. Same lab, same protocol, same
cell type, different humans, each group thick enough to be noise-corrected,
which no single line is (about 100 cells per target against 1,355 pooled).

Usage: python feng_counts_npz.py COUNTS_DIR SCREEN OUT_DIR [WORKERS] [MIN_CELLS]
  SCREEN: TargetedScreen | GenomeWideScreen_FitnessGenes | GenomeWideScreen_NonFitnessGenes
"""

import gzip
import re
import shutil
import subprocess
import pathlib
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np

SEED = 0
CHUNK = 16  # gene rows per task
counts_dir, screen, out_dir = sys.argv[1:4]
WORKERS = int(sys.argv[4]) if len(sys.argv) > 4 else 30
MIN_CELLS = int(sys.argv[5]) if len(sys.argv) > 5 else 20


def stream(path: str):
    """Lines of a .gz, decompressed by pigz when present (it reads and writes in other threads)."""
    if shutil.which("pigz"):
        p = subprocess.Popen(["pigz", "-dc", path], stdout=subprocess.PIPE, bufsize=1 << 24)
        yield from p.stdout
        p.wait()
    else:
        with gzip.open(path, "rb") as f:
            yield from f


def parse(row: bytes) -> np.ndarray:
    """Comma-separated small non-negative integers to int32, vectorized; multi-digit fields done one by one."""
    b = np.frombuffer(row, np.uint8)
    comma = np.flatnonzero(b == 44)
    starts = np.r_[0, comma + 1]
    ends = np.r_[comma, len(b)]
    v = b[starts].astype(np.int32) - 48
    for i in np.flatnonzero(ends - starts > 1):
        v[i] = int(row[starts[i]:ends[i]])
    return v


def split_row(line: bytes) -> tuple[bytes, bytes]:
    line = line.rstrip(b"\r\n")
    k = line.index(b",")
    return line[:k], line[k + 1:]


# Worker state, set once per process.
_state = {}


def init(order: np.ndarray, code: np.ndarray | None, scale: np.ndarray | None, n_codes: int) -> None:
    _state.update(order=order, code=code, scale=scale, n_codes=n_codes)


def totals_task(rows: list[bytes]) -> np.ndarray:
    s = None
    for r in rows:
        v = parse(r)[_state["order"]]
        s = v.astype(np.int64) if s is None else s + v
    return s


def sums_task(job: tuple[list[int], list[bytes]]) -> tuple[list[int], np.ndarray]:
    idx, rows = job
    code, scale, n = _state["code"], _state["scale"], _state["n_codes"]
    out = np.zeros((len(rows), n), np.float64)
    for j, r in enumerate(rows):
        v = parse(r)[_state["order"]]
        nz = np.flatnonzero(v)
        val = np.log1p(v[nz] * scale[nz])
        out[j] = np.bincount(code[nz], val, minlength=n)
    return idx, out


def run(tasks, fn, workers: int, initargs, on_result) -> None:
    with ProcessPoolExecutor(workers, initializer=init, initargs=initargs) as ex:
        pending = set()
        for t in tasks:
            pending.add(ex.submit(fn, t))
            if len(pending) >= 2 * workers:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for f in done:
                    on_result(f.result())
        for f in pending:
            on_result(f.result())


def chunks(path: str, with_names: bool):
    lines = stream(path)
    next(lines)  # header
    names, rows = [], []
    for line in lines:
        name, row = split_row(line)
        names.append(name)
        rows.append(row)
        if len(rows) == CHUNK:
            yield (names, rows) if with_names else rows
            names, rows = [], []
    if rows:
        yield (names, rows) if with_names else rows


if __name__ == "__main__":
    rna = f"{counts_dir}/{screen}_RNA-UMI-Counts.csv.gz"
    with gzip.open(rna, "rt") as f:
        header = f.readline().rstrip("\n").split(",")[1:]
    prefix = (lambda c: re.sub(r"^P(\d+)_", r"PC-P\1-D3_", c)) if screen == "TargetedScreen" else (lambda c: "MP-" + c)
    col = {prefix(c.strip('"')): i for i, c in enumerate(header)}
    meta = {}
    with gzip.open(f"{counts_dir}/{screen}_Cell-Metadata.tsv.gz", "rt") as f:
        next(f)
        for line in f:
            cid, _, guide, cell_line = line.rstrip("\n").split("\t")
            meta[cid] = (guide, cell_line)
    cells = [c for c in meta if c in col]
    print(f"{screen}: {len(header)} matrix columns, {len(meta)} metadata cells, {len(cells)} matched", flush=True)

    def target_of(guide: str) -> str:
        if guide == "unassigned" or ";" in guide or "," in guide:
            return ""
        g = guide.rsplit("_", 1)[0]
        return "__control__" if g.lower().startswith("nontarget") or g.lower().startswith("non-target") else g

    order = np.array([col[c] for c in cells])
    target = np.array([target_of(meta[c][0]) for c in cells], dtype=object)
    line_of = np.array([meta[c][1] for c in cells], dtype=object)
    lines = sorted(set(line_of))
    targets = sorted(set(target) - {"", "__control__"})
    groups = ["__control__", *targets]
    gi = {g: i for i, g in enumerate(groups)}
    li = {l: i for i, l in enumerate(lines)}
    keep = target != ""
    G = len(lines) * len(groups)
    half = np.random.default_rng(SEED).integers(0, 2, len(cells))
    base = np.array([li[l] * len(groups) + gi.get(t, 0) for l, t in zip(line_of, target)])
    code = np.where(keep, base, -1)
    # Codes: [0, G) full, [G, 3G) halves: G + 2 * base + half. Dropped cells go to a trash slot at the end.
    n_codes = 3 * G + 1
    full_code = np.where(keep, base, 3 * G)
    half_code = np.where(keep, G + 2 * base + half, 3 * G)
    print(f"{len(lines)} lines, {len(targets)} targets, {keep.sum()} assigned cells ({(target == '__control__').sum()} non-targeting)", flush=True)

    # Pass 1: totals per cell (matrix column order -> our cell order).
    tot = [np.zeros(len(cells), np.int64)]
    run(chunks(rna, False), totals_task, WORKERS, (order, None, None, 0), lambda r: tot.__setitem__(0, tot[0] + r))
    tot = tot[0]
    print(f"pass 1 done: median total {int(np.median(tot))}", flush=True)
    scale = 1e4 / np.maximum(tot, 1)

    # Pass 2: group sums of log1p(CP10k). bincount over two code sets per row, so tasks return (genes x n_codes).
    gene_names: list[str] = []
    counter = [0]

    def tasks():
        for names, rows in chunks(rna, True):
            idx = list(range(counter[0], counter[0] + len(rows)))
            counter[0] += len(rows)
            gene_names.extend(n.decode().split(":")[1] if n.count(b":") >= 2 else n.decode() for n in names)
            yield idx, rows

    # Each cell sits in a full group and a half group: index every cell twice.
    order2 = np.r_[order, order]
    code2 = np.r_[full_code, half_code]
    scale2 = np.r_[scale, scale]

    # Pass 2's output is genes x 3G, and at G = 56,174 groups a row is 0.67 MB:
    # the first version accumulated rows in a dict and then vstacked them, which
    # held two copies and peaked near 51 GB on a 64 GB box, and the kernel took it
    # at 62 GB. Growing one in-RAM array geometrically is no better -- doubling
    # past 32,768 rows allocates 44 GB while still holding 22 GB.
    # It goes to disk instead: rows are written straight into a memory-mapped .npy
    # sized above any gene count these screens reach, and the pages that are never
    # touched are never materialised, so RAM holds only what the OS caches.
    spill = f"{out_dir}/.pass2_{screen}.npy"
    S = np.lib.format.open_memmap(spill, mode="w+", dtype=np.float32, shape=(1 << 17, 3 * G))
    written = [0]

    def add_sums(r):
        idx, block = r
        for i, row in zip(idx, block):
            S[i] = row[:3 * G]
        written[0] = max(written[0], idx[-1] + 1)

    run(tasks(), sums_task, WORKERS, (order2, code2, scale2, n_codes), add_sums)
    S = S[:written[0]]  # genes x 3G
    print(f"pass 2 done: {S.shape[0]} genes", flush=True)

    n_full = np.bincount(full_code, minlength=n_codes)[:3 * G]
    n_half = np.bincount(half_code, minlength=n_codes)[:3 * G]
    n = n_full[:G] + 0
    nh = n_half[G:3 * G]
    genes = np.array(gene_names)
    # Duplicate symbols (rare): keep the first. `S[first]` would fancy-index the
    # memmap into a full in-RAM copy (24.6 GB on the non-fitness screen), so the
    # kept rows are moved down in place instead.
    _, first = np.unique(genes, return_index=True)
    first = np.sort(first)
    for i, j in enumerate(first):
        if i != j:
            S[i] = S[j]
    genes, S = genes[first], S[:len(first)]

    # A column of S is addressed as `line * len(groups) + group` for the full
    # sums, and `G + 2 * that + half` for the halves, so the array already *is*
    # a (gene, line, group) table and a reshape exposes it. Gathering those
    # columns per target instead -- 34 scattered columns over 36,518 rows, once
    # per target, 4,959 times -- read 497 GB off a 25 GB memory map in an hour
    # and wrote no output, because each element of a gather sits on its own page.
    # One strided pass per file replaces all of it.
    n_lines, n_groups = len(lines), len(groups)
    full_view = S[:, :G].reshape(S.shape[0], n_lines, n_groups)
    half_view = S[:, G:3 * G].reshape(S.shape[0], n_lines, n_groups, 2)
    n_view = n.reshape(n_lines, n_groups)
    nh_view = nh.reshape(n_lines, n_groups, 2)

    # One line's columns are strided by construction -- 19.84 KB of each row,
    # 2.02 MB apart -- so reading them touches the whole spill. That is cheap for
    # one line (its own 2.17 GB, plus what fault-around overshoots) and ruinous
    # for the pooled file, which wants all 34: the map is 68.8 GB against a 44 GB
    # ceiling, so nothing survives from one line's pass to the next and the disk,
    # measured at 183 MB/s with dd iflag=direct, has to serve every line in full.
    # The pooled file reads gene blocks instead: each block is one contiguous
    # read, every byte arrives once, and the striding happens in memory where it
    # is free. Lines are still accumulated one at a time inside the block, in the
    # same order, so the arithmetic is the one already verified against baseline.
    BLOCK = 64

    def counts(line_idx: list[int]) -> tuple:
        """Cell counts per group for these lines. Reads `n` and `nh`, never S."""
        cF = np.zeros(n_groups, np.int64)
        cH = np.zeros((n_groups, 2), np.int64)
        for l in line_idx:
            cF += n_view[l]
            cH += nh_view[l]
        return cF, cH

    def tables(line_idx: list[int]) -> tuple:
        """Every group's full and half sums for these lines, in one pass over S."""
        n_genes = S.shape[0]
        F = np.zeros((n_genes, n_groups), np.float64)
        H = np.zeros((n_genes, n_groups, 2), np.float64)
        if len(line_idx) == 1:
            l = line_idx[0]
            F += full_view[:, l, :]
            H += half_view[:, l, :, :]
            return F, H
        for i in range(0, n_genes, BLOCK):
            blk = np.array(S[i:i + BLOCK, :])          # contiguous, read once
            b = blk.shape[0]
            full = blk[:, :G].reshape(b, n_lines, n_groups)
            half = blk[:, G:3 * G].reshape(b, n_lines, n_groups, 2)
            for l in line_idx:
                F[i:i + b] += full[:, l, :]
                H[i:i + b] += half[:, l, :, :]
        return F, H

    def write(path: str, line_idx: list[int]) -> None:
        def mean(f, h1, h2, c, c1, c2):
            return f / max(c, 1), h1 / max(c1, 1), h2 / max(c2, 1), c, min(c1, c2)
        cF, cH = counts(line_idx)
        # Which targets survive is settled by cell counts, and those live in `n`
        # and `nh`, not in S. Settling it first means a line that keeps nothing --
        # 18 of 24 on the fitness screen, and every one of the non-fitness lines
        # so far -- writes its file without a pass over S, where before it paid a
        # full strided scan of the spill for contents that do not depend on it.
        keep_i = [t_i for t_i in range(1, len(groups))
                  if cF[t_i] >= MIN_CELLS and min(cH[t_i, 0], cH[t_i, 1]) >= MIN_CELLS // 3]
        ctrl_cells = int(cF[0] if screen == "TargetedScreen" else cF.sum())
        kept, D, H1, H2, C = [], [], [], [], []
        if keep_i:
            F, H = tables(line_idx)
            # The control term pools every group on the genome-wide screens, where the
            # untargeted cells are spread over them, and group 0 alone on the targeted one.
            if screen == "TargetedScreen":
                ctrl = mean(F[:, 0], H[:, 0, 0], H[:, 0, 1], cF[0], cH[0, 0], cH[0, 1])
            else:
                ctrl = mean(F.sum(1), H[:, :, 0].sum(1), H[:, :, 1].sum(1),
                            cF.sum(), cH[:, 0].sum(), cH[:, 1].sum())
            for t_i in keep_i:
                m = mean(F[:, t_i], H[:, t_i, 0], H[:, t_i, 1], cF[t_i], cH[t_i, 0], cH[t_i, 1])
                kept.append(groups[t_i])
                D.append(m[0] - ctrl[0]); H1.append(m[1] - ctrl[1]); H2.append(m[2] - ctrl[2]); C.append(m[3])
        np.savez_compressed(path, genes=genes.astype(str), targets=np.array(kept, dtype=str),
                            delta=np.array(D, np.float32), half1=np.array(H1, np.float32), half2=np.array(H2, np.float32),
                            cells=np.array(C, np.int32))
        print(f"wrote {path}: {len(kept)} targets (median {int(np.median(C)) if C else 0} cells), {ctrl_cells} control cells", flush=True)

    tag = {"TargetedScreen": "targeted", "GenomeWideScreen_FitnessGenes": "gw_fitness", "GenomeWideScreen_NonFitnessGenes": "gw_nonfitness"}[screen]
    write(f"{out_dir}/feng_{tag}_pooled.npz", list(range(len(lines))))
    per_line = np.bincount(np.array([li[l] for l in line_of])[keep], minlength=len(lines))
    ranked = np.argsort(-per_line)
    write(f"{out_dir}/feng_{tag}_donorsA.npz", ranked[0::2].tolist())
    write(f"{out_dir}/feng_{tag}_donorsB.npz", ranked[1::2].tolist())
    print("group A:", [lines[i] for i in ranked[0::2]], flush=True)
    print("group B:", [lines[i] for i in ranked[1::2]], flush=True)
    for l in lines:
        write(f"{out_dir}/feng_{tag}_{l}.npz", [li[l]])
    S = None  # drop the mapping before removing the file behind it
    pathlib.Path(spill).unlink(missing_ok=True)
