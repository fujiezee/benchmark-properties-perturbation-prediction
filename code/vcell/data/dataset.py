"""Turn raw Perturb-seq counts into a split, normalized dataset and load it back."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from vcell import files
from vcell.config import Paths

TARGET_COL = "target"
TOTAL_COUNTS_COL = "total_counts"
SPLIT_COL = "split"
CONTROL_SPLIT = "control"
SPLITS = ("train", "val", "test")
# Splits whose cells are scored, so `counts.h5ad` keeps their raw counts.
SCORED_SPLITS = ("val", "test")


@dataclass
class BuildOptions:
    # None keeps every measured gene, as the challenge's own panel does.
    n_hvg: int | None = None
    min_cells: int = 20
    split_fractions: tuple[float, float, float] = (0.7, 0.15, 0.15)
    coexpression_dims: int = 32
    seed: int = 0


def normalize_log1p(adata: ad.AnnData) -> None:
    """Library-size normalize to 10k counts per cell, then log1p (in place).

    Uses `obs["total_counts"]` as the library size when present, so a matrix
    already restricted to a gene subset is scaled by each cell's full library.
    """
    x = sp.csr_matrix(adata.X, dtype=np.float32)
    totals = (
        adata.obs[TOTAL_COUNTS_COL].to_numpy(dtype=np.float32)
        if TOTAL_COUNTS_COL in adata.obs
        else np.asarray(x.sum(axis=1), dtype=np.float32).ravel()
    )
    scale = np.divide(1e4, totals, out=np.zeros_like(totals), where=totals > 0)
    x.data *= np.repeat(scale, np.diff(x.indptr))
    np.log1p(x.data, out=x.data)
    adata.X = x


def coexpression_embedding(control: ad.AnnData, dims: int, seed: int) -> np.ndarray:
    """Embed every gene by its loadings on the top PCs of control-cell expression.

    Uses only unperturbed cells, so it is available for any gene, including
    targets that are never seen perturbed during training.
    """
    x = control.X.toarray() if sp.issparse(control.X) else np.asarray(control.X)
    x = x - x.mean(axis=0)
    std = x.std(axis=0)
    x = x / np.where(std > 0, std, 1.0)
    rng = np.random.default_rng(seed)
    if x.shape[0] > 5000:
        x = x[rng.choice(x.shape[0], 5000, replace=False)]
    dims = min(dims, min(x.shape) - 1)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    return (vt[:dims].T * s[:dims] / np.sqrt(x.shape[0])).astype(np.float32)


def split_perturbations(targets: list[str], fractions: tuple[float, float, float], seed: int) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(sorted(targets))
    n_train = round(len(order) * fractions[0])
    n_val = round(len(order) * fractions[1])
    out = {t: "train" for t in order[:n_train]}
    out |= {t: "val" for t in order[n_train : n_train + n_val]}
    out |= {t: "test" for t in order[n_train + n_val :]}
    return out


def select_readout(
    raw: ad.AnnData, *, control_label: str, options: BuildOptions | None = None
) -> tuple[list[str], tuple[np.ndarray, np.ndarray]]:
    """Pick readout genes and embed every gene by control-cell co-expression.

    Readout genes are the top highly variable genes plus every perturbed gene
    that is measured, or every gene when `options.n_hvg` is None. `raw`
    (integer counts, all genes) is normalized in place; a small pilot
    subsample is enough.
    """
    opts = options or BuildOptions()
    raw.var_names_make_unique()
    normalize_log1p(raw)
    control = raw[raw.obs[TARGET_COL] == control_label]
    emb = coexpression_embedding(control, opts.coexpression_dims, opts.seed)

    if opts.n_hvg is None:
        return list(raw.var_names), (np.asarray(raw.var_names, dtype=str), emb)

    hvg = sc.pp.highly_variable_genes(raw, n_top_genes=opts.n_hvg, flavor="seurat", inplace=False)
    readout = set(raw.var_names[hvg["highly_variable"].to_numpy()])
    readout |= set(raw.obs[TARGET_COL].astype(str)) & set(raw.var_names)
    ordered = [g for g in raw.var_names if g in readout]
    return ordered, (np.asarray(raw.var_names, dtype=str), emb)


def build_dataset(
    raw: ad.AnnData,
    paths: Paths,
    *,
    control_label: str,
    description: dict,
    options: BuildOptions | None = None,
    readout: tuple[list[str], tuple[np.ndarray, np.ndarray]] | None = None,
    counts_path: Path | None = None,
) -> None:
    """Normalize raw counts, split perturbations, and save cells plus pseudobulk.

    `raw` must hold integer counts with gene symbols as `var_names` and the
    perturbed gene symbol in `obs["target"]` (`control_label` for controls).
    `readout` is the output of `select_readout`; when omitted it is computed
    from `raw`. `raw` is modified in place.

    `counts_path` writes the scored cells' integer counts here before
    normalizing, for a screen whose counts cannot be cheaply read a second
    time; otherwise they are re-read later by `save_counts`.
    """
    opts = options or BuildOptions()
    if readout is None:
        readout = select_readout(raw.copy(), control_label=control_label, options=opts)
    genes, (coexp_genes, emb) = readout

    raw.var_names_make_unique()
    if TOTAL_COUNTS_COL not in raw.obs:
        raw.obs[TOTAL_COUNTS_COL] = np.asarray(raw.X.sum(axis=1), dtype=np.float32).ravel()
    counts = raw.obs[TARGET_COL].value_counts()
    keep = sorted(t for t, n in counts.items() if t != control_label and n >= opts.min_cells)
    rows = raw.obs[TARGET_COL].isin([*keep, control_label]).to_numpy()
    cells = raw[rows, genes] if not (rows.all() and list(raw.var_names) == genes) else raw
    cells = cells.copy() if cells.is_view else cells

    # The split needs only the kept targets and the seed, so it is settled here,
    # while the counts are still integers.
    split = split_perturbations(keep, opts.split_fractions, opts.seed)
    if counts_path is not None:
        _write_counts(cells, split, control_label, counts_path)
    normalize_log1p(cells)

    targets = cells.obs[TARGET_COL].astype(str).to_numpy()
    cells.obs = pd.DataFrame(
        {TARGET_COL: targets, SPLIT_COL: [split.get(t, CONTROL_SPLIT) for t in targets]},
        index=cells.obs_names,
    )
    cells.var = pd.DataFrame(index=cells.var_names)

    labels = [control_label, *keep]
    codes = pd.Categorical(targets, categories=labels).codes
    indicator = sp.csr_matrix(
        (np.ones(len(codes), dtype=np.float32), (codes, np.arange(len(codes)))), shape=(len(labels), len(codes))
    )
    sizes = np.asarray(indicator.sum(axis=1)).ravel()
    means = np.asarray((indicator @ cells.X).todense()) / sizes[:, None]

    paths.root.mkdir(parents=True, exist_ok=True)
    with files.replacing(paths.cells) as tmp:
        cells.write_h5ad(tmp, compression="lzf")
    with files.replacing(paths.coexpression) as tmp:
        np.savez_compressed(tmp, genes=coexp_genes, embedding=emb)
    with files.replacing(paths.pseudobulk) as tmp:
        np.savez_compressed(
            tmp,
            genes=np.asarray(cells.var_names, dtype=str),
            labels=np.asarray(labels, dtype=str),
            means=means.astype(np.float32),
        )
    meta = {
        **description,
        "control_label": control_label,
        "n_cells": int(cells.n_obs),
        "n_genes": int(cells.n_vars),
        "perturbations": [{"target": t, "split": split[t], "n_cells": int(counts[t])} for t in keep],
        "options": {**opts.__dict__, "split_fractions": list(opts.split_fractions)},
    }
    files.write_text(paths.meta, json.dumps(meta, indent=2))


def scored_cell_ids(paths: Paths) -> list[str]:
    """Ids of the cells `counts.h5ad` holds: every val, test and control cell in `cells.h5ad`."""
    obs = ad.read_h5ad(paths.cells, backed="r").obs
    return obs.index[obs[SPLIT_COL].isin([*SCORED_SPLITS, CONTROL_SPLIT])].astype(str).tolist()


def _write_counts(cells: ad.AnnData, split: dict[str, str], control_label: str, path: Path) -> None:
    """The scored cells' integer counts, as `save_counts` would have written them."""
    targets = cells.obs[TARGET_COL].astype(str).to_numpy()
    splits = np.array([split.get(t, CONTROL_SPLIT) for t in targets])
    rows = np.isin(splits, [*SCORED_SPLITS, CONTROL_SPLIT])
    out = cells[rows].copy()
    out.X = sp.csr_matrix(out.X, dtype=np.float32)
    out.obs = pd.DataFrame(
        {TARGET_COL: targets[rows], SPLIT_COL: splits[rows]},
        index=cells.obs_names[rows],
    )
    out.var = pd.DataFrame(index=cells.var_names)
    path.parent.mkdir(parents=True, exist_ok=True)
    with files.replacing(path) as tmp:
        out.write_h5ad(tmp, compression="lzf")


def save_counts(counts: ad.AnnData, paths: Paths) -> None:
    """Write raw counts of the scored cells on the readout genes to `counts.h5ad`.

    `counts` holds integer counts for at least those cells and genes, keyed by the
    same cell ids as `cells.h5ad`.
    """
    cells = ad.read_h5ad(paths.cells, backed="r")
    obs = cells.obs[cells.obs[SPLIT_COL].isin([*SCORED_SPLITS, CONTROL_SPLIT])]
    out = counts[obs.index, cells.var_names].copy()
    out.X = sp.csr_matrix(out.X, dtype=np.float32)
    out.obs = obs.copy()
    out.var = pd.DataFrame(index=cells.var_names)
    with files.replacing(paths.counts) as tmp:
        out.write_h5ad(tmp, compression="lzf")


@dataclass
class PerturbationDataset:
    """A prepared dataset: per-target mean expression plus lazily loaded cells.

    Training and serving use only the pseudobulk means; single cells are read
    from disk the first time `adata` is touched (evaluation).
    """

    paths: Paths
    meta: dict
    genes: np.ndarray
    pseudobulk: dict[str, np.ndarray]
    coexpression_genes: np.ndarray
    coexpression: np.ndarray
    # Prior-knowledge embeddings by kind ("go"), rows aligned with `knowledge_genes`.
    knowledge_genes: np.ndarray
    knowledge: dict[str, np.ndarray]
    # The other cell lines' averaged responses (see vcell.data.cross), or None.
    cross: dict[str, np.ndarray] | None = None

    @classmethod
    def load(cls, paths: Paths) -> PerturbationDataset:
        if not paths.pseudobulk.exists():
            raise FileNotFoundError(f"dataset not prepared: {paths.pseudobulk} (run `vcell prepare`)")
        bulk = np.load(paths.pseudobulk)
        emb = np.load(paths.coexpression)
        prior = np.load(paths.knowledge) if paths.knowledge.exists() else None
        cross = dict(np.load(paths.cross)) if paths.cross.exists() else None
        return cls(
            paths=paths,
            meta=json.loads(paths.meta.read_text()),
            genes=bulk["genes"],
            pseudobulk=dict(zip(bulk["labels"].tolist(), bulk["means"], strict=True)),
            coexpression_genes=emb["genes"],
            coexpression=emb["embedding"],
            knowledge_genes=prior["genes"] if prior is not None else np.array([], dtype=str),
            knowledge={k: prior[k] for k in prior.files if k != "genes"} if prior is not None else {},
            cross=cross,
        )

    @property
    def name(self) -> str:
        return self.paths.dataset

    @property
    def control_label(self) -> str:
        return self.meta["control_label"]

    @cached_property
    def adata(self) -> ad.AnnData:
        adata = ad.read_h5ad(self.paths.cells)
        if not sp.issparse(adata.X):
            adata.X = sp.csr_matrix(adata.X)
        return adata

    @cached_property
    def counts(self) -> ad.AnnData:
        """Raw counts of the scored cells (see `save_counts`)."""
        if not self.paths.counts.exists():
            raise FileNotFoundError(f"no raw counts: {self.paths.counts} (run `vcell prepare`)")
        return ad.read_h5ad(self.paths.counts)

    @cached_property
    def perturbations(self) -> pd.DataFrame:
        return pd.DataFrame(self.meta["perturbations"]).set_index("target", drop=False).rename_axis(None)

    def targets(self, split: str) -> list[str]:
        return self.perturbations.index[self.perturbations["split"] == split].tolist()

    def cells_of(self, target: str) -> ad.AnnData:
        return self.adata[self.adata.obs[TARGET_COL] == target]

    @property
    def control_mean(self) -> np.ndarray:
        return self.pseudobulk[self.control_label]

    def delta(self, target: str) -> np.ndarray:
        """Measured mean log-expression change versus control for one target."""
        return self.pseudobulk[target] - self.control_mean

    def deltas(self, targets: list[str]) -> np.ndarray:
        return np.stack([self.delta(t) for t in targets]) if targets else np.zeros((0, len(self.genes)))

    def gene_coexpression(self, gene: str) -> np.ndarray | None:
        idx = self._coexp_index.get(gene)
        return None if idx is None else self.coexpression[idx]

    def gene_knowledge(self, gene: str, kind: str) -> np.ndarray | None:
        """A gene's prior-knowledge embedding, or None when that kind is missing for it."""
        matrix = self.knowledge.get(kind)
        idx = self._knowledge_index.get(gene)
        if matrix is None or idx is None or not np.any(matrix[idx]):
            return None
        return matrix[idx]

    def cross_row(self, target: str) -> int | None:
        """Row of `target` in the cross-line arrays, or None when no other line knocked it down."""
        return None if self.cross is None else self._cross_index.get(target)

    @cached_property
    def _cross_index(self) -> dict[str, int]:
        return {t: i for i, t in enumerate(self.cross["targets"].tolist())} if self.cross is not None else {}

    @cached_property
    def _knowledge_index(self) -> dict[str, int]:
        return {g: i for i, g in enumerate(self.knowledge_genes.tolist())}

    @cached_property
    def _coexp_index(self) -> dict[str, int]:
        return {g: i for i, g in enumerate(self.coexpression_genes)}
