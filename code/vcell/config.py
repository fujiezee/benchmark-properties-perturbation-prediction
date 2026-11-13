"""Filesystem layout for downloaded data and generated artifacts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DATASETS = ("k562", "rpe1", "hepg2", "jurkat", "h1", "demo")


def vcell_home() -> Path:
    return Path(os.environ.get("VCELL_HOME", Path.home() / ".cache" / "vcell"))


def xatlas_path(home: Path) -> Path:
    """X-Atlas/Orion responses to every prepared dataset's knockdowns, shared by all (built on the cloud box)."""
    return home / "external" / "xatlas.npz"


@dataclass(frozen=True)
class Paths:
    """Artifact paths for one dataset (`k562` or `demo`)."""

    dataset: str
    home: Path

    @classmethod
    def for_dataset(cls, dataset: str, home: Path | None = None) -> Paths:
        if dataset not in DATASETS:
            raise ValueError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")
        return cls(dataset=dataset, home=home or vcell_home())

    @property
    def raw_dir(self) -> Path:
        return self.home / "raw"

    @property
    def structures_dir(self) -> Path:
        return self.home / "structures"

    @property
    def root(self) -> Path:
        return self.home / "datasets" / self.dataset

    @property
    def cells(self) -> Path:
        return self.root / "cells.h5ad"

    @property
    def meta(self) -> Path:
        return self.root / "meta.json"

    @property
    def counts(self) -> Path:
        """Raw counts of the held-out and control cells, for scoring."""
        return self.root / "counts.h5ad"

    @property
    def pseudobulk(self) -> Path:
        return self.root / "pseudobulk.npz"

    @property
    def coexpression(self) -> Path:
        return self.root / "coexpression.npz"

    @property
    def knowledge(self) -> Path:
        return self.root / "knowledge.npz"

    @property
    def cross(self) -> Path:
        return self.root / "cross.npz"

    @property
    def xatlas(self) -> Path:
        return xatlas_path(self.home)

    @property
    def gene_info(self) -> Path:
        return self.root / "gene_info.json"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def eval_dir(self) -> Path:
        return self.root / "eval"

    @property
    def leaderboard(self) -> Path:
        return self.root / "leaderboard.json"
