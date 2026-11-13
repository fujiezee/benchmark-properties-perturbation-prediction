# Benchmark properties, not method differences, dominate cross-context perturbation prediction

Analysis code and derived data for the paper of that title (Morgan Wang,
ORCID 0009-0002-2725-2948).

## What is here

```
code/            analysis that produces every number and figure in the paper
  loaders/       build the .npz response matrices from the public deposits
  lib/           ctxlib.py (context loading, split halves), models.py (the seven models)
  vcell/         the two modules the scripts import (config, data.dataset)
results/         every results/*.json the figures and tables read
tool/            resolvability.py, the standalone diagnostic, numpy only
data/            derived response matrices and their independent halves
```

## The primary data are not here

All eight screens are previously published and stay with their own deposits:

| context | deposit |
| --- | --- |
| Replogle K562 essential, K562 genome-wide, RPE1 | figshare+ doi:10.25452/figshare.plus.20029387 |
| Nadig HepG2, Jurkat | GEO GSE264667 |
| Satija VIPerturb K562 | Zenodo doi:10.5281/zenodo.18460279 |
| Gasperini (Shendure) K562, via scPerturb | Zenodo doi:10.5281/zenodo.7041849 |
| Xaira X-Atlas/Orion HEK293T | Hugging Face `Xaira-Therapeutics/X-Atlas-Orion` |

DepMap 24Q4 (`CRISPRGeneEffect.csv`, `CRISPRInferredCommonEssentials.csv`) and
STRING (combined score >= 700) enter the essentiality split and the `network`
ablation respectively.

`code/loaders/` turns each of those into the `.npz` matrices under `data/`, so
the chain from public deposit to published figure is complete. Per-file
fingerprints are in the paper's reporting pack.

## Running it

Three environments, because GEARS and CPA each need versions the others cannot
satisfy.

- Main analysis: Python 3.12.3, numpy 2.5.4, scipy 1.18.1, anndata 0.13.4,
  h5py 3.16.0, polars 2.1.0, pandas 3.1.1, matplotlib 3.11.2, pyarrow 26.0.1
- GEARS: Python 3.12.3, cell-gears 0.1.2, torch 2.5.1+cpu, anndata 0.10.9,
  numpy 1.26.4
- CPA: Python 3.11, cpa-tools, scvi-tools 0.20.3, torch 2.15.0+cpu,
  pytorch-lightning 1.9.5, anndata 0.9.2, mudata 0.2.3, numpy 1.26.4,
  pandas 2.1.4

Put `code/` on `PYTHONPATH`. The scripts take input and output paths as
arguments; each one's docstring gives its usage.

`tool/resolvability.py` needs nothing but numpy and takes any table of
(model, replicate, score).

## Reproducibility

Anchor draws are seeded from a Python string hash and are therefore not
bit-reproducible across processes unless `PYTHONHASHSEED` is fixed. Every
reported quantity averages ten draws, and the separation-to-noise ratios are
computed across them.

## Licence

Code: MIT. Derived data: CC BY 4.0. The primary screens remain under the terms
of their own deposits.
