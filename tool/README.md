# resolvability.py

Whether a benchmark's test set can tell its models apart.

A leaderboard ranks models by a score. The ranking means something only if the
gaps between models exceed the score's own run-to-run noise. This reads both off
the same evaluations.

```
separation-to-noise = spread across models / mean within-model sd across replicates
```

A ratio near 1 means the best-to-worst gap is the size of one model's own noise,
so re-running with different seeds would reshuffle the order.

## Use

Give it a table with `model`, `replicate`, `score`. A replicate is whatever the
evaluation re-randomises: a training seed, a bootstrap resample of the test set,
a draw of few-shot examples. At least two, and the same ones for every model so
the comparison can be paired.

```bash
resolvability.py scores.csv
resolvability.py results.json --score specific --replicate repeat \
  --group source,target --models copy,knn,linear
```

CSV, TSV, or JSON (a list of objects, or `{"rows": [...]}`). `--group` reports
each group separately and prints the median and range across groups.

It reports the ratio, a verdict, the 95% minimum detectable difference from the
paired replicate differences, and how many model pairs are separable.

## What it reproduces

On this project's 21 cross-lab pairs
(`results/models_crosslab.json` + `results/models_orion.json`):

```
21 groups: median ratio 8.40, range 3.93 to 19.57
groups that cannot resolve their models: 0
```

## The ddof trap

`np.std` defaults to `ddof=0`, the population sd, which understates the noise of
a score estimated from R replicates by sqrt(R/(R-1)) and inflates the ratio to
match -- 5.4% at R = 10. This tool uses `ddof=1`. `models.py:174` in this repo
stores its summary `sd` with the default, which is why ratios recomputed here are
below the ones that script's summaries imply. The per-replicate rows are the
source of truth; recompute from those rather than from a stored sd.

No dependencies. scipy is used for an exact t critical value if importable, and
a small table of critical values is built in otherwise.
