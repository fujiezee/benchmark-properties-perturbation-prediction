#!/usr/bin/env python3
"""Can your benchmark's test set tell your models apart?

A leaderboard ranks models by a score. That ranking is only meaningful if the
gaps between models are larger than the score's own run-to-run noise. This
measures both from the same evaluations and says which it is.

The number it reports is the separation-to-noise ratio:

    spread across models / mean within-model sd across replicates

A ratio near 1 means the gap between your best and worst model is the size of
the noise in a single model's score, so the ordering is mostly arbitrary --
re-running with different seeds would reshuffle it. We found ratios of 4 to 20
in cross-lab transfer, and a ratio of 1.5 on one small test set that had looked
like a real result about models becoming indistinguishable; it was a test set too
small to resolve anything.

A replicate is whatever your evaluation re-randomises: a training seed, a
bootstrap resample of the test set, a draw of few-shot examples. You need at
least two, and the replicates should be the same ones for every model, which
lets the comparison be paired and much more sensitive.

Input: a CSV, TSV or JSON table with columns `model`, `replicate`, `score`.
JSON may be a list of objects or {"rows": [...]}. Extra columns are ignored
unless named with --group, which reports each group separately.

    resolvability.py scores.csv
    resolvability.py results.json --score specific --group source,target

Importable: `report(rows)` takes an iterable of (model, replicate, score).
"""

import argparse
import csv
import json
import math
import pathlib
import sys
from collections import defaultdict

try:                                    # only used for an exact critical value
    from scipy.stats import t as _t
except Exception:                       # the normal approximation is fine by ~30 df
    _t = None


def _tcrit(df: int) -> float:
    if _t is not None:
        return float(_t.ppf(0.975, df))
    return 1.96 if df >= 30 else {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57,
                                  6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23,
                                  12: 2.18, 15: 2.13, 20: 2.09, 25: 2.06}.get(df, 2.04)


def _sd(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def report(rows) -> dict:
    """rows: iterable of (model, replicate, score). Returns the report as a dict."""
    by_model: dict[str, dict[str, float]] = defaultdict(dict)
    for model, rep, score in rows:
        by_model[str(model)][str(rep)] = float(score)
    if len(by_model) < 2:
        raise ValueError(f"need at least 2 models, got {len(by_model)}")

    means = {m: sum(v.values()) / len(v) for m, v in by_model.items()}
    sds = {m: _sd(list(v.values())) for m, v in by_model.items()}
    usable = [s for s in sds.values() if s == s]
    if not usable:
        raise ValueError("every model has a single replicate; noise cannot be estimated")

    spread = max(means.values()) - min(means.values())
    noise = sum(usable) / len(usable)
    ratio = spread / noise if noise > 0 else float("inf")

    # Paired comparison where the replicates line up, which they should.
    shared = set.intersection(*(set(v) for v in by_model.values()))
    pairs, sep = [], 0
    mdd = float("nan")
    if len(shared) >= 2:
        reps = sorted(shared)
        mdds = []
        for a in by_model:
            for b in by_model:
                if a >= b:
                    continue
                d = [by_model[a][r] - by_model[b][r] for r in reps]
                sd_d = _sd(d)
                half = _tcrit(len(reps) - 1) * sd_d / math.sqrt(len(reps))
                mean_d = sum(d) / len(d)
                ok = abs(mean_d) > half
                sep += ok
                mdds.append(half)
                pairs.append({"a": a, "b": b, "diff": mean_d,
                              "ci": [mean_d - half, mean_d + half], "separable": ok})
        mdd = sum(mdds) / len(mdds)

    return {"models": len(by_model), "replicates": len(shared) or None,
            "paired": len(shared) >= 2, "means": means, "sds": sds,
            "spread": spread, "noise": noise, "ratio": ratio,
            "min_detectable_difference": mdd,
            "separable_pairs": sep, "total_pairs": len(pairs), "pairs": pairs}


def verdict(ratio: float) -> str:
    if ratio < 2:
        return "cannot resolve these models: the spread is within noise"
    if ratio < 4:
        return "marginal: the ordering is resolvable but the middle ranks are not"
    return "resolvable"


def _print(rep: dict, name: str = "") -> None:
    if name:
        print(f"\n== {name} ==")
    print(f"  {rep['models']} models, {rep['replicates']} replicates"
          f"{' (paired)' if rep['paired'] else ''}")
    print(f"  spread across models   {rep['spread']:.4f}")
    print(f"  mean within-model sd   {rep['noise']:.4f}")
    print(f"  separation-to-noise    {rep['ratio']:.2f}   {verdict(rep['ratio'])}")
    if rep["paired"]:
        print(f"  min detectable diff    {rep['min_detectable_difference']:.4f} (95%, paired)")
        print(f"  separable model pairs  {rep['separable_pairs']} of {rep['total_pairs']}")
    order = sorted(rep["means"], key=lambda m: -rep["means"][m])
    print("  ranking: " + ", ".join(f"{m} {rep['means'][m]:.4f}" for m in order))


def _load(path: pathlib.Path, score_col: str) -> list[dict]:
    text = path.read_text()
    if path.suffix.lower() == ".json":
        d = json.loads(text)
        return d.get("rows", d) if isinstance(d, dict) else d
    delim = "\t" if path.suffix.lower() in (".tsv", ".tab") else ","
    return list(csv.DictReader(text.splitlines(), delimiter=delim))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("table", type=pathlib.Path)
    ap.add_argument("--score", default="score", help="score column (default: score)")
    ap.add_argument("--model", default="model")
    ap.add_argument("--replicate", default="replicate")
    ap.add_argument("--group", default="", help="comma-separated columns to report separately")
    ap.add_argument("--models", default="", help="comma-separated models to keep (default: all)")
    ap.add_argument("--json", action="store_true", help="emit the report as JSON")
    a = ap.parse_args()

    recs = _load(a.table, a.score)
    for col in (a.model, a.replicate, a.score):
        if recs and col not in recs[0]:
            print(f"column {col!r} not in table; columns are {sorted(recs[0])}", file=sys.stderr)
            return 2

    keep = {m for m in a.models.split(",") if m}
    if keep:
        recs = [r for r in recs if str(r[a.model]) in keep]
    keys = [k for k in a.group.split(",") if k]
    groups: dict[tuple, list] = defaultdict(list)
    for r in recs:
        groups[tuple(str(r[k]) for k in keys)].append(
            (r[a.model], r[a.replicate], r[a.score]))

    out = {}
    for g, rows in sorted(groups.items()):
        name = " / ".join(g)
        try:
            rep = report(rows)
        except ValueError as e:
            print(f"\n== {name} ==\n  skipped: {e}")
            continue
        out[name or "all"] = rep
        if not a.json:
            _print(rep, name)
    if a.json:
        print(json.dumps(out, indent=1))
    elif len(out) > 1:
        rs = sorted(r["ratio"] for r in out.values())
        mid = rs[len(rs) // 2] if len(rs) % 2 else (rs[len(rs) // 2 - 1] + rs[len(rs) // 2]) / 2
        print(f"\n  {len(rs)} groups: median ratio {mid:.2f}, range {rs[0]:.2f} to {rs[-1]:.2f}")
        print(f"  groups that cannot resolve their models: {sum(r < 2 for r in rs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
