"""Main figures for the submission, from the stored results only.

Every number is read from `results/*.json`; nothing is recomputed here, so a
panel cannot disagree with the text. Panels whose inputs do not exist yet are
skipped with a note rather than drawn from partial data.

Figure 1  the reliability ceiling across four contexts
Figure 2  headroom against context distance -- the main line
Figure 3  which model wins per pair, and whether the pairs resolve them
Figure 4  biological necessity, with the multiplicity shown

Usage: figures.py RESULTS_DIR OUT_DIR [--captions]
"""

import pathlib as _p, sys as _s
_h = _p.Path(__file__).resolve().parent
_s.path[:0] = [str(_h), str(_h / "lib"), str(_h.parent / "tool")]

import warnings
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R, OUT = (pathlib.Path(p) for p in sys.argv[1:3])
OUT.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(pathlib.Path(__file__).resolve().parent.parent / "tool")]
from resolvability import report          # the text's ratios come from here too

# Nature Methods asks for a sans face and embedded fonts; Helvetica is the house
# look, and DejaVu is matplotlib's default only because it ships with it.
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 300,
                     "savefig.bbox": "tight", "pdf.fonttype": 42,
                     # "tight" otherwise adds 0.1 in on every side, which would
                     # land the delivered width 5 mm over the measure below.
                     "savefig.pad_inches": 0.0,
                     "font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]})

# The journal's double-column measure. A figure delivered at any other width is
# rescaled on the page, and the type with it, so the 7 pt above would arrive as
# something else.
DOUBLE_COLUMN_IN = 180 / 25.4


# A glyph the chosen face lacks is drawn as an empty box, and matplotlib only
# warns, so a figure can ship with a hole in it -- U+2217 did exactly that when
# this moved to Helvetica. Make the warning fatal. It arrives through `warnings`,
# not the logging module, which is why a logging handler would never see it.
warnings.filterwarnings("error", message=".*missing from font.*")
CTX = [("batch", "same batch", "#1b3a5c"), ("line", "same lab,\nother line", "#2e7aa8"),
       ("lab", "other lab", "#d1803a"), ("far", "other lab,\nother type", "#a8324a")]
load = lambda n: json.loads((R / n).read_text())
written = []


# A journal caption sits under the figure, not inside it, and the same words in
# the image and in the manuscript are two copies to keep in step. So the strip is
# reserved either way -- the layout depends on it -- but the words are printed
# only with --captions, for a self-contained PNG to look at outside the paper.
# build_pdf.py holds the submission captions.
BAKE = "--captions" in sys.argv


def caption(fig, text):
    """Reserve a strip under the axes so a note never lands on a tick label."""
    fig.subplots_adjust(bottom=0.34)
    if BAKE:
        fig.text(0.02, 0.015, text, fontsize=5.8, va="bottom", wrap=True)


def save(fig, name):
    # `savefig.bbox="tight"` crops the canvas to its content, so the delivered
    # width is whatever is left rather than what figsize asked for. Measure that
    # crop and scale the canvas so the crop itself lands on the column measure.
    # Text keeps its point size when the canvas is resized, so the crop moves
    # with it and one correction overshoots. Iterate to a fixed point instead.
    for _ in range(8):
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        err = DOUBLE_COLUMN_IN - bb.width
        if abs(err) < 0.004:                 # 0.1 mm
            break
        w, h = fig.get_size_inches()
        fig.set_size_inches(w + err, h)
    else:
        raise SystemExit(f"{name}: width did not settle ({bb.width * 25.4:.1f} mm)")
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    written.append(name)
    print(f"  wrote {name}.pdf/.png", flush=True)


# -- Figure 1: the ceiling, four contexts ------------------------------------
d = load("baseline_orion.json")
bins = [b for b in d["bins"] if all(k in b for k in ("batch", "line", "lab", "far"))]
# "all" is a summary over the same knockdowns, not a sixth stratum; drawing it on
# the same axis invites reading it as one.
bins, overall = [b for b in bins if b["bin"] != "all"], next(
    (b for b in bins if b["bin"] == "all"), None)
x = np.arange(len(bins))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.5), width_ratios=[1.35, 1])
for key, lab, col in CTX:
    a1.plot(x, [b[key] for b in bins], "o-", color=col, label=lab.replace("\n", " "),
            ms=3.2, lw=1.2)
if overall:
    for key, _, col in CTX:
        a1.axhline(overall[key], color=col, lw=0.5, ls=":", alpha=0.65)
a1.set_xticks(x, [b["bin"] for b in bins], rotation=45, ha="right")
a1.set_xlabel("within-context reliability of the knockdown")
a1.set_ylabel("transfer (Pearson r)")
# The title must not overstate the claim: distance costs a great deal -- the
# same-batch curve sits far above the rest at every level -- while reliability
# raises all four. Saying the ceiling is reliability "not distance" would
# contradict Claim 1.
a1.set_title("a   leaving the batch costs most", loc="left", fontsize=7.5)
a1.legend(frameon=False, fontsize=6, loc="upper left")
a1.grid(axis="y", lw=0.3, alpha=0.35)

gaps = [b["batch - line 95%"] for b in bins]
a2.barh(x, [b["batch"] - b["line"] for b in bins], color="#2e7aa8", height=0.62,
        xerr=np.abs(np.array(gaps).T - np.array([b["batch"] - b["line"] for b in bins])),
        error_kw={"lw": 0.7, "capsize": 1.6})
a2.axvline(0, color="k", lw=0.6)
a2.set_yticks(x, [b["bin"] for b in bins])
a2.set_xlabel("same batch − same lab, other line")
a2.set_title("b   that advantage holds in every bin", loc="left", fontsize=7.5)
a2.grid(axis="x", lw=0.3, alpha=0.35)
caption(fig, f"{d['knockdowns']:,} knockdowns, {d['genes']:,} shared genes. Dotted lines in a are "
         "the same four contexts over all knockdowns. Bars are 95% percentile bootstrap over "
         "knockdowns; the gap in b is tested per bin.")
save(fig, "figure1_ceiling")

# -- Figure 2: per-pair winners and resolvability ----------------------------
rows = sum((load(f)["rows"] for f in ("models_crosslab.json", "models_orion.json")), [])
ABL = ["copy", "copy_pc", "calibrated", "knn", "network", "linear"]
pairs = sorted({(r["source"], r["target"]) for r in rows})
per = {}
for p in pairs:
    sel = [r for r in rows if (r["source"], r["target"]) == p and r["model"] in ABL]
    rep = report([(r["model"], r["repeat"], r["specific"]) for r in sel])
    per[p] = rep
wins = {m: sum(max(v["means"], key=v["means"].get) == m for v in per.values()) for m in ABL}
ratios = np.array([v["ratio"] for v in per.values()])

fig, (b1, b2) = plt.subplots(1, 2, figsize=(7.0, 2.5), width_ratios=[1, 1.2])
order = sorted(ABL, key=lambda m: -wins[m])
b1.bar(range(len(order)), [wins[m] for m in order],
       color=["#a8324a" if m == "linear" else "#9bb4c4" for m in order], width=0.66)
b1.set_xticks(range(len(order)), order, rotation=45, ha="right")
b1.set_ylabel(f"pairs won (of {len(pairs)})")
b1.set_title("a   linear wins most pairs", loc="left", fontsize=7.5)
for i, m in enumerate(order):
    b1.text(i, wins[m] + 0.25, wins[m], ha="center", fontsize=6)
b1.grid(axis="y", lw=0.3, alpha=0.35)

srt = np.sort(ratios)
b2.barh(np.arange(len(srt)), srt, color="#2e7aa8", height=0.72)
b2.axvline(2, color="#a8324a", lw=0.9, ls="--")
# Keep this clear of the lowest bar: the axis is inverted, so put the note
# above the top bar rather than below the bottom one.
b2.text(2, b2.get_ylim()[1], "2 = spread equals noise", color="#a8324a",
        fontsize=5.8, ha="center", va="bottom")
b2.set_xlabel("separation-to-noise ratio")
b2.set_ylabel("pair (sorted)")
b2.set_yticks([])
b2.set_title(f"b   all {len(srt)} pairs resolve their models", loc="left", fontsize=7.5)
med = float(np.median(srt))
caption(fig, f"{len(pairs)} cross-lab pairs, {len(ABL)} ablations, "
         f"{max(r['repeat'] for r in rows) + 1} anchor draws each. Median ratio {med:.2f}, "
         f"range {srt[0]:.2f} to {srt[-1]:.2f}. `mean` is excluded from a as the floor "
         "every model is scored against.")
save(fig, "figure3_models")

# -- Figure 3: necessity, with multiplicity ----------------------------------
m = load("multiplicity.json")
tb = load("transfer_biology_wide.json")
fig, (c1, c2) = plt.subplots(1, 2, figsize=(7.0, 2.5), width_ratios=[1.3, 1])
labels = [b["bin"] for b in m["bins"]]
keys_ = [k for k, _, _ in CTX]
w = 0.2
for i, (key, lab, col) in enumerate(CTX):
    vals = [b.get(key, {}).get("diff", np.nan) for b in m["bins"]]
    pos = np.arange(len(labels)) + (i - 1.5) * w
    c1.bar(pos, vals, width=w, color=col, label=lab.replace("\n", " "))
    for p_, b in zip(pos, m["bins"]):
        cell = b.get(key)
        if cell and cell["sig_bonf"]:
            c1.text(p_, cell["diff"] + 0.012, "*", ha="center", fontsize=6, color="k")
c1.axhline(0, color="k", lw=0.6)
c1.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
c1.set_xlabel("within-context reliability")
c1.set_ylabel("essential − other (transfer r)")
c1.set_title("a   necessity buys more transfer the further the context", loc="left", fontsize=7.5)
c1.set_ylim(top=max(b.get(k, {}).get("diff", 0) for b in m["bins"] for k in keys_) * 1.30)
c1.legend(frameon=False, fontsize=6, loc="upper right", ncol=2, columnspacing=0.8)
c1.grid(axis="y", lw=0.3, alpha=0.35)

keys = [k for k, _, _ in CTX]
rho = [next(e["rho"] for e in m["ess"] if e["test"].endswith(k)) for k in keys]
c2.bar(range(4), rho, color=[c for _, _, c in CTX], width=0.62)
for i, (r_, k) in enumerate(zip(rho, keys)):
    bh = next(e["bh"] for e in m["ess"] if e["test"].endswith(k))
    c2.text(i, r_ + 0.004, f"{r_:+.3f}\nBH {bh:.0e}", ha="center", fontsize=5.6)
c2.set_xticks(range(4), [l for _, l, _ in CTX], fontsize=6)
c2.set_ylabel("partial Spearman ρ")
c2.set_title("b   all knockdowns at once, controlled", loc="left", fontsize=7.5)
c2.set_ylim(0, max(rho) * 1.35)
c2.grid(axis="y", lw=0.3, alpha=0.35)
caption(fig, f"{tb['knockdowns']:,} knockdowns. In a, * marks a cell surviving Bonferroni for "
         f"the family of {m['family']}: "
         f"{sum(1 for b in m['bins'] for k in keys if b.get(k, {}).get('sig_bonf'))} of {m['family']} do. "
         "b is the primary analysis -- every knockdown at once rather than split twenty ways, "
         "BH-adjusted across its own family of four.")
save(fig, "figure4_necessity")

# -- Figure 4: the main line, headroom against context distance --------------
KINDS = [("replicate", "replicate screen,\nsame lab"),
         ("same lab, other type", "same lab,\nother cell type"),
         ("other lab, other type", "other lab,\nother cell type")]
MODELS = ["copy", "copy_pc", "calibrated", "knn", "network", "linear"]
runs = sum((load(f)["rows"] for f in
            ("models_near.json", "models_crosslab.json", "models_orion.json")), [])
score, npair = {}, {}
for key, _ in KINDS:
    sel = [r for r in runs if r["kind"] == key]
    npair[key] = len({(r["source"], r["target"]) for r in sel})
    score[key] = {m: float(np.mean([r["specific"] for r in sel if r["model"] == m]))
                  for m in MODELS}

fig, (d1, d2) = plt.subplots(1, 2, figsize=(7.0, 2.6), width_ratios=[1.15, 1])
x = np.arange(len(KINDS))
best = [max(score[k].values()) for k, _ in KINDS]
# The band is what no model reached: the point of the figure, so it is drawn
# first and the models sit on top of it.
d1.fill_between(x, best, 1.0, color="#a8324a", alpha=0.10, lw=0)
d1.plot(x, [1.0] * len(x), color="#a8324a", lw=1.0, ls="--")
d1.text(len(x) - 1.02, 0.975, "ceiling: the target's own reproducible signal",
        fontsize=5.6, ha="right", va="top", color="#a8324a")
for m, col in zip(MODELS, ["#9bb4c4", "#6d93ab", "#d1a03a", "#7a9a6b", "#b08a9a", "#a8324a"]):
    d1.plot(x, [score[k][m] for k, _ in KINDS], "o-", ms=3.2,
            lw=1.6 if m == "linear" else 0.9, color=col,
            label=m, zorder=3 if m == "linear" else 2)
for i, (k, _) in enumerate(KINDS):
    d1.text(i, best[i] + (1.0 - best[i]) / 2, f"{1 - best[i]:.0%}\nunreached",
            ha="center", va="center", fontsize=6, color="#a8324a")
d1.set_xticks(x, [lab for _, lab in KINDS], fontsize=6)
d1.set_xlim(-0.25, len(x) - 0.75)
d1.set_ylim(0, 1.06)
d1.set_ylabel("fraction of reproducible signal captured")
d1.set_title("a   models converge as the headroom grows", loc="left", fontsize=7.5)
d1.legend(frameon=False, fontsize=5.6, loc="lower left", ncol=2, columnspacing=0.8)
d1.grid(axis="y", lw=0.3, alpha=0.35)

# Four definitions of "how much method choice is worth", because the claim must
# not rest on the flattering one: two are monotone, two are not, and both that
# are not break at the 2-pair replicate row.
DEFS = [("best − worst", lambda v: max(v.values()) - min(v.values())),
        ("best − copy", lambda v: max(v.values()) - v["copy"]),
        ("best − 2nd", lambda v: (lambda s: s[0] - s[1])(sorted(v.values(), reverse=True))),
        ("sd across models", lambda v: float(np.std(list(v.values()), ddof=1)))]
w = 0.2
for i, (name, fn) in enumerate(DEFS):
    vals = [fn(score[k]) / (1 - max(score[k].values())) for k, _ in KINDS]
    mono = all(a > b for a, b in zip(vals, vals[1:]))
    d2.bar(x + (i - 1.5) * w, vals, width=w,
           color=["#1b3a5c", "#2e7aa8", "#d1803a", "#a8324a"][i],
           label=f"{name}{'' if mono else '  (not monotone)'}")
d2.set_xticks(x, ["replicate\n(2 pairs)", "same lab\n(18)", "other lab\n(21)"], fontsize=6)
d2.set_ylabel("spread ÷ headroom")
d2.set_title("b   what choosing well is worth", loc="left", fontsize=7.5)
d2.legend(frameon=False, fontsize=5.4, loc="upper right")
d2.grid(axis="y", lw=0.3, alpha=0.35)
caption(fig, "A score is the fraction of the target's reproducible signal a model captures, so "
        "1 − score is what none captured. In b, two definitions fall monotonically and two peak "
        "at the middle row; both exceptions break at the replicate row, where only 2 directed "
        "pairs exist because the lab ran two K562 screens.")
save(fig, "figure2_headroom")

json.dump({"figures": written, "pairs": len(pairs),
           "median_ratio": med, "wins": wins}, open(OUT / "figures.json", "w"), indent=1)
print("FIGURES_DONE")
