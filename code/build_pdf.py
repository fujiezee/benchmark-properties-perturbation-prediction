"""Assemble the submission PDF: one file, text and figures together.

Nature Methods accepts PDF, Word or compiled LaTeX at initial submission and
encourages text and figures in a single file for review. This builds that file
from the Markdown drafts, so the PDF is never edited by hand and cannot drift
from them.

Figures are placed where their callouts fall rather than collected at the end,
which is what "text and figures together" asks for. A `Figure N` callout in the
text binds to `figures/figureN_*.png`; the PNG is used rather than the PDF
because the renderer embeds raster images reliably, and the vector PDFs remain
the files to submit if the editor asks for them separately.

Usage: build_pdf.py SPEC_DIR OUT.pdf
"""

import base64
import pathlib
import re
import shutil
import subprocess
import sys

import markdown

spec, out_p = pathlib.Path(sys.argv[1]), sys.argv[2]
FIGS = spec / "figures"

# Order follows a Nature manuscript: results first, methods after, with the
# argument and reporting material as the back matter a reviewer checks.
PARTS = [("", "title-page.md"), ("Abstract", "abstract.md"), ("Introduction", "introduction.md"),
         ("Results", "results-draft.md"), ("Discussion", "discussion.md"),
         ("Methods", "methods.md"), ("References", "references.md"),
         ("Reporting", "reporting.md")]

CSS = """
@page { size: A4; margin: 20mm 18mm 18mm 18mm; }
body { font: 10pt/1.55 Georgia, 'Times New Roman', serif; color: #111; }
h1 { font-size: 17pt; margin: 0 0 2mm; page-break-before: always; }
h1.first { page-break-before: avoid; }
h2 { font-size: 12pt; margin: 7mm 0 2mm; page-break-after: avoid; }
h3 { font-size: 10.5pt; margin: 5mm 0 1.5mm; page-break-after: avoid; }
p { margin: 0 0 2.6mm; text-align: justify; }
code { font: 9pt 'SF Mono', Menlo, monospace; background: #f4f4f2; padding: 0 2px; }
pre { font: 8.5pt 'SF Mono', Menlo, monospace; background: #f7f7f5;
      padding: 3mm; border-left: 2px solid #ccc; white-space: pre-wrap; }
table { border-collapse: collapse; width: 100%; margin: 3mm 0 4mm;
        font-size: 8.5pt; page-break-inside: avoid; }
th, td { border: 0.4pt solid #bbb; padding: 1.1mm 2mm; text-align: left; vertical-align: top; }
th { background: #f0efec; font-weight: bold; }
blockquote { margin: 3mm 0; padding: 2.5mm 4mm; background: #fbf7ec;
             border-left: 2.5pt solid #d1a03a; }
blockquote p { margin: 0; }
figure { margin: 5mm 0; page-break-inside: avoid; text-align: center; }
figure img { width: 100%; }
figcaption { font-size: 8pt; color: #444; margin-top: 1.5mm; text-align: left; }
ul, ol { margin: 0 0 3mm; padding-left: 6mm; }
li { margin-bottom: 1mm; }
a { color: #1b3a5c; text-decoration: none; }
"""


# Captions are written here rather than derived from the filename, which produced
# "Figure 1. ceiling" and told the reader nothing. Each says what the panels show
# and the n behind them, as a journal caption has to stand on its own.
CAPTIONS = {
    1: "<b>Transfer falls with context.</b> (a) Transfer against the knockdown's own "
       "within-context reliability, for four target contexts; dotted lines are the same "
       "four contexts over all knockdowns. (b) Same-batch minus same-lab-other-line per "
       "reliability bin, 95% percentile bootstrap over knockdowns. n = 1,053 knockdowns, "
       "6,084 shared genes.",
    2: "<b>The further the context, the more is left unreached and the less the choice of "
       "model is worth.</b> A score is the fraction of the target's reproducible signal a "
       "model captures, so 1 &minus; score is what none captured. (a) Each model across "
       "three context distances, with the unreached band shaded. (b) The spread across "
       "models as a fraction of that headroom, under four definitions: two fall "
       "monotonically and two peak at the middle row, and both exceptions break at the "
       "replicate row, where only 2 directed pairs exist. 2, 18 and 21 pairs.",
    3: "<b>A linear map wins, and the ranking is resolvable.</b> (a) Pairs won by each "
       "model over 21 cross-lab pairs, on knockdown-specific transfer; <code>mean</code> "
       "is excluded as the floor every model is scored against. (b) Separation-to-noise "
       "per pair: the spread across models over the within-model standard deviation "
       "across 10 anchor draws. At a ratio of 2 the two are equal.",
    4: "<b>What transfers is the core-essential program.</b> (a) Essential minus other "
       "transfer per reliability bin and context; &#8727; marks a cell surviving "
       "Bonferroni for the family of 20, which 7 of 20 do. (b) Partial Spearman between "
       "essentiality and transfer over all knockdowns at once, controlling reliability "
       "and response magnitude, Benjamini-Hochberg adjusted across its family of four. "
       "n = 9,586 knockdowns.",
}


def figure_html(n: int) -> str:
    hits = sorted(FIGS.glob(f"figure{n}_*.png"))
    if not hits:
        print(f"  WARNING no figure {n} in {FIGS}", flush=True)
        return ""
    b64 = base64.b64encode(hits[0].read_bytes()).decode()
    cap = CAPTIONS.get(n, "")
    return (f'<figure><img src="data:image/png;base64,{b64}"/>'
            f'<figcaption><b>Figure {n}.</b> {cap}</figcaption></figure>')


placed: set[int] = set()
chunks = []
for i, (title, name) in enumerate(PARTS):
    p = spec / name
    if not p.exists():
        print(f"  MISSING {name}, skipped", flush=True)
        continue
    text = p.read_text()
    # Drop the file's own H1 so the part title is the only one -- except for a
    # part with no title of its own, such as the title page, whose H1 *is* the
    # paper's title and must survive.
    if title:
        text = re.sub(r"\A#\s+.*\n", "", text)
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    # The drafts write an em dash as "--", which Markdown leaves as two hyphens.
    body = re.sub(r"(?<=[^-<])--(?=[^->])", "\u2014", body)

    # Place each figure after the paragraph that first cites it.
    def place(m: re.Match) -> str:
        para = m.group(0)
        add = ""
        for n in sorted({int(x) for x in re.findall(r"Figure (\d)", para)} - placed):
            add += figure_html(n)
            placed.add(n)
        return para + add

    body = re.sub(r"<p>.*?</p>", place, body, flags=re.S)
    cls = ' class="first"' if i == 0 else ""
    chunks.append(f"<h1{cls}>{title}</h1>\n{body}" if title
                  else f'<div{cls and " class=\"first\""}>{body}</div>')

missing = [n for n in (1, 2, 3, 4) if n not in placed]
if missing:
    print(f"  figures {missing} had no callout; appended at the end", flush=True)
    chunks.append("<h1>Figures</h1>" + "".join(figure_html(n) for n in missing))

html = (f"<html><head><meta charset='utf-8'><style>{CSS}</style></head>"
        f"<body>{''.join(chunks)}</body></html>")

# Rendered by headless Chrome rather than WeasyPrint, whose wheel needs pango and
# gobject that are not installed here. Chrome is already present, prints the same
# CSS, and needs no new dependency for a one-file build.
src = pathlib.Path(out_p).with_suffix(".html")
src.write_text(html)
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
if not pathlib.Path(CHROME).exists():
    sys.exit(f"no renderer: {CHROME} not found")
profile = pathlib.Path(out_p).parent / ".chrome-print"
cmd = [CHROME, "--headless", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer",
       f"--user-data-dir={profile}", f"--print-to-pdf={out_p}", src.resolve().as_uri()]
r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
if not pathlib.Path(out_p).exists():
    sys.exit(f"chrome produced no pdf:\n{r.stderr[-800:]}")
shutil.rmtree(profile, ignore_errors=True)
src.unlink(missing_ok=True)
print(f"  wrote {out_p} ({pathlib.Path(out_p).stat().st_size/1e6:.1f} MB), "
      f"figures placed: {sorted(placed)}", flush=True)
print("PDF_DONE")
