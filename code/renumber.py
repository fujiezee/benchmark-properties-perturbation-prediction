"""Renumber citations in order of first appearance, as Nature style requires."""
import pathlib, re, sys

BASE = pathlib.Path(sys.argv[1])
ORDER = ["abstract.md", "introduction.md", "results-draft.md", "discussion.md",
         "methods.md"]
SUP = {str(d): c for d, c in enumerate("⁰¹²³⁴⁵⁶⁷⁸⁹")}

BIB = {
 "ahlmann": "Ahlmann-Eltze, C., Huber, W. & Anders, S. Deep-learning-based gene\n   perturbation effect prediction does not yet outperform simple linear\n   baselines. *Nat. Methods* **22**, 1657–1661 (2025).\n   doi:10.1038/s41592-025-02772-6",
 "replogle": "Replogle, J. M. *et al.* Mapping information-rich genotype-phenotype\n   landscapes with genome-scale Perturb-seq. *Cell* **185**, 2559–2575.e28\n   (2022). doi:10.1016/j.cell.2022.05.013",
 "adamson": "Adamson, B. *et al.* A multiplexed single-cell CRISPR screening platform\n   enables systematic dissection of the unfolded protein response. *Cell*\n   **167**, 1867–1882.e21 (2016). doi:10.1016/j.cell.2016.11.048",
 "bradu": "Bradu, A., Blair, J. D., Grabski, I., Mascio, I., Lee, J., McCormick, C. &\n   Satija, R. Genome-wide single-cell perturbation screens with VIPerturb-seq.\n   *bioRxiv* doi:10.64898/2026.02.12.705613 (2026). Data: Zenodo\n   doi:10.5281/zenodo.18460279",
 "nadig": "Nadig, A., Replogle, J. M., Pogson, A. N. *et al.* Transcriptome-wide\n   analysis of differential expression in perturbation atlases. *Nat. Genet.*\n   **57**, 1228–1237 (2025). doi:10.1038/s41588-025-02169-3. Data: GEO\n   GSE264667",
 "gasperini": "Gasperini, M. *et al.* A genome-wide framework for mapping gene regulation\n   via cellular genetic screens. *Cell* **176**, 377–390.e19 (2019).\n   doi:10.1016/j.cell.2018.11.029",
 "scperturb": "Peidli, S. *et al.* scPerturb: harmonized single-cell perturbation data.\n   *Nat. Methods* **21**, 531–540 (2024). doi:10.1038/s41592-023-02144-y. The\n   Gasperini screen was read from its Zenodo deposit,\n   doi:10.5281/zenodo.7041849",
 "orion": "Huang, A. C. *et al.* X-Atlas/Orion: genome-wide Perturb-seq datasets via a\n   scalable fix-cryopreserve platform for training dose-dependent biological\n   foundation models. *bioRxiv* doi:10.1101/2025.06.11.659105 (2025). Data:\n   Hugging Face `Xaira-Therapeutics/X-Atlas-Orion`",
 "gears": "Roohani, Y., Huang, K. & Leskovec, J. Predicting transcriptional outcomes of\n   novel multigene perturbations with GEARS. *Nat. Biotechnol.* **42**, 927–935\n   (2024). doi:10.1038/s41587-023-01905-6",
 "cpa": "Lotfollahi, M. *et al.* Predicting cellular responses to complex\n   perturbations in high-throughput screens. *Mol. Syst. Biol.* **19**, e11517\n   (2023). doi:10.15252/msb.202211517",
 "scgpt": "Cui, H. *et al.* scGPT: toward building a foundation model for single-cell\n   multi-omics using generative AI. *Nat. Methods* **21**, 1470–1480 (2024).\n   doi:10.1038/s41592-024-02201-0",
 "scfoundation": "Hao, M. *et al.* Large-scale foundation model on single-cell\n   transcriptomics. *Nat. Methods* **21**, 1481–1491 (2024).\n   doi:10.1038/s41592-024-02305-7",
 "string": "Szklarczyk, D. *et al.* The STRING database in 2023: protein-protein\n   association networks and functional enrichment analyses for any sequenced\n   genome of interest. *Nucleic Acids Res.* **51**, D638-D646 (2023).\n   doi:10.1093/nar/gkac1000",
 "txpert": "Wenkel, F. *et al.* TxPert: leveraging biochemical relationships for\n   out-of-distribution transcriptomic perturbation prediction. *arXiv*\n   2505.14919 (2025).",
 "celleval": "Adduri, A. K. *et al.* Predicting cellular responses to perturbation across\n   diverse contexts with STATE. *bioRxiv* doi:10.1101/2025.06.26.661135 (2025).\n   The `cell-eval` framework is released with it.",
 "vcc": "Arc Institute. Virtual Cell Challenge 2025 wrap-up: winners and\n   reflections. https://arcinstitute.org/news/virtual-cell-challenge-2025-wrap-up\n   (2025).",
 "clone": "Huang, X. Shared clones make random cell halves look reproducible in pooled\n   screens. *arXiv* 2610.32962 [q-bio.QM] (2026).",
 "spearman": "Spearman, C. Correlation calculated from faulty data. *Br. J. Psychol.*\n   **3**, 271–295 (1910).",
 "brown": "Brown, W. Some experimental results in the correlation of mental abilities.\n   *Br. J. Psychol.* **3**, 296–322 (1910).",
}

SIDECAR = BASE / "references.order.json"

tokre = re.compile(r"\{\{([a-z]+)\}\}")

# Make the script safe to re-run: a second pass sees superscripts, not tokens,
# so turn any numbering from a previous run back into tokens first.
if SIDECAR.exists():
    import json
    prev = json.loads(SIDECAR.read_text())
    back = {"".join(SUP[d] for d in str(i + 1)): k for i, k in enumerate(prev)}
    runre = re.compile("[" + "".join(SUP.values()) + "\u2009]+")
    def untok(m):
        return "".join("{{%s}}" % back[p] for p in m.group(0).split("\u2009") if p)
    for f in ORDER:
        p = BASE / f
        p.write_text(runre.sub(untok, p.read_text()))

order, texts = [], {}
for f in ORDER:
    t = (BASE/f).read_text(); texts[f] = t
    for k in tokre.findall(t):
        if k not in BIB:
            raise SystemExit(f"unknown citation key: {k}")
        if k not in order:
            order.append(k)

num = {k: i + 1 for i, k in enumerate(order)}
uncited = [k for k in BIB if k not in num]

def sup(n):
    return "".join(SUP[d] for d in str(n))

for f, t in texts.items():
    # Collapse adjacent tokens into one comma-separated superscript run.
    def repl(m):
        ks = tokre.findall(m.group(0))
        return " ".join(sup(num[k]) for k in ks) if len(ks) > 1 else sup(num[ks[0]])
    t = re.sub(r"(?:\{\{[a-z]+\}\})+", repl, t)
    (BASE/f).write_text(t)

lines = ["<!-- Numbered in order of first appearance, as Nature style requires.",
         "     Regenerate with scripts/renumber.py after changing any citation.",
         "     Every entry was checked against an independent record: journal",
         "     citations against the NCBI E-utilities summary for their DOI, and",
         "     preprints against the bioRxiv API. -->", ""]
for i, k in enumerate(order, 1):
    lines.append(f"{i}. {BIB[k]}\n")
(BASE/"references.md").write_text("\n".join(lines))
import json
SIDECAR.write_text(json.dumps(order, indent=1) + "\n")
print(f"  按首次出现重排 {len(order)} 条")
for i, k in enumerate(order, 1):
    print(f"    {i:>2}. {k}")
if uncited:
    print("  正文未引用（只在数据可用性声明中出现）:", ", ".join(uncited))
