from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
GENE_EXPR_DIR = DATA_DIR / "gene expression"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PARQUET_DIR = OUTPUTS_DIR / "parquet"
FIGURES_DIR = PROJECT_ROOT / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"

# ── Nomenclature / spine sources ──────────────────────────────────────────────
NOMENCLATURE_DIR  = DATA_DIR / "nomenclature"
SAMPLE_INFO       = NOMENCLATURE_DIR / "9_DepMap_sample_info.csv"
HPA_DESC          = NOMENCLATURE_DIR / "11_hpa_rna_celline_description.tsv"
GEO_INFO          = NOMENCLATURE_DIR / "10_GEOInfo.txt"
CELLOSAURUS       = NOMENCLATURE_DIR / "7_cellosaurus.csv"
OMICS_PROFILES    = NOMENCLATURE_DIR / "8_DepMap_OmicsProfiles.csv"
CELL_LINE_LOOKUP  = OUTPUTS_DIR / "cell_line_lookup.parquet"

MODELS_DIR        = PROJECT_ROOT / "models"
CLASSICAL_DIR     = MODELS_DIR / "classical"
MASTER_MERGED     = OUTPUTS_DIR / "master_merged.parquet"
MASTER_CONFIDENCE = OUTPUTS_DIR / "master_with_confidence.parquet"

FILES = {
    "hpa":      GENE_EXPR_DIR / "1_4_hpa_rna_celline.tsv",
    "depmap":   GENE_EXPR_DIR / "2_DepMap_OmicsExpressionAllGenesTPMLogp1Profile.csv",
    "geo":      GENE_EXPR_DIR / "3_GEOexpression.txt",
    "ms_ccle":  GENE_EXPR_DIR / "4_Harmonized_MS_CCLE_Gygi_subsetted.csv",
}

# DepMap 23Q4 gene-level copy number (log2 relative copy ratio), keyed by
# ModelID (ACH-) — NOT ProfileID (PR-) like the expression/mutation files.
# Matches the release of OMICS_PROFILES / FILES["depmap"] above (both 23Q4),
# deliberately NOT the mutations file (24Q4) — see
# models/classical/copy_number_scorer.py for the resolution chain.
COPY_NUMBER_FILE = GENE_EXPR_DIR / "OmicsCNGene.csv"

# ── Dataset citations (D1-D5) ─────────────────────────────────────────────────
# Used consistently across similarity results, agentic justifications, and UI.
DATASET_CITATIONS: dict[str, dict] = {
    "D1": {
        "key":      "D1",
        "name":     "Human Protein Atlas (HPA) RNA",
        "citation": "Uhlén M et al. (2015). Tissue-based map of the human proteome. Science 347(6220):1260419.",
        "pmid":     "25613900",
        "url":      "https://www.proteinatlas.org",
    },
    "D2": {
        "key":      "D2",
        "name":     "DepMap RNA (CCLE)",
        "citation": "Ghandi M et al. (2019). Next-generation characterization of the Cancer Cell Line Encyclopedia. Nature 569:503-508.",
        "pmid":     "31068700",
        "url":      "https://depmap.org/portal",
    },
    "D3": {
        "key":      "D3",
        "name":     "GEO Expression",
        "citation": "Barrett T et al. (2013). NCBI GEO: archive for functional genomics data sets. Nucleic Acids Res 41:D991-5.",
        "pmid":     "23193258",
        "url":      "https://www.ncbi.nlm.nih.gov/geo",
    },
    "D4": {
        "key":      "D4",
        "name":     "CCLE Proteomics (Gygi MS)",
        "citation": "Nusinow DP et al. (2020). Quantitative proteomics of the Cancer Cell Line Encyclopedia. Cell 180:387-402.",
        "pmid":     "31978347",
        "url":      "https://depmap.org/portal/download",
    },
    "D5": {
        "key":      "D5",
        "name":     "Cellosaurus",
        "citation": "Bairoch A (2018). The Cellosaurus, a cell-line knowledge resource. J Biomol Tech 29:25-38.",
        "pmid":     "29805321",
        "url":      "https://www.cellosaurus.org",
    },
}


DATASET_CITATIONS: dict[str, dict] = {
    "HPA_RNA": {
        "name":      "Human Protein Atlas (HPA) — RNA Expression",
        "citation":  "Uhlén M et al. Tissue-based map of the human proteome. Science. 2015;347(6220):1260419.",
        "pmid":      "25613900",
        "url":       "https://www.proteinatlas.org/",
    },
    "DepMap_TPM": {
        "name":      "Cancer Dependency Map (DepMap) — RNA Expression",
        "citation":  "Ghandi M et al. Next-generation characterization of the Cancer Cell Line Encyclopedia. Nature. 2019;569(7757):503-508.",
        "pmid":      "31068700",
        "url":       "https://depmap.org/",
    },
    "GEO_expression": {
        "name":      "NCBI Gene Expression Omnibus (GEO)",
        "citation":  "Barrett T et al. NCBI GEO: archive for functional genomics data sets—update. Nucleic Acids Res. 2013;41:D991-5.",
        "pmid":      "23193258",
        "url":       "https://www.ncbi.nlm.nih.gov/geo/",
    },
    "CCLE_proteomics": {
        "name":      "Cancer Cell Line Encyclopedia (CCLE) — MS Proteomics",
        "citation":  "Nusinow DP et al. Quantitative Proteomics of the Cancer Cell Line Encyclopedia. Cell. 2020;180(2):387-402.",
        "pmid":      "31978347",
        "url":       "https://portals.broadinstitute.org/ccle",
    },
    "Cellosaurus": {
        "name":      "Cellosaurus",
        "citation":  "Bairoch A. The Cellosaurus, a Cell-Line Knowledge Resource. J Biomol Tech. 2018;29(2):25-38.",
        "pmid":      "29805321",
        "url":       "https://www.cellosaurus.org/",
    },
}


def setup_dirs():
    for d in (DATA_DIR, GENE_EXPR_DIR, OUTPUTS_DIR, PARQUET_DIR, FIGURES_DIR, DOCS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Gene expr dir: {GENE_EXPR_DIR}")
    print(f"Parquet dir  : {PARQUET_DIR}")
    missing = [name for name, path in FILES.items() if not path.exists()]
    if missing:
        print(f"WARNING - missing source files: {missing}")
    else:
        print("All source data files found.")


if __name__ == "__main__":
    setup_dirs()
