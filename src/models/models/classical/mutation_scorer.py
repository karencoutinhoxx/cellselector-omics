import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import DATA_DIR, OMICS_PROFILES, SAMPLE_INFO

# ─────────────────────────────────────────────────────────────────────────────
# Mutation impact scoring — the PRIMARY signal for loss-of-function genes.
#
# LOF genes (BRCA1, BRCA2, RB1, ATM, TP53, ...) are selected by researchers on
# MUTATION STATUS (a damaging variant / deletion), not expression level, so
# ranking them by RNA — as every other scorer here does — is measuring the
# wrong thing. Cross-validated MRR for this class was near zero. This scorer
# reads real per-cell-line somatic variant calls and turns pathogenicity
# evidence into a 0-1 score per cell line.
#
# ID resolution reuses the exact ProfileID -> ModelID -> cellosaurus_id chain
# already used in scripts/preprocessing/preprocess_file2_depmap.py and
# merge/build_merged_table.py (the mutation CSV keys on ProfileID / PR- ids):
#     PR-  --(8_DepMap_OmicsProfiles.csv: ProfileID->ModelID)-->  ACH-
#     ACH- --(9_DepMap_sample_info.csv:   DepMap_ID->RRID)     -->  CVCL_
# ─────────────────────────────────────────────────────────────────────────────

MUTATION_FEATURES_PATH = DATA_DIR / "member3" / "mutation_gene_features.csv"

# Columns actually used (the CSV has 70) — keeps the one-time load to ~120 MB.
_USECOLS = [
    "HugoSymbol", "ProteinChange", "DNAChange", "VariantType",
    "MolecularConsequence", "VepImpact", "VepClinSig",
    "Hotspot", "LikelyLoF", "TranscriptLikelyLof",
    "OncogeneHighImpact", "TumorSuppressorHighImpact",
    "RevelScore", "AMClass", "AMPathogenicity",
    "ProfileID",
]

_MUTATION_CACHE: dict[str, pd.DataFrame] = {}
_MUTATION_DF: pd.DataFrame | None = None
_PR_TO_CVCL: dict[str, str] | None = None


def _load_mutation_id_mapping() -> dict[str, str]:
    """
    Build ProfileID (PR-) -> cellosaurus_id (CVCL_), reusing the same two
    nomenclature files and the same two-hop chain as
    preprocess_file2_depmap.py / build_merged_table.py. Not reimplemented —
    same construction, just chained through to a single dict here.

    ~84% of the mutation CSV's ProfileIDs resolve (the rest are profiles with
    no RRID in DepMap sample_info), matching the rate seen elsewhere.
    """
    global _PR_TO_CVCL
    if _PR_TO_CVCL is not None:
        return _PR_TO_CVCL

    prof_df = pd.read_csv(
        OMICS_PROFILES, usecols=["ProfileID", "ModelID"], low_memory=False
    )
    pr_to_ach = dict(zip(prof_df["ProfileID"], prof_df["ModelID"]))

    samp = pd.read_csv(SAMPLE_INFO, usecols=["DepMap_ID", "RRID"], low_memory=False)
    dep_to_cvcl = {r.DepMap_ID: r.RRID for r in samp.itertuples() if pd.notna(r.RRID)}

    _PR_TO_CVCL = {
        pr: dep_to_cvcl[ach]
        for pr, ach in pr_to_ach.items()
        if ach in dep_to_cvcl
    }
    return _PR_TO_CVCL


def _load_mutation_df() -> pd.DataFrame:
    """One-time load of the slim mutation table, with cellosaurus_id resolved."""
    global _MUTATION_DF
    if _MUTATION_DF is not None:
        return _MUTATION_DF

    if not MUTATION_FEATURES_PATH.exists():
        _MUTATION_DF = pd.DataFrame(columns=_USECOLS + ["cellosaurus_id"])
        return _MUTATION_DF

    df = pd.read_csv(MUTATION_FEATURES_PATH, usecols=_USECOLS, low_memory=False)
    df["cellosaurus_id"] = df["ProfileID"].map(_load_mutation_id_mapping())
    df = df.dropna(subset=["cellosaurus_id"])
    _MUTATION_DF = df
    return _MUTATION_DF


def _is_true(val) -> bool:
    return val is True or (isinstance(val, str) and val.strip().lower() == "true")


def _clinsig_is_pathogenic(val) -> bool:
    """
    VepClinSig is a '&'-delimited multi-value string, e.g.
    'pathogenic/likely_pathogenic&likely_pathogenic&pathogenic'. Token-split
    on both '&' and '/' and look for an explicit pathogenic call — while
    NOT counting 'conflicting_interpretations_of_pathogenicity', whose only
    match would be the substring 'pathogenicity'.
    """
    if not isinstance(val, str):
        return False
    tokens = {t.strip() for part in val.split("&") for t in part.split("/")}
    return bool(tokens & {"pathogenic", "likely_pathogenic",
                          "pathogenic_low_penetrance"})


def _clinsig_is_benign(val) -> bool:
    if not isinstance(val, str):
        return False
    tokens = {t.strip() for part in val.split("&") for t in part.split("/")}
    return bool(tokens & {"benign", "likely_benign"}) and not _clinsig_is_pathogenic(val)


def _num(val) -> float | None:
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _variant_impact(row) -> float:
    """
    Pathogenicity score (0-1) for a single variant call, combining whatever
    evidence is populated (many rows have only some fields).

    Categorical evidence (each a strong, independent positive signal):
      - Hotspot                        recurrent characterised cancer mutation
      - LikelyLoF / TranscriptLikelyLof     predicted loss-of-function
      - VepClinSig pathogenic/likely_pathogenic   clinical database call
      - HIGH VEP impact / TumorSuppressorHighImpact / OncogeneHighImpact
            nonsense / frameshift / splice — protein-truncating

    The more of these agree, the higher the score (3+ => 1.0). When NO
    categorical field is set, fall back to the continuous missense
    predictors (REVEL, AlphaMissense), which are already on a 0-1 scale.
    An explicit benign ClinSig with no countervailing evidence caps low.
    """
    hotspot     = _is_true(row.get("Hotspot"))
    lof         = _is_true(row.get("LikelyLoF")) or _is_true(row.get("TranscriptLikelyLof"))
    pathogenic  = _clinsig_is_pathogenic(row.get("VepClinSig"))
    high_impact = (
        str(row.get("VepImpact")).strip().upper() == "HIGH"
        or _is_true(row.get("TumorSuppressorHighImpact"))
        or _is_true(row.get("OncogeneHighImpact"))
    )

    strong = sum([hotspot, lof, pathogenic, high_impact])

    revel = _num(row.get("RevelScore"))
    am    = _num(row.get("AMPathogenicity"))
    continuous = max([v for v in (revel, am) if v is not None], default=0.0)

    if strong >= 3:
        categorical = 1.0
    elif strong == 2:
        categorical = 0.90
    elif strong == 1:
        categorical = 0.75
    else:
        categorical = 0.0

    if strong == 0:
        if _clinsig_is_benign(row.get("VepClinSig")):
            return min(continuous, 0.20)
        return continuous

    return max(categorical, continuous)


def _variant_detail(row, gene: str, score: float) -> str:
    """Short human-readable label for the highest-impact variant in a line."""
    pieces = [gene]
    pc = row.get("ProteinChange")
    if isinstance(pc, str) and pc.strip():
        pieces.append(pc.strip())
    elif isinstance(row.get("DNAChange"), str) and row["DNAChange"].strip():
        pieces.append(row["DNAChange"].strip())

    tags = []
    if _is_true(row.get("Hotspot")):
        tags.append("hotspot")
    if _is_true(row.get("LikelyLoF")) or _is_true(row.get("TranscriptLikelyLof")):
        tags.append("likely LoF")
    if _clinsig_is_pathogenic(row.get("VepClinSig")):
        tags.append("pathogenic")
    if str(row.get("VepImpact")).strip().upper() == "HIGH":
        tags.append("high impact")
    if not tags:
        revel = _num(row.get("RevelScore"))
        am = _num(row.get("AMPathogenicity"))
        if revel is not None:
            tags.append(f"REVEL {revel:.2f}")
        if am is not None:
            tags.append(f"AlphaMissense {am:.2f}")

    label = " ".join(pieces)
    if tags:
        label += ", " + ", ".join(tags)
    return f"{label} (impact {score:.3f})"


def score_mutation_impact(gene: str) -> pd.DataFrame:
    """
    Per-cell-line mutation impact score (0-1) for `gene`, from somatic
    variant pathogenicity evidence.

    A cell line with NO variant record for this gene gets 0.0, not NaN:
    absence of a call is itself informative (it most likely means the gene
    is wild-type in that line). When a line has multiple variants in the
    gene, the most damaging one wins (max).

    Returns: DataFrame [cellosaurus_id, mutation_impact_score, mutation_detail]
    — mutation_detail is a short string for agentic rationale, "" if none.
    Cached per gene (same pattern as score_pathway_activity /
    score_crispr_dependency).
    """
    if gene in _MUTATION_CACHE:
        return _MUTATION_CACHE[gene]

    df = _load_mutation_df()
    sub = df[df["HugoSymbol"] == gene]

    empty = pd.DataFrame({
        "cellosaurus_id":        pd.Series(dtype="object"),
        "mutation_impact_score": pd.Series(dtype="float64"),
        "mutation_detail":       pd.Series(dtype="object"),
    })
    if len(sub) == 0:
        _MUTATION_CACHE[gene] = empty
        return empty

    sub = sub.copy()
    sub["_impact"] = sub.apply(_variant_impact, axis=1)

    rows = []
    for cvcl, grp in sub.groupby("cellosaurus_id"):
        best = grp.loc[grp["_impact"].idxmax()]
        score = float(best["_impact"])
        rows.append({
            "cellosaurus_id":        cvcl,
            "mutation_impact_score": score,
            "mutation_detail":       _variant_detail(best, gene, score) if score > 0 else "",
        })

    result = pd.DataFrame(rows)
    _MUTATION_CACHE[gene] = result
    return result


def significant_variants(genes) -> pd.DataFrame:
    """
    Load-bearing subset for Neo4j graph ingestion — NOT the full 1M-row file.

    Rows for `genes` where the variant is LikelyLoF OR Hotspot OR a
    pathogenic / likely_pathogenic ClinSig call. Deduplicated to one row per
    (gene, cell line, protein_change), keeping the highest-impact call.

    Columns: gene, cellosaurus_id, protein_change, hotspot, likely_lof,
             clinical_significance, impact_score
    """
    cols = ["gene", "cellosaurus_id", "protein_change", "hotspot",
            "likely_lof", "clinical_significance", "impact_score"]
    df = _load_mutation_df()
    want = set(genes)
    sub = df[df["HugoSymbol"].isin(want)].copy()
    if len(sub) == 0:
        return pd.DataFrame(columns=cols)

    sub["_hotspot"]    = sub["Hotspot"].map(_is_true)
    sub["_lof"]        = sub["LikelyLoF"].map(_is_true) | sub["TranscriptLikelyLof"].map(_is_true)
    sub["_pathogenic"] = sub["VepClinSig"].map(_clinsig_is_pathogenic)
    sub = sub[sub["_hotspot"] | sub["_lof"] | sub["_pathogenic"]]
    if len(sub) == 0:
        return pd.DataFrame(columns=cols)

    sub["_impact"] = sub.apply(_variant_impact, axis=1)
    sub["_pchange"] = (
        sub["ProteinChange"].where(sub["ProteinChange"].notna(), sub["DNAChange"])
        .fillna("").astype(str).str.strip()
    )

    out = pd.DataFrame({
        "gene":                  sub["HugoSymbol"].values,
        "cellosaurus_id":        sub["cellosaurus_id"].values,
        "protein_change":        sub["_pchange"].values,
        "hotspot":               sub["_hotspot"].values,
        "likely_lof":            sub["_lof"].values,
        "clinical_significance": sub["VepClinSig"].fillna("").astype(str).values,
        "impact_score":          sub["_impact"].round(4).values,
    })
    out = (
        out.sort_values("impact_score", ascending=False)
           .drop_duplicates(subset=["gene", "cellosaurus_id", "protein_change"], keep="first")
           .reset_index(drop=True)
    )
    return out


if __name__ == "__main__":
    for g in ["BRCA1", "KRAS", "TP53", "ATM"]:
        r = score_mutation_impact(g)
        print(f"\n=== {g}: {len(r)} cell lines with a {g} variant ===")
        if len(r):
            print(r.sort_values("mutation_impact_score", ascending=False).head(10).to_string(index=False))

