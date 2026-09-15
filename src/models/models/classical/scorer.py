from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import (
    CELL_LINE_LOOKUP,
    GEO_INFO,
    MASTER_CONFIDENCE,
    PARQUET_DIR,
    SAMPLE_INFO,
)

# NOTE: 0.30+0.15+0.20+0.15+0.10 = 0.90, not the 1.00 stated when these
# were given — same gap as the previous FIXED_WEIGHTS revision, just with
# different individual numbers. geo_confirmation adds ±0.10 on top
# (additive, not part of this weighted sum), so a perfect score with a GEO
# confirmation bonus still reaches 1.00; without GEO data, the ceiling is
# 0.90. Kept these exact values rather than silently rescaling them —
# flagging again since the arithmetic still doesn't match the stated intent.
FIXED_WEIGHTS = {
    "rna":     0.35,
    "protein": 0.15,
    "quality": 0.20,
    "context": 0.15,
    "pathway": 0.15,
}

# GEO confirmation bonus/penalty — additive, not part of weighted sum
GEO_BONUS   = +0.10
GEO_PENALTY = -0.10

GENE_CLASSES: dict[str, list[str]] = {
    "ubiquitous": [
        "PARP1", "CDK4", "CCND1", "ACTB",
        "GAPDH", "MDM2", "CDK2", "CDK6", "PCNA", "MKI67",
        # TP53: its validation targets (HCT116, U-2 OS, MCF-7) are canonical
        # WILD-TYPE-p53 reference lines — the opposite selection criterion
        # from the mutation-driven LOF targets used for BRCA1/BRCA2/RB1/ATM.
        # Mutation-primary LOF scoring correctly sank those wild-type lines,
        # so TP53 does not belong in loss_of_function for this pipeline.
        "TP53",
    ],
    "loss_of_function": [
        "BRCA1", "BRCA2", "RB1", "ATM", "PTEN",
        "APC", "VHL", "MLH1", "MSH2",
        # Added via the CIViC-sourced independent validation-set expansion
        # (COSMIC CGC required registration we didn't have; CIViC's open
        # API + our own mutation_gene_features.csv verification stood in).
        "CDKN2A", "SMAD4", "STK11", "NF1", "ARID1A", "SMARCA4", "CDH1",
    ],
}

GENE_ROLES: dict[str, str] = {
    # Receptor tyrosine kinases
    "EGFR":   "receptor tyrosine kinase",
    "ERBB2":  "receptor tyrosine kinase (HER2)",
    "ERBB3":  "receptor tyrosine kinase (HER3)",
    "MET":    "receptor tyrosine kinase",
    "KIT":    "receptor tyrosine kinase",
    "ALK":    "receptor tyrosine kinase",
    "RET":    "receptor tyrosine kinase",
    "FLT3":   "receptor tyrosine kinase",
    "PDGFRA": "receptor tyrosine kinase",
    # Hormone receptors
    "ESR1": "estrogen receptor",
    "AR":   "androgen receptor",
    "PGR":  "progesterone receptor",
    # Immune checkpoint / surface markers
    "CD274": "immune checkpoint marker (PD-L1)",
    "PDCD1": "immune checkpoint marker (PD-1)",
    "CTLA4": "immune checkpoint marker",
    # Tumor suppressors
    "TP53":  "tumor suppressor",
    "RB1":   "tumor suppressor",
    "PTEN":  "tumor suppressor",
    "BRCA1": "tumor suppressor (DNA repair)",
    "BRCA2": "tumor suppressor (DNA repair)",
    "APC":   "tumor suppressor",
    "VHL":   "tumor suppressor",
    # Oncogenes / signaling
    "KRAS":   "oncogene (RAS family GTPase)",
    "BRAF":   "oncogene (kinase)",
    "MYC":    "oncogene (transcription factor)",
    "MYCN":   "oncogene (transcription factor)",
    "PIK3CA": "oncogene (kinase)",
    # Proliferation markers
    "MKI67": "proliferation marker (Ki-67)",
    "PCNA":  "proliferation marker",
    # Tumor suppressors — CIViC-sourced expansion
    "CDKN2A":  "tumor suppressor (cell-cycle inhibitor, p16)",
    "SMAD4":   "tumor suppressor (TGF-beta signaling)",
    "STK11":   "tumor suppressor (kinase, LKB1)",
    "NF1":     "tumor suppressor (RAS-GAP)",
    "ARID1A":  "tumor suppressor (chromatin remodeling, SWI/SNF)",
    "SMARCA4": "tumor suppressor (chromatin remodeling, SWI/SNF, BRG1)",
    "CDH1":    "tumor suppressor (cell adhesion, E-cadherin)",
    # Oncogenes / signaling — CIViC-sourced expansion
    "NRAS":   "oncogene (RAS family GTPase)",
    "CTNNB1": "oncogene (WNT signaling, beta-catenin)",
    "AKT1":   "oncogene (kinase)",
    "NOTCH1": "oncogene (receptor, context-dependent — also a tumor suppressor in some tissues)",
    "JAK2":   "oncogene (kinase)",
    "MAP2K1": "oncogene (kinase, MEK1)",
    "FGFR1":  "receptor tyrosine kinase (amplification-driven)",
}


def get_gene_role(gene: str) -> str | None:
    return GENE_ROLES.get(gene.upper())


def classify_gene(gene: str) -> str:
    if gene in GENE_CLASSES["loss_of_function"]:
        return "loss_of_function"
    if gene in GENE_CLASSES["ubiquitous"]:
        return "ubiquitous"
    return "tissue_specific"


def load_mappings() -> tuple[dict, dict, dict]:
    """Return (hpa_to_cvcl, ach_to_cvcl, gsm_to_cvcl)."""
    lkp = pd.read_parquet(CELL_LINE_LOOKUP, columns=["cellosaurus_id", "hpa_name"])
    # Filter both columns from the SAME row-aligned frame before zipping —
    # zipping two independently-.dropna()'d Series silently pairs values by
    # position, not by original row, whenever the two columns have
    # different NaN counts (hpa_name has many more NaNs than cellosaurus_id
    # here, which shifted ~94% of pairs onto the wrong cell line).
    lkp_hpa = lkp.dropna(subset=["hpa_name"])
    hpa_to_cvcl = dict(zip(lkp_hpa["hpa_name"], lkp_hpa["cellosaurus_id"]))

    samp = pd.read_csv(SAMPLE_INFO, usecols=["DepMap_ID", "RRID"], low_memory=False)
    ach_to_cvcl = {r.DepMap_ID: r.RRID for r in samp.itertuples() if pd.notna(r.RRID)}

    geo = pd.read_csv(
        GEO_INFO, sep="\t",
        usecols=["Geo_accession", "Cellosaurus_ID"],
        low_memory=False,
    )
    geo_valid = geo.dropna(subset=["Geo_accession", "Cellosaurus_ID"])
    gsm_to_cvcl = dict(zip(geo_valid["Geo_accession"], geo_valid["Cellosaurus_ID"]))

    return hpa_to_cvcl, ach_to_cvcl, gsm_to_cvcl


_TIE_BREAK_KEYS = ("mutation_impact_score", "rna_score", "quality_score")


def rank_sort(df: "pd.DataFrame", score_col: str) -> "pd.DataFrame":
    """
    Sort `df` for ranking / reciprocal-rank: primary key `score_col`
    descending, then a fixed chain of raw (pre-clip) discriminators, then
    cellosaurus_id ascending as the final deterministic fallback.

    final_score is clipped to [0, 1] (see ranker.rank / mrr_score), so
    strongly-mutated loss_of_function lines — and strong-expression + GEO
    tissue_specific lines — pile up at an identical primary score of exactly
    1.0. Without these extra keys, the order among them is just whatever row
    order the merges happened to leave, and the "winner" of a large tie
    block is decided by cellosaurus_id sort alone (which inflated BRCA1's
    LOO-CV RR to 1.0). mutation_impact_score and rna_score are NOT clipped,
    so they still separate those rows.

    Returns a new frame with cellosaurus_id as a column and a clean
    RangeIndex (callers read rank as row position).
    """
    if "cellosaurus_id" not in df.columns:
        df = df.reset_index()
    keys = [score_col]
    ascending = [False]
    for k in _TIE_BREAK_KEYS:
        if k != score_col and k in df.columns:
            keys.append(k)
            ascending.append(False)
    keys.append("cellosaurus_id")
    ascending.append(True)
    return df.sort_values(keys, ascending=ascending, kind="stable").reset_index(drop=True)


def _pct_rank(series: pd.Series) -> np.ndarray:
    return series.rank(pct=True, method="average").values


def _level_label_with_percentile(score: float | None, percentile: float | None) -> dict:
    """
    Qualitative Low/Medium/High label PLUS the precise underlying number,
    so the coarse label never has to stand alone — a user can always drill
    down to the exact score/percentile behind it.

    score and percentile are typically the same value here: hpa_score /
    depmap_score / protein_score are themselves already 0-1 percentile
    ranks (see _pct_rank, used by score_rna_expression / score_protein_expression),
    not raw magnitudes — so the "percentile" IS the score, not a
    re-ranking of it. (A caller could pass a genuinely different
    percentile if one were ever computed separately.)

    Returns {"label", "score", "percentile"} — percentile pre-formatted
    as "68th percentile" (or None) so callers can drop it straight into
    a UI or an LLM prompt.
    """
    if score is None or pd.isna(score):
        return {"label": "No data", "score": None, "percentile": None}

    if score >= 0.66:
        label = "High"
    elif score >= 0.33:
        label = "Medium"
    else:
        label = "Low"

    pct_display = (
        f"{percentile * 100:.0f}th percentile"
        if percentile is not None and pd.notna(percentile)
        else None
    )

    return {
        "label":      label,
        "score":      round(float(score), 3),
        "percentile": pct_display,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY RNA SCORING  (HPA + DepMap)
# GEO is computed separately and returned as a ±0.10 confirmation signal
# ─────────────────────────────────────────────────────────────────────────────

def score_rna_expression(
    gene: str,
    hpa_to_cvcl: dict,
    gsm_to_cvcl: dict,
    gene_class: str | None = None,
) -> pd.DataFrame:
    """
    Score RNA expression for a gene across all cell lines.

    HPA + DepMap are the primary sources (equal weight when both present).
    GEO acts as a confirmation signal only (±0.10 additive bonus, not mixed
    into the percentile ranking).

    gene_class controls the RNA scoring strategy:
      - "tissue_specific" / None  →  rna_score = mean(hpa_rank, depmap_rank)
      - "ubiquitous"              →  rna_score = 1 - CV(hpa_rank, depmap_rank)
                                      where CV = |h-d| / (h+d)
                                      (consistency beats expression level)
      - "loss_of_function"        →  same as tissue_specific
                                      (caller adds LOF flag in the result)

    Returns columns:
        cellosaurus_id, rna_score, hpa_score, depmap_score,
        n_primary_sources, missing_data_flag,
        geo_confirmation, geo_n_samples
    """
    _empty = pd.DataFrame(
        columns=[
            "cellosaurus_id", "rna_score", "hpa_score", "depmap_score",
            "n_primary_sources", "missing_data_flag",
            "geo_confirmation", "geo_n_samples",
        ]
    )

    # ── HPA: nTPM only, threshold > 1 ────────────────────────────────────────
    hpa_raw = pd.read_parquet(
        PARQUET_DIR / "gene_expr_hpa_preprocessed.parquet",
        filters=[("gene_symbol", "=", gene), ("units", "=", "nTPM")],
        columns=["original_id", "expression_value"],
    )
    hpa_df = pd.DataFrame(columns=["cellosaurus_id", "hpa_score"])
    if len(hpa_raw) > 0:
        agg = hpa_raw.groupby("original_id", observed=True)["expression_value"].mean()
        agg = agg[agg > 1.0]
        if len(agg) > 0:
            cvcl = agg.index.map(hpa_to_cvcl)
            mask = cvcl.notna()
            hpa_df = (
                pd.DataFrame({
                    "cellosaurus_id": cvcl[mask].values,
                    "hpa_score": _pct_rank(agg[mask]),
                })
                .groupby("cellosaurus_id")["hpa_score"].mean()
                .reset_index()
            )

    # ── DepMap: TPM_log1p, threshold > 0.5 ───────────────────────────────────
    dep_raw = pd.read_parquet(
        PARQUET_DIR / "gene_expr_depmap_preprocessed.parquet",
        filters=[("gene_symbol", "=", gene)],
        columns=["cellosaurus_id", "expression_value"],
    )
    dep_df = pd.DataFrame(columns=["cellosaurus_id", "depmap_score"])
    if len(dep_raw) > 0:
        dep_raw = dep_raw.dropna(subset=["cellosaurus_id"])
        dep_raw = dep_raw[dep_raw["expression_value"] > 0.5]
        if len(dep_raw) > 0:
            agg = dep_raw.groupby("cellosaurus_id", observed=True)["expression_value"].mean()
            dep_df = pd.DataFrame({
                "cellosaurus_id": agg.index.astype(str),
                "depmap_score": _pct_rank(agg),
            })

    # ── GEO: confirmation signal only ────────────────────────────────────────
    # Multiple GSMs per cell line → take median; CV = std/|median|
    geo_raw = pd.read_parquet(
        PARQUET_DIR / "gene_expr_geo_preprocessed.parquet",
        filters=[("gene_symbol", "=", gene)],
        columns=["original_id", "expression_value"],
    )
    geo_df = pd.DataFrame(columns=["cellosaurus_id", "geo_expressed", "geo_cv", "geo_n_samples"])
    if len(geo_raw) > 0:
        geo_raw["cellosaurus_id"] = geo_raw["original_id"].map(gsm_to_cvcl)
        geo_raw = geo_raw.dropna(subset=["cellosaurus_id"])
        if len(geo_raw) > 0:
            g = geo_raw.groupby("cellosaurus_id")["expression_value"]
            med = g.median()
            std = g.std().fillna(0.0)
            n   = g.count()
            cv  = (std / med.abs().clip(lower=1e-9)).clip(upper=10.0)
            geo_df = pd.DataFrame({
                "cellosaurus_id": med.index,
                "geo_median":     med.values,
                "geo_cv":         cv.values,
                "geo_n_samples":  n.values,
                "geo_expressed":  ((med.values > 0) & (cv.values < 0.5)),
            })

    # ── Merge primary sources ─────────────────────────────────────────────────
    all_cvcl = set(hpa_df["cellosaurus_id"]) | set(dep_df["cellosaurus_id"])
    if not all_cvcl:
        return _empty

    result = (
        pd.DataFrame({"cellosaurus_id": list(all_cvcl)})
        .merge(hpa_df, on="cellosaurus_id", how="left")
        .merge(dep_df, on="cellosaurus_id", how="left")
    )

    # Combine HPA and DepMap; strategy depends on gene_class
    def _rna(row):
        h, d = row["hpa_score"], row["depmap_score"]
        has_h, has_d = pd.notna(h), pd.notna(d)
        if gene_class == "ubiquitous":
            if has_h and has_d:
                # Cross-source consistency: 1 - |h-d|/(h+d)
                denom = float(h) + float(d)
                cv = abs(float(h) - float(d)) / denom if denom > 0 else 0.0
                return 1.0 - cv, 2, False
            elif has_h or has_d:
                return 0.5, 1, False   # neutral: can't assess consistency
            return 0.0, 0, True
        else:
            # tissue_specific and loss_of_function: percentile-rank average
            if has_h and has_d:
                return 0.5 * float(h) + 0.5 * float(d), 2, False
            elif has_h:
                return float(h), 1, False
            elif has_d:
                return float(d), 1, False
            return 0.0, 0, True   # flagged: no primary data

    tmp = result.apply(_rna, axis=1, result_type="expand")
    result["rna_score"]         = tmp[0]
    result["n_primary_sources"] = tmp[1].astype(int)
    result["missing_data_flag"] = tmp[2]

    # ── GEO confirmation bonus / penalty ─────────────────────────────────────
    result = result.merge(
        geo_df[["cellosaurus_id", "geo_expressed", "geo_n_samples"]],
        on="cellosaurus_id", how="left",
    )

    def _geo_conf(row):
        n = row["geo_n_samples"]
        if pd.isna(n):
            return 0.0                               # no GEO data → neutral
        # Scale the ±0.10 by GEO sample count: one sample is weak evidence of
        # agreement (or disagreement), several concordant samples is strong.
        #   n >= 3 → full   | n == 2 → 0.6x (+/-0.06) | n == 1 → 0.3x (+/-0.03)
        # The geo_expressed consistency gate (median > 0 AND cv < 0.5) is
        # unchanged — only the magnitude now depends on n.
        n = int(n)
        scale = 1.0 if n >= 3 else (0.6 if n == 2 else (0.3 if n == 1 else 0.0))
        if row["geo_expressed"]:
            return round(GEO_BONUS * scale, 3)       # GEO confirms expression
        elif row["rna_score"] > 0:
            return round(GEO_PENALTY * scale, 3)     # GEO contradicts primary
        return 0.0                                   # both say absent → neutral

    result["geo_confirmation"] = result.apply(_geo_conf, axis=1)

    return result[[
        "cellosaurus_id", "rna_score", "hpa_score", "depmap_score",
        "n_primary_sources", "missing_data_flag",
        "geo_confirmation", "geo_n_samples",
    ]]


def score_protein_expression(gene: str, ach_to_cvcl: dict) -> pd.DataFrame:
    """
    Percentile-rank protein expression from CCLE MS proteomics.
    No threshold — MS intensity scale is different from RNA.

    Returns columns: cellosaurus_id, protein_score
    """
    prot_raw = pd.read_parquet(
        PARQUET_DIR / "gene_expr_ccle_proteomics_preprocessed.parquet",
        filters=[("gene_symbol", "=", gene)],
        columns=["original_id", "expression_value"],
    )
    if len(prot_raw) == 0:
        return pd.DataFrame({
            "cellosaurus_id": pd.Series(dtype="object"),
            "protein_score":  pd.Series(dtype="float64"),
        })

    prot_raw["cellosaurus_id"] = prot_raw["original_id"].map(ach_to_cvcl)
    prot_raw = prot_raw.dropna(subset=["cellosaurus_id"])
    if len(prot_raw) == 0:
        return pd.DataFrame({
            "cellosaurus_id": pd.Series(dtype="object"),
            "protein_score":  pd.Series(dtype="float64"),
        })

    agg = prot_raw.groupby("cellosaurus_id")["expression_value"].mean()
    return pd.DataFrame({"cellosaurus_id": agg.index, "protein_score": _pct_rank(agg)})


# ─────────────────────────────────────────────────────────────────────────────
# CRISPR DEPENDENCY SCORING (DepMap)
# Gene ESSENTIALITY, not expression — kept as a separate evidence column,
# never mixed into the weighted final_score (see ranker.rank()).
# ─────────────────────────────────────────────────────────────────────────────

def score_crispr_dependency(gene: str) -> pd.DataFrame:
    """
    Load CRISPR dependency scores for a gene.

    High dependency = gene is essential for that cell line's survival —
    a different signal from high expression (a gene can be highly expressed
    without being essential, and vice versa).

    Returns columns: cellosaurus_id, dependency_score, dependency_percentile
    """
    _empty = pd.DataFrame(
        columns=["cellosaurus_id", "dependency_score", "dependency_percentile"]
    )

    import os as _os
    if not _os.path.exists(PARQUET_DIR / "crispr_dependency_depmap_preprocessed.parquet"):
        return _empty
    crispr_raw = pd.read_parquet(
        PARQUET_DIR / "crispr_dependency_depmap_preprocessed.parquet",
        filters=[("gene_symbol", "=", gene)],
        columns=["cellosaurus_id", "dependency_score"],
    )
    crispr_raw = crispr_raw.dropna(subset=["cellosaurus_id"])
    if len(crispr_raw) == 0:
        return _empty

    agg = crispr_raw.groupby("cellosaurus_id")["dependency_score"].mean()
    return pd.DataFrame({
        "cellosaurus_id":         agg.index,
        "dependency_score":       agg.values,
        "dependency_percentile":  _pct_rank(agg),
    })


def score_data_quality(
    cellosaurus_ids,
    rna_df: pd.DataFrame,
    protein_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Data quality score per cell line.

    quality = 0.40*(n_primary_sources/2)
            + 0.40*(1 - cross_source_std_HPA_DepMap)
            + 0.20*completeness

    cross_source_std is between HPA and DepMap only (GEO excluded — units unknown).

    Returns columns: cellosaurus_id, quality_score, n_sources,
                     cross_source_std, completeness
    """
    conf = pd.read_parquet(MASTER_CONFIDENCE, columns=["cellosaurus_id", "confidence"])

    rna_cols = ["cellosaurus_id", "n_primary_sources", "hpa_score", "depmap_score"]
    rna_sub  = rna_df[[c for c in rna_cols if c in rna_df.columns]]

    result = (
        pd.DataFrame({"cellosaurus_id": list(cellosaurus_ids)})
        .merge(rna_sub, on="cellosaurus_id", how="left")
        .merge(conf, on="cellosaurus_id", how="left")
    )
    for col in ["n_primary_sources", "hpa_score", "depmap_score"]:
        if col not in result.columns:
            result[col] = np.nan

    # std between HPA and DepMap only — NaN if either missing (then =0)
    result["cross_source_std"] = (
        result[["hpa_score", "depmap_score"]]
        .std(axis=1, skipna=True)
        .fillna(0.0)
    )
    result["n_sources"]    = result["n_primary_sources"].fillna(0.0)
    result["completeness"] = result["confidence"].fillna(0.0)

    result["quality_score"] = (
        0.40 * (result["n_sources"] / 2.0)        # /2 = two primary sources
        + 0.40 * (1.0 - result["cross_source_std"])
        + 0.20 * result["completeness"]
    )

    return result[["cellosaurus_id", "quality_score", "n_sources",
                   "cross_source_std", "completeness"]]


def score_context(
    cellosaurus_ids,
    disease_filter: str | None = None,
    lineage_filter: str | None = None,
) -> pd.DataFrame:
    """
    Context relevance score.

    disease_match: 1.0 exact, 0.5 partial (substring), 0.0 none
    lineage_match: 1.0 match, 0.0 none
    context_score = max(disease_match, lineage_match)

    Returns columns: cellosaurus_id, context_score, disease, lineage
    """
    lkp = pd.read_parquet(
        CELL_LINE_LOOKUP, columns=["cellosaurus_id", "disease", "lineage"]
    )
    result = (
        pd.DataFrame({"cellosaurus_id": list(cellosaurus_ids)})
        .merge(lkp, on="cellosaurus_id", how="left")
    )

    def _dmatch(val: str | float) -> float:
        if not pd.notna(val) or not val:
            return 0.0
        v, f = str(val).lower().strip(), disease_filter.lower().strip()
        if v == f:
            return 1.0
        return 0.5 if (f in v or v in f) else 0.0

    def _lmatch(val: str | float) -> float:
        if not pd.notna(val) or not val:
            return 0.0
        v, f = str(val).lower().strip(), lineage_filter.lower().strip()
        return 1.0 if (f in v or v == f) else 0.0

    result["disease_match"] = (
        result["disease"].apply(_dmatch) if disease_filter else 0.0
    )
    result["lineage_match"] = (
        result["lineage"].apply(_lmatch) if lineage_filter else 0.0
    )
    result["context_score"] = result[["disease_match", "lineage_match"]].max(axis=1)

    return result[["cellosaurus_id", "context_score", "disease", "lineage"]]


if __name__ == "__main__":
    print("Scorer self-test (EGFR)...")
    hpa_to_cvcl, ach_to_cvcl, gsm_to_cvcl = load_mappings()

    rna = score_rna_expression("EGFR", hpa_to_cvcl, gsm_to_cvcl)
    print(f"RNA: {len(rna)} cell lines | "
          f"GEO bonus: {(rna['geo_confirmation'] > 0).sum()} | "
          f"GEO penalty: {(rna['geo_confirmation'] < 0).sum()} | "
          f"missing: {rna['missing_data_flag'].sum()}")
    print(rna.sort_values("rna_score", ascending=False).head(5).to_string(index=False))

    prot = score_protein_expression("EGFR", ach_to_cvcl)
    print(f"\nProtein: {len(prot)} cell lines")
    print(prot.sort_values("protein_score", ascending=False).head(5).to_string(index=False))

    crispr = score_crispr_dependency("EGFR")
    print(f"\nCRISPR dependency: {len(crispr)} cell lines")
    print(crispr.sort_values("dependency_score", ascending=False).head(5).to_string(index=False))

