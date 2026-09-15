from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# ── Feature vector definition ─────────────────────────────────────────────────
FEATURE_NAMES: list[str] = [
    # Data coverage (binary, 8)
    "has_metabolomics", "has_mirna", "has_proteomics",
    "has_hpa_expr", "has_depmap_expr", "has_geo_expr",
    "has_mutations", "has_fusions",
    # Genomic characteristics (normalised 0-1, 4)
    "MSIScore_norm", "Ploidy_norm", "CIN_norm", "evidence_norm",
    # Gene-specific expression profile (0-1, 7)
    "hpa_score", "depmap_score", "geo_score",
    "protein_score", "rna_score", "quality_score", "context_score",
]

_FEATURE_LABELS: dict[str, str] = {
    "has_metabolomics": "metabolomics data coverage",
    "has_mirna":        "miRNA data coverage",
    "has_proteomics":   "proteomics coverage",
    "has_hpa_expr":     "HPA RNA expression data",
    "has_depmap_expr":  "DepMap RNA expression data",
    "has_geo_expr":     "GEO expression data",
    "has_mutations":    "mutation data",
    "has_fusions":      "fusion gene data",
    "MSIScore_norm":    "microsatellite instability score",
    "Ploidy_norm":      "genomic ploidy",
    "CIN_norm":         "chromosomal instability index",
    "evidence_norm":    "nomenclature source coverage",
    "hpa_score":        "HPA expression level",
    "depmap_score":     "DepMap expression level",
    "geo_score":        "GEO expression signal",
    "protein_score":    "protein expression level",
    "rna_score":        "combined RNA expression",
    "quality_score":    "data quality",
    "context_score":    "tissue/disease relevance",
}

_DATA_TYPE_MAP: dict[str, str] = {
    "has_hpa_expr":     "HPA",
    "has_depmap_expr":  "DepMap",
    "has_geo_expr":     "GEO",
    "has_proteomics":   "Proteomics",
    "has_metabolomics": "Metabolomics",
    "has_mirna":        "miRNA",
    "has_mutations":    "Mutations",
    "has_fusions":      "Fusions",
}

# ── Feature weights ───────────────────────────────────────────────────────────
FEATURE_WEIGHTS: dict[str, float] = {
    "has_metabolomics": 0.3,
    "has_mirna":        0.3,
    "has_proteomics":   0.5,
    "has_hpa_expr":     0.5,
    "has_depmap_expr":  0.5,
    "has_geo_expr":     0.5,
    "has_mutations":    0.3,
    "has_fusions":      0.3,
    "MSIScore_norm":    1.0,
    "Ploidy_norm":      1.0,
    "CIN_norm":         1.0,
    "evidence_norm":    1.0,
    "hpa_score":        2.5,
    "depmap_score":     2.5,
    "geo_score":        2.0,
    "protein_score":    2.0,
    "rna_score":        3.0,
    "quality_score":    2.0,
    "context_score":    2.0,
}

_WEIGHT_VECTOR = np.array(
    [FEATURE_WEIGHTS[f] for f in FEATURE_NAMES], dtype=np.float64
)

_EXPRESSION_FEATURES = frozenset(
    ["rna_score", "hpa_score", "depmap_score", "protein_score",
     "geo_score", "quality_score"]
)

# ── Dataset citations (D1-D5) ─────────────────────────────────────────────────
# Canonical metadata shared across similarity, agentic pipeline, and API.
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


# ── Feature matrix ────────────────────────────────────────────────────────────

def build_feature_matrix(
    gene: str,
    rank_results_df: pd.DataFrame,
    all_cell_lines_df: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """
    Build a normalised, weighted (n_cells × 19) feature matrix for all cell lines.

    Expression scores come from rank_results_df where available; unranked
    cell lines receive 0 for expression features.
    Returns (weighted_matrix, cellosaurus_ids).
    """
    df = all_cell_lines_df.copy()

    for col in list(_DATA_TYPE_MAP):
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(float)
        else:
            df[col] = 0.0

    def _mm(series: pd.Series, fill_val: float) -> pd.Series:
        s = series.fillna(fill_val)
        lo, hi = float(s.min()), float(s.max())
        if hi > lo:
            return (s - lo) / (hi - lo)
        return pd.Series(0.0, index=s.index)

    if "MSIScore" in df.columns:
        df["MSIScore_norm"] = _mm(df["MSIScore"], fill_val=0.0)
    else:
        df["MSIScore_norm"] = 0.0

    if "Ploidy" in df.columns:
        df["Ploidy_norm"] = _mm(df["Ploidy"], fill_val=float(df["Ploidy"].median()))
    else:
        df["Ploidy_norm"] = 0.0

    if "CIN" in df.columns:
        df["CIN_norm"] = _mm(df["CIN"], fill_val=0.0)
    else:
        df["CIN_norm"] = 0.0

    if "evidence_count" in df.columns:
        df["evidence_norm"] = (df["evidence_count"].fillna(0) / 3.0).clip(0.0, 1.0)
    else:
        df["evidence_norm"] = 0.0

    expr_src = [
        "cellosaurus_id", "hpa_score", "depmap_score", "geo_confirmation",
        "protein_score", "rna_score", "quality_score", "context_score",
    ]
    if rank_results_df is not None and len(rank_results_df) > 0:
        avail    = [c for c in expr_src if c in rank_results_df.columns]
        expr_sub = rank_results_df[avail].copy()
        if "geo_confirmation" in expr_sub.columns:
            expr_sub["geo_score"] = (
                (expr_sub["geo_confirmation"] / 0.10) + 1.0
            ) / 2.0
            expr_sub["geo_score"] = expr_sub["geo_score"].clip(0.0, 1.0).fillna(0.5)
            expr_sub = expr_sub.drop(columns=["geo_confirmation"])
        df = df.merge(expr_sub, on="cellosaurus_id", how="left")

    for col in ["hpa_score", "depmap_score", "geo_score",
                "protein_score", "rna_score", "quality_score", "context_score"]:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    df = df.reset_index(drop=True)
    matrix          = df[FEATURE_NAMES].values.astype(np.float64) * _WEIGHT_VECTOR
    cellosaurus_ids = df["cellosaurus_id"].tolist()
    return matrix, cellosaurus_ids


# ── Cosine similarity ─────────────────────────────────────────────────────────

def compute_similarity(
    query_cellosaurus_ids: list[str],
    feature_matrix: np.ndarray,
    cellosaurus_ids: list[str],
    top_k: int = 5,
) -> dict[str, list[dict]]:
    """Vectorised cosine similarity; query lines excluded from their own results."""
    id_to_idx = {cvcl: i for i, cvcl in enumerate(cellosaurus_ids)}
    query_set = set(query_cellosaurus_ids)

    norms  = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
    norms  = np.where(norms == 0, 1e-8, norms)
    normed = feature_matrix / norms

    output: dict[str, list[dict]] = {}
    for qcvcl in query_cellosaurus_ids:
        if qcvcl not in id_to_idx:
            output[qcvcl] = []
            continue
        q_idx = id_to_idx[qcvcl]
        sims  = normed @ normed[q_idx]
        for excl in query_set:
            if excl in id_to_idx:
                sims[id_to_idx[excl]] = -1.0
        top_idxs = np.argsort(sims)[::-1][:top_k]
        output[qcvcl] = [
            {
                "cellosaurus_id":   cellosaurus_ids[idx],
                "similarity_score": float(round(float(sims[idx]), 4)),
                "_query_vec":       feature_matrix[q_idx],
                "_similar_vec":     feature_matrix[idx],
            }
            for idx in top_idxs
            if sims[idx] >= 0
        ]
    return output


# ── Similarity reason ─────────────────────────────────────────────────────────

def similarity_reason(
    query_vec: np.ndarray,
    similar_vec: np.ndarray,
    feature_names: list[str],
) -> str:
    """Expression features checked first; up to 3 plain-English clauses returned."""
    diffs      = np.abs(query_vec - similar_vec)
    avg_vals   = (query_vec + similar_vec) / 2.0
    info_score = avg_vals / (diffs + 0.05)

    expr_idxs  = [i for i, f in enumerate(feature_names) if f in _EXPRESSION_FEATURES]
    other_idxs = [i for i, f in enumerate(feature_names) if f not in _EXPRESSION_FEATURES]
    ordered    = (
        sorted(expr_idxs,  key=lambda i: -float(info_score[i])) +
        sorted(other_idxs, key=lambda i: -float(info_score[i]))
    )

    reasons: list[str] = []
    for idx in ordered:
        if len(reasons) >= 3:
            break
        avg   = float(avg_vals[idx])
        diff  = float(diffs[idx])
        label = _FEATURE_LABELS.get(feature_names[idx], feature_names[idx])
        if avg < 0.05 and diff < 0.05:
            continue
        if avg >= 0.65 and diff <= 0.20:
            reasons.append(f"both show high {label}")
        elif avg >= 0.35 and diff <= 0.25:
            reasons.append(f"similar {label}")
        elif diff <= 0.15:
            reasons.append(f"comparable {label}")

    return "; ".join(reasons) if reasons else "similar multi-omics profile"


# ── Main entry point ──────────────────────────────────────────────────────────

def find_alternatives(
    gene: str,
    rank_results_df: pd.DataFrame,
    master_merged_path,
    top_k: int = 3,
    disease_filter: str | None = None,
    lineage_filter: str | None = None,
) -> dict[str, list[dict]]:
    """
    Find the top_k most similar alternative cell lines for each recommended line.

    If disease_filter / lineage_filter is supplied, candidates are restricted
    to matching cell lines (case-insensitive substring). Falls back to
    cross-disease alternatives when fewer than top_k same-context lines exist,
    marking fallback entries with a 'note' field.

    Returns {cellosaurus_id: [{cellosaurus_id, official_name, similarity_score,
                               shared_data_types, similarity_reason, note,
                               citations}, ...]}
    """
    from src.models.classical.ranker import rank as _rank_full
    from config.config import CELL_LINE_LOOKUP

    use_ctx = bool(disease_filter or lineage_filter)

    # Full expression scores for ALL cell lines (for the feature matrix)
    print(f"  [similarity] Scoring {gene} across all cell lines...")
    full_scores = _rank_full(gene, top_n=None)

    master_df = pd.read_parquet(master_merged_path)

    # Disease / lineage lookup
    lkp = pd.read_parquet(CELL_LINE_LOOKUP, columns=["cellosaurus_id", "disease", "lineage"])
    disease_map = dict(zip(lkp["cellosaurus_id"], lkp["disease"].fillna("")))
    lineage_map = dict(zip(lkp["cellosaurus_id"], lkp["lineage"].fillna("")))

    if use_ctx:
        df_lo = (disease_filter or "").lower()
        lf_lo = (lineage_filter or "").lower()

        def _in_ctx(cvcl: str) -> bool:
            d = disease_map.get(cvcl, "").lower()
            l = lineage_map.get(cvcl, "").lower()
            return (df_lo and df_lo in d) or (lf_lo and lf_lo in l)

        ctx_set = {c for c in lkp["cellosaurus_id"] if _in_ctx(c)}
    else:
        ctx_set = None

    matrix, cvcl_ids = build_feature_matrix(gene, full_scores, master_df)
    name_map  = dict(zip(master_df["cellosaurus_id"], master_df["official_name"]))
    id_to_idx = {c: i for i, c in enumerate(cvcl_ids)}

    query_ids = rank_results_df["cellosaurus_id"].tolist()
    fetch_k   = max(top_k * 10, 30) if use_ctx else top_k
    raw       = compute_similarity(query_ids, matrix, cvcl_ids, top_k=fetch_k)

    _cit_base = {
        "data_sources": [
            "Expression similarity computed using HPA [D1], DepMap [D2], GEO [D3], CCLE Proteomics [D4]",
            "Cell line identity verified via Cellosaurus [D5]",
        ],
        "method": (
            "Cosine similarity on 19-dimensional normalised "
            "multi-omics feature vector (expression features weighted 2-3×; "
            "coverage flags 0.3-0.5×)"
        ),
        "dataset_keys": {
            k: {"name": v["name"], "citation": v["citation"],
                "pmid": v["pmid"], "url": v["url"]}
            for k, v in DATASET_CITATIONS.items()
        },
    }

    def _annotate(s: dict, q_vec: np.ndarray, note: str | None) -> dict:
        scvcl = s["cellosaurus_id"]
        s_vec = s["_similar_vec"]
        shared = [
            label
            for feat, label in _DATA_TYPE_MAP.items()
            if feat in FEATURE_NAMES
            and float(q_vec[FEATURE_NAMES.index(feat)]) / FEATURE_WEIGHTS.get(feat, 1.0) > 0.5
            and float(s_vec[FEATURE_NAMES.index(feat)]) / FEATURE_WEIGHTS.get(feat, 1.0) > 0.5
        ]
        return {
            "cellosaurus_id":    scvcl,
            "official_name":     name_map.get(scvcl, scvcl),
            "similarity_score":  s["similarity_score"],
            "shared_data_types": shared,
            "similarity_reason": similarity_reason(q_vec, s_vec, FEATURE_NAMES),
            "note":              note,
            "citations": {
                **_cit_base,
                "cellosaurus_url": f"https://www.cellosaurus.org/{scvcl}",
            },
        }

    result: dict[str, list[dict]] = {}
    for qcvcl, similar_list in raw.items():
        if qcvcl not in id_to_idx:
            result[qcvcl] = []
            continue
        q_vec = matrix[id_to_idx[qcvcl]]

        if use_ctx:
            in_ctx  = [s for s in similar_list if s["cellosaurus_id"] in ctx_set]
            out_ctx = [s for s in similar_list if s["cellosaurus_id"] not in ctx_set]
            if len(in_ctx) >= top_k:
                chosen = [_annotate(s, q_vec, None) for s in in_ctx[:top_k]]
            else:
                fallback = (
                    "insufficient same-disease alternatives, "
                    "showing cross-disease similar lines"
                )
                chosen = [_annotate(s, q_vec, None) for s in in_ctx]
                chosen += [_annotate(s, q_vec, fallback) for s in out_ctx[:top_k - len(chosen)]]
        else:
            chosen = [_annotate(s, q_vec, None) for s in similar_list[:top_k]]

        result[qcvcl] = chosen

    return result

