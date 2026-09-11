from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import CELL_LINE_LOOKUP
from models.classical.pathway_scorer import score_pathway_activity
from models.classical.mutation_scorer import score_mutation_impact
from models.classical.copy_number_scorer import (
    AMPLIFICATION_DRIVEN_GENES,
    apply_amplification_copy_number_weight,
    score_copy_number,
)
from models.classical.scorer import (
    FIXED_WEIGHTS,
    _level_label_with_percentile,
    classify_gene,
    load_mappings,
    rank_sort,
    score_context,
    score_crispr_dependency,
    score_data_quality,
    score_protein_expression,
    score_rna_expression,
)

# Re-export for external consumers (evaluate, weights_learned, etc.)
__all__ = ["FIXED_WEIGHTS", "rank", "explain", "AMPLIFICATION_DRIVEN_GENES"]

# AMPLIFICATION_DRIVEN_GENES, AMPLIFICATION_COPY_NUMBER_WEIGHT, and
# apply_amplification_copy_number_weight() now live in copy_number_scorer.py
# (imported above) — single source of truth shared with weights_learned.py's
# _precompute_scores and cross_validated_evaluation.py's Config C, which
# each need the same gene set / weight / rescale construction and previously
# didn't know about it at all (see copy_number_scorer.py's comment).

_CACHED_LEARNED_WEIGHTS_BY_CLASS: dict | None = None


def _get_learned_weights_by_class() -> dict:
    """
    {tissue_specific, ubiquitous, loss_of_function} -> weights dict, cached
    once per process. Pathway activity's value as a signal genuinely varies
    by gene class (see weights_learned.optimise_weights_by_class), so
    production scoring selects the class-appropriate weights per query
    rather than one global set for every gene — otherwise the scoring
    transparency panel's "tuned independently per gene class" claim
    wouldn't actually be true of what's running.
    """
    global _CACHED_LEARNED_WEIGHTS_BY_CLASS
    if _CACHED_LEARNED_WEIGHTS_BY_CLASS is None:
        from models.classical.weights_learned import optimise_weights_by_class
        print("[ranker] Computing per-gene-class learned weights (one-time, cached)...")
        _CACHED_LEARNED_WEIGHTS_BY_CLASS = optimise_weights_by_class(include_mutation=True)
    return _CACHED_LEARNED_WEIGHTS_BY_CLASS


def _select_class_weights(weights_by_class: dict, gene_class: str) -> dict:
    """loss_of_function falls back to tissue_specific when the validation
    set has no dedicated LOF-only genes (see optimise_weights_by_class)."""
    return weights_by_class.get(gene_class) or weights_by_class["tissue_specific"]


def _quality_explanation(row) -> str:
    n_sources = int(row.get("n_sources", 0) or 0)
    parts = [f"{n_sources}/2 primary RNA sources available"]

    hpa = row.get("hpa_score")
    depmap = row.get("depmap_score")
    if pd.notna(hpa) and pd.notna(depmap):
        diff = abs(hpa - depmap)
        agreement = "agree closely" if diff < 0.15 else (
            "agree moderately" if diff < 0.35 else "disagree"
        )
        parts.append(f"HPA and DepMap {agreement}")

    return "; ".join(parts)


def _context_explanation(row, disease_filter, lineage_filter) -> str:
    """
    Explain context_score by checking which field actually produced the
    match, using the SAME conditions score_context() uses (see
    scorer.score_context's _dmatch / _lmatch) — not an independent
    judgment that could disagree with the real score.

    Disease matching has an exact (1.0) and partial/substring (0.5) tier.
    Lineage matching does NOT — score_context._lmatch scores any
    substring-or-exact lineage match as a full 1.0, with no 0.5 tier.
    So a lineage substring match is reported as an exact match here too:
    calling it "partial" would contradict the 1.0 score sitting right
    next to it.
    """
    context = row.get("context_score", 0) or 0
    disease_raw = row.get("disease")
    lineage_raw = row.get("lineage")
    disease = disease_raw if pd.notna(disease_raw) else ""
    lineage = lineage_raw if pd.notna(lineage_raw) else ""
    disease_l, lineage_l = disease.lower(), lineage.lower()
    df = disease_filter.lower().strip() if disease_filter else None
    lf = lineage_filter.lower().strip() if lineage_filter else None

    disease_exact   = bool(df) and disease_l == df
    disease_partial = bool(df) and not disease_exact and (df in disease_l or disease_l in df)
    lineage_match   = bool(lf) and (lf in lineage_l or lineage_l == lf)

    if disease_exact:
        return f"Exact match: disease '{disease}' matches your search"
    elif lineage_match:
        return f"Exact match: lineage '{lineage}' matches your search"
    elif disease_partial:
        return f"Partial match: disease '{disease}' contains your search term"
    elif context >= 1.0:
        return "Exact match to your search filter"
    elif context >= 0.5:
        return "Partial match to your search filter"
    else:
        return "No match to disease/tissue filter"


def rank(
    gene: str,
    disease_filter: str | None = None,
    lineage_filter: str | None = None,
    top_n: int | None = 10,
    weights: dict | None = None,
    use_learned_weights: bool = False,
    expression_threshold: bool = True,
    exclude_genes: list[str] | None = None,
    include_alternatives: bool = False,
    alternatives_top_k: int = 3,
    include_pathway: bool = True,
) -> pd.DataFrame:
    """
    Rank cell lines by suitability for studying a given gene.

    Final score formula:
        weights["rna"]     * rna_score
      + weights["protein"] * protein_score
      + weights["quality"] * quality_score
      + weights["context"] * context_score
      + weights.get("pathway", 0.0) * pathway_activity_score
      + geo_confirmation_bonus  (additive ±0.10, not weighted)

    pathway_activity_score: graph-enhanced signal (see
        pathway_scorer.score_pathway_activity) — fraction of the target
        gene's KEGG pathway neighbors (via Neo4j) that are ALSO expressed
        in this cell line, not just the target gene in isolation. Only
        meaningful for genes ingested into Neo4j (see
        models/graph/ingest.py); for any other gene this degrades to 0.0
        for every cell line rather than erroring. include_pathway=False
        skips computing it entirely (score forced to 0.0 for every row,
        which is rank-order-equivalent to omitting the term, since a
        uniform per-gene offset doesn't change relative ranking) — used by
        scripts/evaluation/compare_with_without_pathway.py for a
        controlled A/B comparison.

    exclude_genes: optional list of gene symbols whose expression should
        penalise the final score. For each excluded gene, an expression score
        is computed and applied as:
            final_score *= (1 - excluded_rna_score * 0.5)
        Columns added: excluded_{g}_score, excluded_{g}_flag (score > 0.5),
        exclusion_warning (True if any flag is set).

    dependency_score / dependency_percentile: CRISPR gene-essentiality signal
        (DepMap). Answers a different question from expression — how much a
        cell line NEEDS the gene to survive, not how much it makes of it —
        so it is attached as an informational column only and never enters
        the weighted final_score.

    hpa_evidence / depmap_evidence / protein_evidence: dicts of
        {label, score, percentile} — a qualitative Low/Medium/High/No data
        label alongside the exact underlying value, so the label never has
        to stand alone (see scorer._level_label_with_percentile).
        geo_evidence: {label: Confirms/Contradicts/No data, score, percentile
        (always None — GEO's signal is a confirmation bonus, not a rank)}.
        Surfaces per-source agreement or disagreement directly, rather than
        only the blended quality_score.

    vs_next_rank: a concrete, numeric explanation of why this cell line
        ranks above the next one down (see explain_rank_difference), or
        None for the last row.

    Returns columns:
        cellosaurus_id, official_name, final_score,
        rna_score, protein_score, quality_score, context_score,
        geo_confirmation, n_sources, disease, lineage,
        hpa_score, depmap_score, missing_data_flag,
        dependency_score, dependency_percentile,
        hpa_evidence, depmap_evidence, geo_evidence, protein_evidence,
        vs_next_rank,
        exclusion_warning [, excluded_{g}_score, excluded_{g}_flag ...]
    """
    gene_class = classify_gene(gene)
    if weights is None:
        if gene_class == "loss_of_function":
            # LOF genes rank on damaging-mutation status, not expression —
            # use the mutation-aware learned class weights (mutation=0.80 +
            # a rescaled expression baseline; see
            # weights_learned._apply_lof_mutation_weight) regardless of
            # use_learned_weights. A caller that passes an explicit `weights`
            # dict still gets exactly that.
            weights = _select_class_weights(_get_learned_weights_by_class(), "loss_of_function")
        elif use_learned_weights:
            weights = _select_class_weights(_get_learned_weights_by_class(), gene_class)
        else:
            weights = FIXED_WEIGHTS

    # Gated on the key, not on whether `weights` came from the block above —
    # eval scripts (full_evaluation.py, cross_validated_evaluation.py, ...)
    # build a per-class weights dict externally and always pass it in
    # explicitly (never weights=None), so gating this on "weights is None"
    # would make it fire for the production API path but silently never for
    # the eval scripts. A caller that already set "copy_number" explicitly
    # (e.g. the grid search that found 0.30) is respected as-is, not
    # re-overridden — this only fills it in when absent.
    if gene in AMPLIFICATION_DRIVEN_GENES and "copy_number" not in weights:
        weights = apply_amplification_copy_number_weight(weights)

    hpa_to_cvcl, ach_to_cvcl, gsm_to_cvcl = load_mappings()

    rna_df     = score_rna_expression(gene, hpa_to_cvcl, gsm_to_cvcl,
                                      gene_class=gene_class)
    protein_df = score_protein_expression(gene, ach_to_cvcl)

    all_cvcl = set(rna_df["cellosaurus_id"]) | set(protein_df["cellosaurus_id"])
    if not all_cvcl:
        print(f"No expression data found for gene: {gene}")
        return pd.DataFrame()

    # sorted(): a bare set → list is hash-seed-dependent, which would make
    # tie-breaking in the final sort_values below vary between processes.
    result = (
        pd.DataFrame({"cellosaurus_id": sorted(all_cvcl)})
        .merge(rna_df, on="cellosaurus_id", how="left")
        .merge(protein_df, on="cellosaurus_id", how="left")
    )

    # ── Per-source evidence (label + exact score + percentile) — computed
    # before the fillna(0.0) calls below, which would otherwise collapse
    # "no proteomics data" and "proteomics data present but low" into the
    # same 0.0 and mislabel both as "Low" instead of "No data".
    #
    # hpa_score / depmap_score / protein_score are ALREADY 0-1 percentile
    # ranks (see scorer._pct_rank) computed globally across every cell line
    # with data for this gene — not raw magnitudes. So the "percentile"
    # passed in is the score itself, not a re-rank of it. Re-ranking again
    # here (e.g. via .rank(pct=True) on `result`) would silently produce a
    # LOCAL percentile relative to whatever subset happens to be in
    # `result` at that point (and after top_n truncation, relative to just
    # the displayed top N) — a different, misleading number masquerading
    # as the real one.
    result["hpa_evidence"] = result["hpa_score"].apply(
        lambda s: _level_label_with_percentile(s, s)
    )
    result["depmap_evidence"] = result["depmap_score"].apply(
        lambda s: _level_label_with_percentile(s, s)
    )
    result["protein_evidence"] = result["protein_score"].apply(
        lambda s: _level_label_with_percentile(s, s)
    )
    result["geo_evidence"] = result["geo_confirmation"].apply(
        lambda x: {
            "label": ("Confirms" if x > 0 else "Contradicts" if x < 0 else "No data"),
            "score": round(float(x), 3) if pd.notna(x) else None,
            "percentile": None,
        }
    )

    # astype(float): a left-merge against an empty score frame (e.g. a gene
    # with no CCLE proteomics) yields an object-dtype all-NaN column;
    # .fillna(0.0) then leaves it object-dtype, which propagates into
    # final_score and breaks numeric ops like .describe() downstream.
    result["rna_score"]        = result["rna_score"].fillna(0.0).astype(float)
    result["protein_score"]    = result["protein_score"].fillna(0.0).astype(float)
    result["geo_confirmation"]  = result["geo_confirmation"].fillna(0.0).astype(float)

    quality_df = score_data_quality(all_cvcl, rna_df, protein_df)
    result = result.merge(quality_df, on="cellosaurus_id", how="left")
    result["quality_score"] = result["quality_score"].fillna(0.0).astype(float)

    context_df = score_context(all_cvcl, disease_filter, lineage_filter)
    result = result.merge(context_df, on="cellosaurus_id", how="left")
    result["context_score"] = result["context_score"].fillna(0.0).astype(float)

    # ── Plain-English explanations for the two composite scores — these
    # blend multiple signals into one number, which is exactly why they
    # need a sentence next to them instead of standing as a bare pill.
    result["quality_explanation"] = result.apply(_quality_explanation, axis=1)
    result["context_explanation"] = result.apply(
        lambda r: _context_explanation(r, disease_filter, lineage_filter), axis=1
    )

    # ── Pathway activity (graph-enhanced scoring) ────────────────────────────
    # include_pathway=False intentionally skips the Neo4j call entirely
    # (not just zeroing the weight) — used for the controlled with/without
    # comparison in scripts/evaluation/compare_with_without_pathway.py.
    if include_pathway:
        try:
            pathway_df = score_pathway_activity(
                gene, cell_line_ids=set(result["cellosaurus_id"])
            )
            result = result.merge(pathway_df, on="cellosaurus_id", how="left")
            result["pathway_activity_score"] = result["pathway_activity_score"].fillna(0.0)
            result["pathway_genes_expressed"] = result["pathway_genes_expressed"].fillna(0).astype(int)
            result["pathway_genes_total"] = result["pathway_genes_total"].fillna(0).astype(int)
        except Exception as exc:
            print(f"[ranker] Pathway scoring failed: {exc}")
            result["pathway_activity_score"] = 0.0
            result["pathway_genes_expressed"] = 0
            result["pathway_genes_total"] = 0
    else:
        result["pathway_activity_score"] = 0.0
        result["pathway_genes_expressed"] = 0
        result["pathway_genes_total"] = 0

    # ── Mutation impact (PRIMARY signal for loss-of-function genes) ──────────
    # LOF genes (BRCA1, TP53, ...) are chosen by researchers on damaging
    # mutation status, not expression level. For other classes it's a small
    # secondary signal at most (activating mutations reinforcing an
    # expression-based pick). See models/classical/mutation_scorer.py.
    try:
        mutation_df = score_mutation_impact(gene)
        result = result.merge(mutation_df, on="cellosaurus_id", how="left")
        result["mutation_impact_score"] = result["mutation_impact_score"].fillna(0.0).astype(float)
        result["mutation_detail"] = result["mutation_detail"].fillna("")
    except Exception as exc:
        print(f"[ranker] Mutation scoring failed: {exc}")
        result["mutation_impact_score"] = 0.0
        result["mutation_detail"] = ""

    # ── Copy-number amplification (PRIMARY signal for amplification-driven
    # genes) ──────────────────────────────────────────────────────────────
    # Opt-in per gene (AMPLIFICATION_DRIVEN_GENES), not gene_class-wide —
    # see models/classical/copy_number_scorer.py. No default weight is
    # baked in here; a caller must pass an explicit "copy_number" key in
    # `weights` to activate the branch below (same gating pattern as LOF's
    # "mutation" key). Until weights_learned.py grows a class/gene for this,
    # `use_learned_weights=True` callers (i.e. production) never populate
    # that key, so this is inert in production today.
    if gene in AMPLIFICATION_DRIVEN_GENES:
        try:
            cn_df = score_copy_number(gene)
            result = result.merge(cn_df, on="cellosaurus_id", how="left")
            result["copy_number_score"] = result["copy_number_score"].fillna(0.0).astype(float)
            result["copy_number_detail"] = result["copy_number_detail"].fillna("")
        except Exception as exc:
            print(f"[ranker] Copy-number scoring failed: {exc}")
            result["copy_number_score"] = 0.0
            result["copy_number_detail"] = ""

    if gene_class == "loss_of_function" and "mutation" in weights:
        # Mutation-primary LOF vector {mutation, rna, protein, quality,
        # context} (no pathway) — see weights_learned._apply_lof_mutation_weight.
        # Gated on the "mutation" key so an expression-only eval baseline
        # config passed for a LOF gene still runs the standard formula below
        # (unchanged from its historical behaviour).
        result["final_score"] = (
            weights["mutation"] * result["mutation_impact_score"]
            + weights.get("rna", 0.0)     * result["rna_score"]
            + weights.get("protein", 0.0) * result["protein_score"]
            + weights.get("quality", 0.0) * result["quality_score"]
            + weights.get("context", 0.0) * result["context_score"]
            + result["geo_confirmation"]   # additive, not weighted
        )
    else:
        # tissue_specific / ubiquitous (and LOF with a legacy expression-only
        # weights dict): unchanged expression formula, plus a small fixed
        # mutation bonus for tissue_specific only (an activating oncogene /
        # RTK mutation reinforcing the expression-based ranking).
        mutation_bonus = (
            0.05 * result["mutation_impact_score"]
            if gene_class == "tissue_specific" else 0.0
        )
        # Gated on both the gene AND an explicit "copy_number" weights key —
        # see the AMPLIFICATION_DRIVEN_GENES block above. No default weight;
        # a caller (e.g. the grid search) must pass one explicitly.
        copy_number_term = (
            weights["copy_number"] * result["copy_number_score"]
            if gene in AMPLIFICATION_DRIVEN_GENES and "copy_number" in weights
            else 0.0
        )
        result["final_score"] = (
            weights["rna"]     * result["rna_score"]
            + weights["protein"] * result["protein_score"]
            + weights["quality"] * result["quality_score"]
            + weights["context"] * result["context_score"]
            + weights.get("pathway", 0.0) * result["pathway_activity_score"]
            + copy_number_term
            + result["geo_confirmation"]   # additive, not weighted
            + mutation_bonus
        )
    result["final_score"] = result["final_score"].clip(0.0, 1.0)
    result["gene_class"] = classify_gene(gene)

    # ── Exclusion-gene penalties ──────────────────────────────────────────────
    if exclude_genes:
        for excl_gene in exclude_genes:
            excl_rna = score_rna_expression(excl_gene, hpa_to_cvcl, gsm_to_cvcl)
            cvcl_map = dict(zip(excl_rna["cellosaurus_id"], excl_rna["rna_score"]))
            score_col = f"excluded_{excl_gene}_score"
            flag_col  = f"excluded_{excl_gene}_flag"
            result[score_col] = result["cellosaurus_id"].map(
                lambda c, m=cvcl_map: float(m.get(c, 0.0))
            )
            result["final_score"] = (
                result["final_score"] * (1 - result[score_col] * 0.5)
            ).clip(0.0, 1.0)
            result[flag_col] = result[score_col] > 0.5
        result["exclusion_warning"] = result[
            [f"excluded_{g}_flag" for g in exclude_genes]
        ].any(axis=1)

    if disease_filter or lineage_filter:
        result = result[result["context_score"] > 0]

    lkp = pd.read_parquet(CELL_LINE_LOOKUP, columns=["cellosaurus_id", "official_name"])
    result = result.merge(lkp, on="cellosaurus_id", how="left")

    # Multi-key tie-break: final_score is clipped to 1.0, so mutated-LOF and
    # strong-expression lines saturate; mutation_impact_score / rna_score /
    # quality_score (unclipped) break those ties before cellosaurus_id does.
    result = rank_sort(result, "final_score")
    if top_n is not None:
        result = result.head(top_n)

    result["gene_class"] = gene_class

    # ── CRISPR dependency (essentiality) — additive column, NOT weighted ────
    # Kept separate from final_score: essentiality and expression answer
    # different questions, so they must not be blended into one number.
    crispr_df = score_crispr_dependency(gene)
    result = result.merge(crispr_df, on="cellosaurus_id", how="left")

    # ── Adjacent-rank comparisons — must run after final ordering/truncation
    # (sort_values + head(top_n) above) is locked in, since it compares each
    # row to the NEXT row in the returned order.
    result = add_rank_comparisons(result, weights)

    out_cols = [
        "cellosaurus_id", "official_name", "final_score",
        "rna_score", "protein_score", "quality_score", "context_score",
        "geo_confirmation", "n_sources", "disease", "lineage",
        "hpa_score", "depmap_score", "missing_data_flag", "gene_class",
        "dependency_score", "dependency_percentile",
        "hpa_evidence", "depmap_evidence", "geo_evidence", "protein_evidence",
        "vs_next_rank", "quality_explanation", "context_explanation",
        "pathway_activity_score", "pathway_genes_expressed", "pathway_genes_total",
        "mutation_impact_score", "mutation_detail",
    ]
    if gene in AMPLIFICATION_DRIVEN_GENES:
        out_cols += ["copy_number_score", "copy_number_detail"]
    if exclude_genes:
        out_cols.append("exclusion_warning")
        for g in exclude_genes:
            out_cols.extend([f"excluded_{g}_score", f"excluded_{g}_flag"])

    result = result[out_cols].reset_index(drop=True)

    if include_alternatives:
        # Lazy import avoids circular dependency (ranker ↔ similarity)
        from models.classical.similarity import find_alternatives
        from config import MASTER_MERGED
        alts = find_alternatives(
            gene, result, MASTER_MERGED,
            top_k=alternatives_top_k,
            disease_filter=disease_filter,
            lineage_filter=lineage_filter,
        )
        result["alternatives"] = result["cellosaurus_id"].map(
            lambda cvcl: alts.get(cvcl, [])
        )

    return result


def explain_rank_difference(row_a: pd.Series, row_b: pd.Series, weights: dict) -> str:
    """
    Given two ranked cell lines, explain in concrete numeric terms why
    row_a ranks above/below row_b: which weighted score component
    contributed the most to the gap, and by how much.
    """
    components = ["rna", "protein", "quality", "context", "pathway", "mutation"]
    _col = {"pathway": "pathway_activity_score", "mutation": "mutation_impact_score"}
    _wfallback = {"mutation": 0.0}  # not weighted outside loss_of_function
    diffs = []
    for comp in components:
        col = _col.get(comp, f"{comp}_score")
        val_a = row_a.get(col, 0) or 0
        val_b = row_b.get(col, 0) or 0
        # weights.get(...): learned weights don't carry a "pathway" key
        # (see rank()'s docstring) — same 0.10 fallback used in final_score.
        weighted_diff = weights.get(comp, _wfallback.get(comp, 0.10)) * (val_a - val_b)
        diffs.append((comp, val_a, val_b, weighted_diff))

    diffs.sort(key=lambda x: abs(x[3]), reverse=True)
    top_comp, val_a, val_b, weighted_diff = diffs[0]

    direction = "higher" if weighted_diff > 0 else "lower"
    name_a = row_a.get("official_name", "Cell line A")
    name_b = row_b.get("official_name", "Cell line B")

    return (
        f"{name_a} ranks {'above' if weighted_diff > 0 else 'below'} "
        f"{name_b} primarily due to {direction} {top_comp} "
        f"score ({val_a:.2f} vs {val_b:.2f}, contributing "
        f"{abs(weighted_diff):.3f} to the final score gap)."
    )


def add_rank_comparisons(result: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """
    For each cell line, add an explanation of why it ranks above the
    next-ranked cell line (None for the last row). Must be called after
    `result` is in its final sorted/truncated order — it compares each row
    to the literal next row, not by any score field.
    """
    result = result.reset_index(drop=True)
    comparisons = []
    for i in range(len(result)):
        if i < len(result) - 1:
            comparisons.append(
                explain_rank_difference(result.iloc[i], result.iloc[i + 1], weights)
            )
        else:
            comparisons.append(None)
    result["vs_next_rank"] = comparisons
    return result


def explain(row) -> str:
    """Generate a plain-English explanation for one ranked result row."""
    name = (
        row.get("official_name")
        if pd.notna(row.get("official_name"))
        else row.get("cellosaurus_id", "Unknown")
    )

    gene       = row.get("gene") or ""
    gene_class = row.get("gene_class") or "tissue_specific"
    rna        = float(row.get("rna_score") or 0)
    n          = int(row.get("n_sources") or 0)
    geo_c      = float(row.get("geo_confirmation") or 0)

    geo_str = ""
    if geo_c > 0:
        geo_str = " GEO independently confirms expression."
    elif geo_c < 0:
        geo_str = " Note: GEO data contradicts primary RNA sources."

    disease = row.get("disease") or ""
    lineage = row.get("lineage") or ""
    context = float(row.get("context_score") or 0)
    context_str = ""
    if context >= 1.0:
        label = disease if pd.notna(disease) and disease else lineage
        context_str = f" {label} exactly matches query."
    elif context >= 0.5:
        label = disease if pd.notna(disease) and disease else lineage
        context_str = f" {label} partially matches query."

    confidence_pct = int(round(float(row.get("final_score") or 0) * 100))

    if gene_class == "ubiquitous":
        # RNA score represents cross-source consistency, not expression level
        if rna > 0.85:
            cons_desc = "very high"
        elif rna > 0.65:
            cons_desc = "good"
        elif rna > 0.4:
            cons_desc = "moderate"
        else:
            cons_desc = "low"
        core = (
            f"{name} shows {cons_desc} cross-source consistency for this "
            f"broadly-expressed gene (consistency score {rna:.2f} across "
            f"{n} of 2 primary RNA sources)."
        )
    else:
        # tissue_specific and loss_of_function: expression-level description
        if rna > 0.8:
            expr_desc = "strongly"
        elif rna > 0.5:
            expr_desc = "moderately"
        elif rna > 0:
            expr_desc = "weakly"
        else:
            expr_desc = "not detectably (no primary RNA data)"

        hpa_s = row.get("hpa_score")
        dep_s = row.get("depmap_score")
        avail = [float(v) for v in [hpa_s, dep_s] if pd.notna(v)]
        consistency = "high" if len(avail) < 2 else (
            "high" if abs(avail[0] - avail[1]) < 0.15 else
            "moderate" if abs(avail[0] - avail[1]) < 0.35 else "low"
        )
        core = (
            f"{name} expresses the target gene {expr_desc} (RNA score {rna:.2f}) "
            f"confirmed across {n} of 2 primary RNA sources with {consistency} consistency."
        )

    lof_note = ""
    if gene_class == "loss_of_function":
        gene_label = gene if gene else "this gene"
        lof_note = (
            f" Note: {gene_label} is typically studied via loss-of-function. "
            f"These results show lines with HIGH expression (useful as controls). "
            f"Lines with known {gene_label} mutations may be more relevant for LOF studies."
        )

    return f"{core}{geo_str}{context_str}{lof_note} Overall confidence: {confidence_pct}%"
