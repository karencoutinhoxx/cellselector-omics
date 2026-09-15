import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import PARQUET_DIR
from src.models.agentic.pathways import (
    get_cached_pathway_genes,
    get_kegg_pathways,
    resolve_gene_symbol,
)
from src.models.classical.scorer import _pct_rank

# ─────────────────────────────────────────────────────────────────────────────
# Single-sample GSEA (ssGSEA) pathway enrichment — replaces the naive
# presence/absence pathway_activity_score for the two gene classes where
# today's investigation showed that scorer fails even with clean, complete
# pathway data (tissue_specific, ubiquitous — see pathway_grid_search.py:
# pathway=0.00 remained optimal after both the KEGG relevance filter AND a
# full, un-truncated neighbor sample, ruling out data quality as the cause).
#
# Reference: Barbie et al. 2009, Nature 462:108-112, Online Methods
# ("Systematic RNA interference reveals that oncogenic KRAS-driven cancers
# require TBK1") — the ssGSEA statistic there is the GSEA running-sum
# enrichment score computed against ONE sample's within-sample gene rank
# (not a phenotype-labeled group of samples like classic GSEA), SUMMED
# across the walk rather than taking the max |deviation| (that max-deviation
# variant is classic GSEA's ES; ssGSEA's is the sum — see also GSVA's
# ssGSEA mode, which implements the same construction:
# https://bioconductor.org/packages/release/bioc/vignettes/GSVA/inst/doc/GSVA.html).
# ─────────────────────────────────────────────────────────────────────────────

_RANK_MATRIX: pd.DataFrame | None = None
_VALID_SYMBOLS: set[str] | None = None
_SSGSEA_CACHE: dict[str, pd.DataFrame] = {}

_WEIGHT_EXPONENT = 0.25  # Barbie et al. 2009 default


def compute_ssgsea_score(
    gene_expression_ranks: pd.Series,
    pathway_genes: set[str],
    weight_exponent: float = _WEIGHT_EXPONENT,
) -> float:
    """
    Single-sample GSEA-style enrichment score for one gene set (pathway) in
    one cell line, per Barbie et al. 2009 (see module docstring).

    gene_expression_ranks: ALL genes for ONE cell line, indexed by gene
    symbol, values = expression rank WITHIN that cell line, ASCENDING (rank
    1 = lowest-expressed gene, rank N = highest-expressed gene — the same
    convention pandas.Series.rank() produces by default). The walk below
    sorts this DESCENDING internally (walks from most-highly-expressed gene
    down to least), which is the standard GSEA/ssGSEA running-sum direction.

    Algorithm:
    1. Sort genes by rank descending (highest expression first).
    2. Walk down the list, maintaining a running sum:
       - Gene IN pathway_genes: running sum += rank^weight_exponent,
         normalized so all such increases across the full pathway gene set
         sum to 1.0 (hit_norm below).
       - Gene NOT in pathway_genes: running sum -= 1/(N - n_hits), a fixed
         step normalized by the number of non-pathway genes, so all such
         decreases also sum to 1.0 in magnitude.
    3. ssGSEA variant (per Barbie et al., as opposed to classic GSEA's max
       |deviation| statistic): the enrichment score is the SUM of the
       running-sum trajectory across the whole walk, not its peak.

    Returns the raw (unnormalized) enrichment score — can be any real
    number, roughly scaled by N (the ranking's length) and by how
    systematically pathway_genes cluster toward high rank. NOT 0-1 range;
    see score_pathway_ssgsea() for the 0-1 normalization applied on top
    (percentile rank across cell lines, matching every other scorer's
    convention in this project — there's no natural theoretical min/max to
    normalize against directly).

    Degenerate cases (no pathway genes present in the ranking, or the
    pathway is the entire ranking) return 0.0 — no enrichment is definable.
    """
    ranked = gene_expression_ranks.sort_values(ascending=False)
    in_set = ranked.index.isin(pathway_genes)
    n = len(ranked)
    n_hits = int(in_set.sum())
    if n_hits == 0 or n_hits == n:
        return 0.0

    hit_weights = np.where(in_set, ranked.values.astype(float) ** weight_exponent, 0.0)
    hit_norm = hit_weights.sum()
    if hit_norm == 0:
        return 0.0
    miss_step = 1.0 / (n - n_hits)

    running = np.where(in_set, hit_weights / hit_norm, -miss_step)
    cumulative = np.cumsum(running)
    return float(cumulative.sum())


def _load_rank_matrix() -> pd.DataFrame:
    """
    Full gene-expression rank matrix: every gene, every cell line, ranked
    WITHIN each cell line (ascending — see compute_ssgsea_score's
    docstring). Computed once per process (~30s: 0.5s load + ~29s the
    groupby-rank, per the feasibility check) and cached — this is the
    gene-INDEPENDENT cost that makes ssGSEA tractable per-target-gene;
    every score_pathway_ssgsea() call after the first reuses this.

    Source: DepMap 23Q4 expression (gene_expr_depmap_preprocessed.parquet)
    — same release as COPY_NUMBER_FILE / the rest of the "primary DepMap"
    data this project uses, and the only one of the project's expression
    sources with a direct cellosaurus_id column and genuine full-transcriptome
    coverage per cell line (53,529 genes x 1,399 cell lines) — HPA's parquet
    lacks a direct cellosaurus_id column and would need the extra
    original_id mapping hop.
    """
    global _RANK_MATRIX, _VALID_SYMBOLS
    if _RANK_MATRIX is not None:
        return _RANK_MATRIX

    df = pd.read_parquet(
        PARQUET_DIR / "gene_expr_depmap_preprocessed.parquet",
        columns=["gene_symbol", "expression_value", "cellosaurus_id"],
    )
    df = df.dropna(subset=["cellosaurus_id"])
    df["rank"] = df.groupby("cellosaurus_id")["expression_value"].rank(method="average")

    _VALID_SYMBOLS = set(df["gene_symbol"].unique())
    _RANK_MATRIX = df
    return df


def _resolve_pathway_genes(raw_genes: set[str]) -> tuple[set[str], dict]:
    """
    Resolve a pathway's gene symbols against the expression parquet's true
    gene_symbol universe, via pathways.resolve_gene_symbol's KEGG-alias
    fallback (fixes cases like "P3R3URF-PIK3R3" not existing in the parquet
    even though its alias "PIK3R3" does — see pathways.py's
    _ensure_gene_symbols / resolve_gene_symbol).

    Returns (resolved_symbols, stats) where stats reports how many genes
    resolved directly, via alias fallback, or not at all — this is the
    effective gene-set size ssGSEA actually scores against, which can be
    smaller than the raw pathway membership count.
    """
    _load_rank_matrix()  # ensures _VALID_SYMBOLS is populated
    valid = _VALID_SYMBOLS or set()

    resolved: set[str] = set()
    direct, via_alias, unresolved = 0, 0, []
    for g in raw_genes:
        if g in valid:
            resolved.add(g)
            direct += 1
            continue
        alt = resolve_gene_symbol(g, valid)
        if alt is not None:
            resolved.add(alt)
            via_alias += 1
        else:
            unresolved.append(g)

    stats = {
        "raw_count": len(raw_genes),
        "direct": direct,
        "via_alias": via_alias,
        "unresolved": len(unresolved),
        "unresolved_genes": unresolved,
        "resolved_count": len(resolved),
    }
    return resolved, stats


def _batch_raw_es(pathway_genes: set[str], df: pd.DataFrame) -> pd.Series:
    """
    Vectorized equivalent of calling compute_ssgsea_score() once per cell
    line for ONE gene set — same math (see that function's docstring),
    computed across the whole rank matrix at once rather than in a
    per-cell-line Python loop, which is what keeps this fast. Returns raw
    (unnormalized) ES indexed by cellosaurus_id.
    """
    df = df.copy()
    df["_in_set"] = df["gene_symbol"].isin(pathway_genes)
    df = df.sort_values(["cellosaurus_id", "rank"], ascending=[True, False])
    df["_hit_weight"] = np.where(df["_in_set"], df["rank"].astype(float) ** _WEIGHT_EXPONENT, 0.0)

    per_line = df.groupby("cellosaurus_id")
    hit_norm = per_line["_hit_weight"].transform("sum")
    n_total = per_line["_in_set"].transform("size")
    n_hits = per_line["_in_set"].transform("sum")
    n_miss = (n_total - n_hits).replace(0, 1)  # guard div-by-zero; degenerate lines get 0 contribution anyway
    miss_step = 1.0 / n_miss

    hit_norm_safe = hit_norm.replace(0, 1)
    running = np.where(df["_in_set"], df["_hit_weight"] / hit_norm_safe, -miss_step)
    df["_running"] = running
    df["_cumulative"] = df.groupby("cellosaurus_id")["_running"].cumsum()

    raw_es = df.groupby("cellosaurus_id")["_cumulative"].sum()
    # Degenerate cell lines (0 hits, or hit_norm was 0) get raw ES forced
    # to 0.0 — same degenerate-case handling as compute_ssgsea_score.
    # Aggregated fresh here (not via the earlier row-broadcast .transform
    # Series) so the index lines up directly with raw_es's grouped index.
    hits_per_line = per_line["_hit_weight"].sum()
    n_hits_per_line = per_line["_in_set"].sum()
    degenerate = (hits_per_line == 0) | (n_hits_per_line == 0)
    return raw_es.where(~degenerate, 0.0)


def _per_pathway_percentiles(gene: str) -> tuple[dict[str, pd.Series], dict[str, int], dict]:
    """
    Shared groundwork for every pathway-aggregation strategy tested today
    (max-across-pathways, specificity-weighted, ...): per-pathway percentile
    ES series + each pathway's resolved gene-set size (needed for
    specificity weighting: w_i = 1/S_i) + resolve stats. Extracted out of
    score_pathway_ssgsea so alternate combination strategies can reuse the
    same (expensive: rank matrix + per-pathway ssGSEA walks) computation
    without re-deriving it or hitting KEGG/the rank matrix twice.

    Returns ({pathway_name: percentile Series indexed by cellosaurus_id},
             {pathway_name: resolved gene-set size}, resolve_stats_by_pathway)
    """
    pathways = get_kegg_pathways(gene)
    if not pathways:
        return {}, {}, {}

    df = _load_rank_matrix()
    resolve_stats_by_pathway: dict[str, dict] = {}
    per_pathway_pct: dict[str, pd.Series] = {}
    per_pathway_size: dict[str, int] = {}

    for p in pathways:
        raw_genes = set(get_cached_pathway_genes(p["id"]))
        raw_genes.discard(gene)
        if not raw_genes:
            continue
        pathway_genes, stats = _resolve_pathway_genes(raw_genes)
        resolve_stats_by_pathway[p["name"]] = stats
        if not pathway_genes:
            continue
        raw_es = _batch_raw_es(pathway_genes, df)
        per_pathway_pct[p["name"]] = pd.Series(_pct_rank(raw_es), index=raw_es.index)
        per_pathway_size[p["name"]] = len(pathway_genes)

    return per_pathway_pct, per_pathway_size, resolve_stats_by_pathway


def score_pathway_ssgsea(gene: str) -> pd.DataFrame:
    """
    ssGSEA enrichment score of `gene`'s KEGG pathways, scored PER PATHWAY
    independently (not unioned into one combined gene set), taking the MAX
    score across all of the gene's relevance-filtered pathways per cell
    line.

    Union-then-score was tried first and rejected: EGFR's combined gene set
    across its 22 relevant pathways is ~1,900 genes — large enough that the
    enrichment statistic measures "broadly active across signal
    transduction" rather than anything EGFR-specific, and known
    EGFR-mutant lines (HCC827, NCI-H3255) landed at the 41st/42nd
    percentile, not near the top. Scoring each pathway on its own terms and
    taking the best one is closer to how ssGSEA/GSVA is normally used (one
    coherent gene set per test) and let a cell line's genuine strength in
    ANY one of the gene's pathways stand on its own, rather than being
    diluted by averaging against every other pathway's noise.

    Each pathway's raw ES is percentile-ranked INDEPENDENTLY before the max
    is taken (not a max of raw ES values, which aren't on a comparable
    scale across pathways of very different sizes) — see _batch_raw_es.

    Returns: DataFrame [cellosaurus_id, ssgsea_score, ssgsea_detail] —
    ssgsea_score is the max per-pathway percentile rank (0-1); ssgsea_detail
    names which pathway produced that max, for interpretability (same
    "detail" convention as mutation_detail / copy_number_detail).

    Cached per gene (same pattern as score_mutation_impact /
    score_copy_number). The expensive part (the rank matrix) is cached
    ONCE across all genes via _load_rank_matrix.
    """
    if gene in _SSGSEA_CACHE:
        return _SSGSEA_CACHE[gene]

    empty = pd.DataFrame({
        "cellosaurus_id": pd.Series(dtype="object"),
        "ssgsea_score":   pd.Series(dtype="float64"),
        "ssgsea_detail":  pd.Series(dtype="object"),
    })

    n_pathways_total = len(get_kegg_pathways(gene))  # cached by pathways.py — cheap on a repeat call
    per_pathway_pct, per_pathway_size, resolve_stats_by_pathway = _per_pathway_percentiles(gene)
    if not per_pathway_pct:
        _SSGSEA_CACHE[gene] = empty
        return empty

    pct_df = pd.DataFrame(per_pathway_pct)  # rows=cellosaurus_id, cols=pathway name
    best_score = pct_df.max(axis=1)
    best_pathway = pct_df.idxmax(axis=1)

    result = pd.DataFrame({
        "cellosaurus_id": pct_df.index,
        "ssgsea_score": best_score.values,
        "ssgsea_detail": best_pathway.values,
    }).reset_index(drop=True)

    # Aggregate resolve stats across all of the gene's pathways, for STEP 3
    # -style reporting (inspectable via result.attrs).
    total_raw = sum(s["raw_count"] for s in resolve_stats_by_pathway.values())
    total_resolved = sum(s["resolved_count"] for s in resolve_stats_by_pathway.values())
    result.attrs["resolve_stats"] = {
        "per_pathway": resolve_stats_by_pathway,
        "n_pathways_scored": len(per_pathway_pct),
        "n_pathways_total": n_pathways_total,
        "total_raw_gene_mentions": total_raw,
        "total_resolved_gene_mentions": total_resolved,
    }
    _SSGSEA_CACHE[gene] = result
    return result

