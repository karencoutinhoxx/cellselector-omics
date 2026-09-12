import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from models.classical.scorer import classify_gene, load_mappings, score_rna_expression
from models.graph.neo4j_client import run_query

# ─────────────────────────────────────────────────────────────────────────────
# Graph-enhanced scoring: "of the genes that share a KEGG pathway with the
# target gene, how many are ALSO expressed in this cell line?" A cell line
# where the whole pathway is active — not just the target gene in isolation —
# is a biologically stronger experimental model. This is the one component
# of final_score that genuinely depends on the Neo4j graph rather than a
# flat table lookup.
#
# NOTE ON COVERAGE: pathway neighbors only exist in Neo4j for the 25 genes
# ingested by models/graph/ingest.py (VALIDATION_SET) plus whichever genes
# turned up as THEIR pathway neighbors. For any other gene, the Cypher query
# below returns no neighbors and this degrades gracefully to
# pathway_activity_score = 0.0 for every cell line — safe (never breaks
# ranking), but the 10% final_score weight is a no-op outside that set.
# ─────────────────────────────────────────────────────────────────────────────

_PATHWAY_SCORE_CACHE: dict = {}


def score_pathway_activity(
    gene: str,
    cell_line_ids: set[str] | None = None,
    expression_threshold: float = 0.3,
) -> pd.DataFrame:
    """
    For each cell line, compute the fraction of the target gene's KEGG
    pathway neighbors that are also expressed above a threshold in that
    cell line.

    Uses the Neo4j graph to find pathway neighbors (a genuine graph query),
    then checks expression data for each neighbor gene.

    cell_line_ids is accepted for call-site convenience (ranker.py passes
    the current query's candidate set) but does NOT scope the cached
    computation — the cache key is `gene` alone, so the cached result must
    stay valid for ANY caller regardless of which candidate set they pass.
    Computing a cell_line_ids-filtered result and caching it under just
    `gene` would silently return a stale, wrongly-scoped result to a later
    call for the same gene with a different candidate set (e.g. a
    different disease_filter). The caller's left-merge back onto its own
    `result` naturally drops any extra rows anyway, so filtering here would
    have been redundant even if it were safe.

    Returns: cellosaurus_id, pathway_activity_score (0-1),
             pathway_genes_expressed (count), pathway_genes_total (count)
    """
    if gene in _PATHWAY_SCORE_CACHE:
        return _PATHWAY_SCORE_CACHE[gene]

    # Step 1: Get pathway neighbor genes from Neo4j.
    # Ingestion stores (Pathway)-[:CONTAINS]->(Gene) — the arrow points
    # FROM the pathway TO its member genes, not the other way around.
    try:
        neighbors = run_query(
            """
            MATCH (g:Gene {symbol: $gene})-[:MEMBER_OF]->(p:Pathway)
                  -[:CONTAINS]->(neighbor:Gene)
            WHERE neighbor.symbol <> $gene
            RETURN DISTINCT neighbor.symbol AS gene_symbol
            """,
            {"gene": gene},
        )
        neighbor_genes = [r["gene_symbol"] for r in neighbors]
    except Exception as exc:
        print(f"[pathway_scorer] Neo4j query failed: {exc}")
        neighbor_genes = []

    if not neighbor_genes:
        empty = pd.DataFrame({
            "cellosaurus_id":          pd.Series(dtype="object"),
            "pathway_activity_score":  pd.Series(dtype="float64"),
            "pathway_genes_expressed": pd.Series(dtype="int64"),
            "pathway_genes_total":     pd.Series(dtype="int64"),
        })
        _PATHWAY_SCORE_CACHE[gene] = empty
        return empty

    # Step 2: For each neighbor gene (capped at 50 — matches the raised
    # ingest.py max_neighbor_genes cap, was 20 to match the old max_pathways
    # x max_neighbor_genes-starved graph), get expression scores across all
    # cell lines and tally how many neighbors clear the threshold per cell
    # line. Raised as a controlled test: isolates whether the naive scorer's
    # earlier "pathway=0.00 optimal" verdict was neighbor-starved (this cap
    # was quietly re-truncating an already-corrected, already-larger
    # neighbor_genes list) versus a structural problem with the
    # presence/absence mechanism itself, now that both pathway relevance
    # (STEP A) and neighbor sample size are no longer the limiting factor.
    hpa_to_cvcl, ach_to_cvcl, gsm_to_cvcl = load_mappings()

    attempted = neighbor_genes[:50]
    total_neighbors = len(attempted)  # decremented on failure below
    all_expression: dict[str, int] = {}  # cvcl -> count of expressed neighbors

    for ngene in attempted:
        try:
            ngene_class = classify_gene(ngene)
            rna = score_rna_expression(ngene, hpa_to_cvcl, gsm_to_cvcl, gene_class=ngene_class)
            expressed_lines = set(
                rna[rna["rna_score"] >= expression_threshold]["cellosaurus_id"]
            )
            for cvcl in expressed_lines:
                all_expression[cvcl] = all_expression.get(cvcl, 0) + 1
        except Exception:
            total_neighbors -= 1
            continue

    if total_neighbors <= 0:
        total_neighbors = 1  # avoid division by zero

    rows = [
        {
            "cellosaurus_id":          cvcl,
            "pathway_activity_score":  count / total_neighbors,
            "pathway_genes_expressed": count,
            "pathway_genes_total":     total_neighbors,
        }
        for cvcl, count in all_expression.items()
    ]

    result = pd.DataFrame(rows)
    _PATHWAY_SCORE_CACHE[gene] = result
    return result


if __name__ == "__main__":
    print("Pathway activity self-test (EGFR)...")
    df = score_pathway_activity("EGFR")
    print(f"{len(df)} cell lines scored")
    if len(df) > 0:
        print(df.sort_values("pathway_activity_score", ascending=False).head(10).to_string(index=False))
