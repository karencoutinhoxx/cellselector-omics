import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import CELL_LINE_LOOKUP
from src.models.graph.neo4j_client import run_query

# ─────────────────────────────────────────────────────────────────────────────
# Cypher-backed replacements for the old NetworkX query functions
# (models/graph/knowledge_graph.py, now retired). Neo4j holds the graph
# structure (gene/pathway/cell_line identities + relationships); cell line
# display names still come from the existing parquet spine, same as
# everywhere else in this app.
# ─────────────────────────────────────────────────────────────────────────────


def _cell_line_names() -> dict[str, str]:
    lkp = pd.read_parquet(CELL_LINE_LOOKUP, columns=["cellosaurus_id", "official_name"])
    return dict(zip(lkp["cellosaurus_id"], lkp["official_name"]))


def query_pathway_neighbor_genes(gene: str) -> list[dict]:
    """
    Genuine 2-hop Cypher query: genes sharing a pathway with the target gene.
    """
    query = """
    MATCH (g:Gene {symbol: $gene})-[:MEMBER_OF]->(p:Pathway)
          -[:CONTAINS]->(neighbor:Gene)
    WHERE neighbor.symbol <> $gene
    WITH neighbor, collect(DISTINCT {
        pathway_id: p.pathway_id,
        pathway_name: p.name
    }) AS shared_pathways
    RETURN neighbor.symbol AS gene,
           shared_pathways,
           size(shared_pathways) AS pathway_count
    ORDER BY pathway_count DESC
    LIMIT 50
    """
    try:
        results = run_query(query, {"gene": gene})
    except Exception as exc:
        print(f"[queries] Neo4j unavailable (query_pathway_neighbor_genes): {exc}")
        return []
    return [
        {"gene": r["gene"], "shared_pathways": r["shared_pathways"]}
        for r in results
    ]


def query_best_cell_lines_via_pathway(
    gene: str,
    disease_filter: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    """
    Multi-hop Cypher query: best cell lines connected to gene directly OR
    via pathway-neighbor genes, optionally restricted to cell lines whose
    disease matches disease_filter (case-insensitive substring, matching
    score_context()'s convention elsewhere in the app).

    The two OPTIONAL MATCHes are independent (target's own cell lines vs.
    neighbor genes' cell lines) but share the same MATCH'd g, so Cypher
    cross-joins their rows — one row per (c1, c2) combination. Each WHERE
    is placed directly after its own OPTIONAL MATCH so the disease filter
    (and the neighbor self-exclusion) apply within that match only; a
    single WHERE after both matches would filter the cross-joined row as a
    whole, which could let a non-matching c1 survive by being paired in
    some row with a matching c2 (or vice versa).
    """
    query = """
    MATCH (g:Gene {symbol: $gene})
    OPTIONAL MATCH (g)-[r1:EXPRESSED_IN]->(c1:CellLine)
    WHERE $disease_filter IS NULL
       OR toLower(c1.disease) CONTAINS toLower($disease_filter)
    OPTIONAL MATCH (g)-[:MEMBER_OF]->(:Pathway)
                    -[:CONTAINS]->(neighbor:Gene)
                    -[r2:EXPRESSED_IN]->(c2:CellLine)
    WHERE neighbor.symbol <> $gene
      AND ($disease_filter IS NULL
           OR toLower(c2.disease) CONTAINS toLower($disease_filter))
    WITH
        collect(DISTINCT {
            cellosaurus_id: c1.cellosaurus_id,
            gene: $gene,
            is_target: true,
            score: r1.score
        }) AS direct,
        collect(DISTINCT {
            cellosaurus_id: c2.cellosaurus_id,
            gene: neighbor.symbol,
            is_target: false,
            score: r2.score
        }) AS pathway_connected
    RETURN direct + pathway_connected AS all_results
    """
    try:
        results = run_query(query, {"gene": gene, "disease_filter": disease_filter})
    except Exception as exc:
        print(f"[queries] Neo4j unavailable (query_best_cell_lines_via_pathway): {exc}")
        return []
    if not results:
        return []

    rows = [r for r in results[0]["all_results"] if r.get("cellosaurus_id") is not None]

    # Group by cell line (a cell line can be reached via several connecting
    # genes) — matches the shape the old NetworkX version returned, which
    # the frontend is already built against: official_name + a
    # connecting_genes list per cell line, not one flattened row per edge.
    by_cell: dict[str, list[dict]] = {}
    for r in rows:
        by_cell.setdefault(r["cellosaurus_id"], []).append({
            "gene":             r["gene"],
            "is_target":        r["is_target"],
            "expression_score": r.get("score") or 0,
        })

    names = _cell_line_names()
    grouped = []
    for cvcl, connecting_genes in by_cell.items():
        connecting_genes.sort(key=lambda g: (not g["is_target"], -g["expression_score"]))
        grouped.append({
            "cellosaurus_id":   cvcl,
            "official_name":    names.get(cvcl, cvcl),
            "connecting_genes": connecting_genes,
            "max_score":        max(g["expression_score"] for g in connecting_genes),
        })

    grouped.sort(key=lambda x: x["max_score"], reverse=True)
    return grouped[:top_k]


def get_mutations_for_gene(gene: str, min_impact: float = 0.0) -> list[dict]:
    """
    All HAS_MUTATION edges from `gene` (see ingest.ingest_mutation_edges),
    highest impact first. Used as a graph-backed path for the mutation
    scorer and for agentic rationale generation.

    Returns [{cellosaurus_id, official_name, protein_change, hotspot,
              likely_lof, clinical_significance, impact_score}, ...].
    """
    query = """
    MATCH (g:Gene {symbol: $gene})-[m:HAS_MUTATION]->(c:CellLine)
    WHERE m.impact_score >= $min_impact
    RETURN c.cellosaurus_id       AS cellosaurus_id,
           m.protein_change        AS protein_change,
           m.hotspot               AS hotspot,
           m.likely_lof            AS likely_lof,
           m.clinical_significance AS clinical_significance,
           m.impact_score          AS impact_score
    ORDER BY m.impact_score DESC
    """
    try:
        results = run_query(query, {"gene": gene, "min_impact": min_impact})
    except Exception as exc:
        print(f"[queries] Neo4j unavailable (get_mutations_for_gene): {exc}")
        return []
    names = _cell_line_names()
    for r in results:
        r["official_name"] = names.get(r["cellosaurus_id"], r["cellosaurus_id"])
    return results


def find_cell_lines_by_mutation_and_lineage(
    gene: str,
    disease_substring: str,
    min_impact: float = 0.5,
) -> list[dict]:
    """
    Cross-dataset multi-hop query: cell lines with a damaging variant in
    `gene` (DepMap mutation data, via HAS_MUTATION edges) AND a disease
    lineage matching `disease_substring` (Cellosaurus/nomenclature metadata,
    via the CellLine.disease property set by enrich_cell_line_lineage).

    Mutation evidence and lineage metadata coexist today only in the flat
    parquet spine; here they are one traversal over the same integrated
    graph that also holds pathway (MEMBER_OF/CONTAINS) and expression
    (EXPRESSED_IN) structure — the "coherent cross-dataset analysis" the
    brief's Objective #1 calls for.

    Returns [{cellosaurus_id, official_name, disease, lineage,
              protein_change, impact_score}, ...], highest impact first.
    """
    query = """
    MATCH (g:Gene {symbol: $gene})-[m:HAS_MUTATION]->(c:CellLine)
    WHERE m.impact_score >= $min_impact
      AND c.disease IS NOT NULL
      AND toLower(c.disease) CONTAINS toLower($disease_substring)
    RETURN c.cellosaurus_id       AS cellosaurus_id,
           c.disease              AS disease,
           c.lineage              AS lineage,
           m.protein_change       AS protein_change,
           m.impact_score         AS impact_score
    ORDER BY m.impact_score DESC, c.cellosaurus_id
    """
    try:
        results = run_query(query, {
            "gene": gene,
            "disease_substring": disease_substring,
            "min_impact": min_impact,
        })
    except Exception as exc:
        print(f"[queries] Neo4j unavailable (find_cell_lines_by_mutation_and_lineage): {exc}")
        return []
    names = _cell_line_names()
    for r in results:
        r["official_name"] = names.get(r["cellosaurus_id"], r["cellosaurus_id"])
    return results


def find_receptor_genes_in_pathway(gene: str) -> list[dict]:
    """
    Genes sharing a KEGG pathway with `gene`, filtered to those carrying a
    receptor / marker / tumor-suppressor / oncogene role — pathway topology
    (MEMBER_OF/CONTAINS) combined with gene-role metadata (Gene.role, set by
    enrich_gene_roles) in one traversal.

    Returns [{gene_symbol, role, pathway_name}, ...].
    """
    query = """
    MATCH (g:Gene {symbol: $gene})-[:MEMBER_OF]->(p:Pathway)
          -[:CONTAINS]->(neighbor:Gene)
    WHERE neighbor.role IS NOT NULL
      AND neighbor.symbol <> $gene
    RETURN DISTINCT neighbor.symbol AS gene_symbol,
           neighbor.role            AS role,
           p.name                   AS pathway_name
    ORDER BY gene_symbol
    """
    try:
        return run_query(query, {"gene": gene})
    except Exception as exc:
        print(f"[queries] Neo4j unavailable (find_receptor_genes_in_pathway): {exc}")
        return []


def graph_to_json(gene: str) -> dict:
    """
    Return the full subgraph around a gene as nodes+edges JSON for frontend
    visualization.

    Run as three separately-scoped queries rather than one long chain of
    OPTIONAL MATCHes: chaining them (target's cell lines, target's
    pathways, pathways' neighbor genes, neighbor genes' cell lines all in
    one MATCH) produces a cartesian product of unrelated combinations
    (every target cell line repeated once per neighbor/pathway/cell-line
    combination), and a flat LIMIT on top of that can silently drop real
    pathways or neighbor genes before they're ever returned. Ingestion
    already bounds this gene's graph size (max_pathways / max_neighbor_genes
    / max_cell_lines_per_gene in ingest.py), so no extra capping is needed
    here.
    """
    nodes: dict[str, dict] = {gene: {"id": gene, "type": "gene", "role": "target"}}
    edges: list[dict] = []

    try:
        target_lines = run_query(
            """
            MATCH (g:Gene {symbol: $gene})-[r:EXPRESSED_IN]->(c:CellLine)
            RETURN c.cellosaurus_id AS cvcl, r.score AS score
            """,
            {"gene": gene},
        )
        for row in target_lines:
            cvcl = row["cvcl"]
            nodes[cvcl] = {"id": cvcl, "type": "cell_line"}
            edges.append({"source": gene, "target": cvcl, "relation": "EXPRESSED_IN", "weight": row["score"]})

        pathways = run_query(
            """
            MATCH (g:Gene {symbol: $gene})-[:MEMBER_OF]->(p:Pathway)
            RETURN p.pathway_id AS id, p.name AS name, p.url AS url
            """,
            {"gene": gene},
        )
        for pw in pathways:
            pid = pw["id"]
            nodes[pid] = {"id": pid, "type": "pathway", "name": pw["name"], "url": pw["url"]}
            edges.append({"source": gene, "target": pid, "relation": "MEMBER_OF"})

            neighbors = run_query(
                """
                MATCH (p:Pathway {pathway_id: $pid})-[:CONTAINS]->(ng:Gene)
                OPTIONAL MATCH (ng)-[r:EXPRESSED_IN]->(nc:CellLine)
                RETURN ng.symbol AS gene, nc.cellosaurus_id AS cvcl, r.score AS score
                """,
                {"pid": pid},
            )
            for row in neighbors:
                ngene = row["gene"]
                if ngene == gene:
                    continue
                if ngene not in nodes:
                    nodes[ngene] = {"id": ngene, "type": "gene", "role": "pathway_neighbor"}
                    edges.append({"source": pid, "target": ngene, "relation": "CONTAINS"})

                ncvcl = row["cvcl"]
                if ncvcl:
                    nodes[ncvcl] = {"id": ncvcl, "type": "cell_line"}
                    edges.append({
                        "source": ngene, "target": ncvcl,
                        "relation": "EXPRESSED_IN", "weight": row["score"],
                    })
    except Exception as exc:
        print(f"[queries] Neo4j unavailable (graph_to_json): {exc}")
        # Fall through with whatever nodes/edges were gathered before the
        # failure (at minimum, the target gene node itself).

    node_list = list(nodes.values())
    return {
        "gene":       gene,
        "nodes":      node_list,
        "edges":      edges,
        "node_count": len(node_list),
        "edge_count": len(edges),
    }


if __name__ == "__main__":
    print("Neo4j graph query self-test (EGFR)...")
    j = graph_to_json("EGFR")
    print(f"Nodes: {j['node_count']}  Edges: {j['edge_count']}")
    by_type: dict[str, int] = {}
    for n in j["nodes"]:
        by_type[n["type"]] = by_type.get(n["type"], 0) + 1
    print("Node types:", by_type)

    neighbors = query_pathway_neighbor_genes("EGFR")
    print(f"\nPathway-neighbor genes: {len(neighbors)}")
    for n in neighbors[:10]:
        print(" ", n["gene"], "shares", len(n["shared_pathways"]), "pathway(s)")

    best = query_best_cell_lines_via_pathway("EGFR")
    print(f"\nBest cell lines via pathway: {len(best)}")
    for b in best:
        print(" ", b["cellosaurus_id"], b["official_name"], "max_score=", round(b["max_score"], 3))

