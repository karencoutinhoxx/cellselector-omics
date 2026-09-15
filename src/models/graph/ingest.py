import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import CELL_LINE_LOOKUP
from src.models.agentic.pathways import get_cached_pathway_genes, get_kegg_pathways
from src.models.classical.scorer import classify_gene, load_mappings, score_rna_expression
from src.models.classical.weights_learned import VALIDATION_SET
from src.models.graph.neo4j_client import run_query

# ─────────────────────────────────────────────────────────────────────────────
# One-time (or periodically re-run) ETL: populate Neo4j with gene / pathway /
# cell-line nodes for the validation gene set. NOT called per API request —
# api/main.py's /graph/* endpoints only read what's already here.
# ─────────────────────────────────────────────────────────────────────────────

_LOOKUP       = pd.read_parquet(CELL_LINE_LOOKUP, columns=["cellosaurus_id", "disease", "lineage"])
_DISEASE_MAP  = dict(zip(_LOOKUP["cellosaurus_id"], _LOOKUP["disease"]))
_LINEAGE_MAP  = dict(zip(_LOOKUP["cellosaurus_id"], _LOOKUP["lineage"]))


def _lookup(m: dict, cvcl: str) -> str:
    v = m.get(cvcl)
    return v if isinstance(v, str) and v else ""


def _ingest_expression_edges(gene: str, hpa_to_cvcl: dict, gsm_to_cvcl: dict, max_cell_lines: int) -> None:
    """Score gene's RNA expression and MERGE EXPRESSED_IN edges to its top cell lines."""
    gene_class = classify_gene(gene)
    rna = score_rna_expression(gene, hpa_to_cvcl, gsm_to_cvcl, gene_class=gene_class)
    if len(rna) == 0:
        return

    top = rna.nlargest(max_cell_lines, "rna_score")
    rows = [
        {
            "cvcl":    row["cellosaurus_id"],
            "score":   float(row["rna_score"]),
            "disease": _lookup(_DISEASE_MAP, row["cellosaurus_id"]),
            "lineage": _lookup(_LINEAGE_MAP, row["cellosaurus_id"]),
        }
        for _, row in top.iterrows()
    ]
    run_query(
        """
        UNWIND $rows AS row
        MERGE (c:CellLine {cellosaurus_id: row.cvcl})
        SET c.disease = row.disease, c.lineage = row.lineage
        WITH c, row
        MATCH (g:Gene {symbol: $gene})
        MERGE (g)-[r:EXPRESSED_IN]->(c)
        SET r.score = row.score
        """,
        {"rows": rows, "gene": gene},
    )


def ingest_gene(
    gene: str,
    hpa_to_cvcl: dict,
    gsm_to_cvcl: dict,
    scored_genes: set[str],
    max_pathways: int = 10,
    max_neighbor_genes: int = 50,
    max_cell_lines_per_gene: int = 10,
    max_cell_lines_per_neighbor: int = 3,
) -> None:
    """
    Ingest one gene: its own node + expression edges, its KEGG pathways, and
    each pathway's neighbor genes — WITH their own expression edges too.
    (Neighbor genes need expression data of their own, or
    query_best_cell_lines_via_pathway would never find a pathway-connected
    cell line for any gene outside the validation set.)

    scored_genes tracks which genes have already had RNA expression scored
    and written in this run, so a gene that shows up as a neighbor under
    multiple target genes' pathways (common — pathways overlap heavily)
    only gets scored once.

    max_pathways / max_neighbor_genes: raised from 3/15 (was: same values as
    the OLD, now-fixed get_kegg_pathways()/get_genes_in_pathway() caps,
    compounding them) to 10/50. get_kegg_pathways() now returns the true,
    relevance-filtered list (some genes have 40+ real pathways; KEGG's
    "Human Diseases" noise is already excluded upstream), and pathways
    themselves can genuinely have 300+ true members — going fully uncapped
    here would mean scoring RNA expression for potentially thousands of
    unique neighbor genes per ingestion run (each a real parquet read), so
    this stays a bounded sample rather than the complete pathway, just a
    much less severe truncation than before.
    """
    print(f"Ingesting {gene}...")

    run_query("MERGE (g:Gene {symbol: $symbol})", {"symbol": gene})

    if gene not in scored_genes:
        try:
            _ingest_expression_edges(gene, hpa_to_cvcl, gsm_to_cvcl, max_cell_lines_per_gene)
        except Exception as exc:
            print(f"  [warning] expression scoring failed for {gene}: {exc}")
        scored_genes.add(gene)

    try:
        pathways = get_kegg_pathways(gene)
        for pathway in pathways[:max_pathways]:
            pid, pname, purl = pathway["id"], pathway["name"], pathway["url"]

            run_query(
                """
                MERGE (p:Pathway {pathway_id: $pid})
                SET p.name = $pname, p.url = $purl
                WITH p
                MATCH (g:Gene {symbol: $gene})
                MERGE (g)-[:MEMBER_OF]->(p)
                """,
                {"pid": pid, "pname": pname, "purl": purl, "gene": gene},
            )

            neighbor_genes = [
                n for n in get_cached_pathway_genes(pid)[:max_neighbor_genes] if n != gene
            ]
            if not neighbor_genes:
                continue

            run_query(
                """
                UNWIND $neighbor_genes AS ngene
                MERGE (ng:Gene {symbol: ngene})
                WITH ng
                MATCH (p:Pathway {pathway_id: $pid})
                MERGE (p)-[:CONTAINS]->(ng)
                """,
                {"neighbor_genes": neighbor_genes, "pid": pid},
            )

            for ngene in neighbor_genes:
                if ngene in scored_genes:
                    continue
                try:
                    _ingest_expression_edges(
                        ngene, hpa_to_cvcl, gsm_to_cvcl, max_cell_lines_per_neighbor
                    )
                except Exception as exc:
                    print(f"  [warning] expression scoring failed for neighbor {ngene}: {exc}")
                scored_genes.add(ngene)
    except Exception as exc:
        print(f"  [warning] pathway ingestion failed for {gene}: {exc}")

    print(f"  Done: {gene}")


def ingest_mutation_edges(batch_size: int = 500) -> dict:
    """
    Create (:Gene)-[:HAS_MUTATION {protein_change, hotspot, likely_lof,
    clinical_significance, impact_score}]->(:CellLine) edges.

    Scope (deliberately narrow — this is NOT the full 1M-row mutation file):
      - genes: every loss_of_function gene (mutation is their PRIMARY signal)
        plus the validation set's tissue_specific genes (secondary bonus path)
      - variants: LikelyLoF OR Hotspot OR pathogenic/likely_pathogenic ClinSig
        (see mutation_scorer.significant_variants)

    Returns {"genes": [...], "edge_count": int, "cell_lines": int}.
    """
    from src.models.classical.mutation_scorer import significant_variants
    from src.models.classical.scorer import GENE_CLASSES

    lof_genes = list(GENE_CLASSES["loss_of_function"])
    ts_genes  = [g for g in VALIDATION_SET if classify_gene(g) == "tissue_specific"]
    genes = sorted(set(lof_genes) | set(ts_genes))
    print(f"Ingesting HAS_MUTATION edges for {len(genes)} genes: {genes}")

    variants = significant_variants(genes)
    print(f"  {len(variants)} significant variants "
          f"({variants['cellosaurus_id'].nunique()} distinct cell lines)")
    if len(variants) == 0:
        return {"genes": genes, "edge_count": 0, "cell_lines": 0}

    rows = [
        {
            "gene":     r["gene"],
            "cvcl":     r["cellosaurus_id"],
            "pchange":  r["protein_change"] or "(unspecified)",
            "hotspot":  bool(r["hotspot"]),
            "lof":      bool(r["likely_lof"]),
            "clinsig":  r["clinical_significance"] or "",
            "impact":   float(r["impact_score"]),
            "disease":  _lookup(_DISEASE_MAP, r["cellosaurus_id"]),
            "lineage":  _lookup(_LINEAGE_MAP, r["cellosaurus_id"]),
        }
        for r in variants.to_dict("records")
    ]

    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        run_query(
            """
            UNWIND $rows AS row
            MERGE (g:Gene {symbol: row.gene})
            MERGE (c:CellLine {cellosaurus_id: row.cvcl})
            SET c.disease = row.disease, c.lineage = row.lineage
            MERGE (g)-[m:HAS_MUTATION {protein_change: row.pchange}]->(c)
            SET m.hotspot = row.hotspot,
                m.likely_lof = row.lof,
                m.clinical_significance = row.clinsig,
                m.impact_score = row.impact
            """,
            {"rows": chunk},
        )
        print(f"  ...{min(i + batch_size, len(rows))}/{len(rows)} edges merged")

    return {
        "genes": genes,
        "edge_count": len(rows),
        "cell_lines": variants["cellosaurus_id"].nunique(),
    }


def enrich_cell_line_lineage(batch_size: int = 500) -> dict:
    """
    Set disease / lineage / tissue_type on CellLine nodes ALREADY in the
    graph, from cell_line_lookup.parquet — the nomenclature spine where
    these live (master_merged.parquet has no disease/lineage columns; this
    is the same source _ingest_expression_edges already reads). MATCH, not
    MERGE: enriches existing nodes only, never adds one.

    In this schema `lineage` IS the tissue of origin (blood, colorectal,
    central_nervous_system, ...); there is no separate tissue_type column,
    so tissue_type mirrors lineage. Empty strings are written as null
    (SET c.prop = null removes the property in Cypher).
    """
    existing = {r["id"] for r in run_query(
        "MATCH (c:CellLine) RETURN c.cellosaurus_id AS id")}
    lkp = pd.read_parquet(
        CELL_LINE_LOOKUP, columns=["cellosaurus_id", "disease", "lineage"]
    )

    def _clean(v) -> str | None:
        return v.strip() if isinstance(v, str) and v.strip() else None

    rows = [
        {"cvcl": r["cellosaurus_id"],
         "disease": _clean(r["disease"]),
         "lineage": _clean(r["lineage"])}
        for _, r in lkp.iterrows()
        if r["cellosaurus_id"] in existing
    ]
    with_disease = sum(1 for r in rows if r["disease"])
    for i in range(0, len(rows), batch_size):
        run_query(
            """
            UNWIND $rows AS row
            MATCH (c:CellLine {cellosaurus_id: row.cvcl})
            SET c.disease = row.disease,
                c.lineage = row.lineage,
                c.tissue_type = row.lineage
            """,
            {"rows": rows[i:i + batch_size]},
        )
    return {"cell_lines_matched": len(rows), "with_disease": with_disease,
            "graph_cell_lines": len(existing)}


def enrich_gene_roles() -> dict:
    """
    Set `role` on Gene nodes ALREADY in the graph, from GENE_ROLES
    (models/classical/scorer.py) — receptor tyrosine kinase / tumor
    suppressor / oncogene / hormone receptor / immune checkpoint marker /
    proliferation marker. MATCH, not MERGE: existing Gene nodes only.
    """
    from src.models.classical.scorer import GENE_ROLES

    existing = {r["s"] for r in run_query("MATCH (g:Gene) RETURN g.symbol AS s")}
    rows = [{"gene": g, "role": role}
            for g, role in GENE_ROLES.items() if g in existing]
    skipped = [g for g in GENE_ROLES if g not in existing]
    run_query(
        """
        UNWIND $rows AS row
        MATCH (g:Gene {symbol: row.gene})
        SET g.role = row.role
        """,
        {"rows": rows},
    )
    return {"genes_enriched": len(rows),
            "skipped_not_in_graph": sorted(skipped)}


def enrich_all() -> dict:
    """Steps 1-2: enrich existing CellLine + Gene nodes with lineage / role."""
    cl = enrich_cell_line_lineage()
    print(f"  CellLine lineage: {cl}")
    gr = enrich_gene_roles()
    print(f"  Gene roles: {gr}")
    return {"cell_line": cl, "gene": gr}


def ingest_all() -> None:
    genes = sorted(VALIDATION_SET.keys())
    print(f"Ingesting {len(genes)} genes into Neo4j...")

    hpa_to_cvcl, ach_to_cvcl, gsm_to_cvcl = load_mappings()
    scored_genes: set[str] = set()

    t0 = time.time()
    for i, gene in enumerate(genes, 1):
        print(f"[{i}/{len(genes)}]  (elapsed {time.time() - t0:.0f}s, {len(scored_genes)} genes scored so far)")
        ingest_gene(gene, hpa_to_cvcl, gsm_to_cvcl, scored_genes)
        time.sleep(0.5)  # courtesy pause between genes

    print(f"Ingestion complete in {time.time() - t0:.0f}s. Unique genes scored: {len(scored_genes)}")

    print("\nEnriching nodes with lineage / role properties...")
    enrich_all()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "mutations":
        print(f"\nHAS_MUTATION ingestion complete: {ingest_mutation_edges()}")
    elif cmd == "enrich":
        print(f"\nEnrichment complete: {enrich_all()}")
    else:
        ingest_all()

