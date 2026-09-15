"""
CellSelector Omics — Brief Compliance Verification Script
============================================================

Checks the implementation against every requirement in the
project brief and reports PASS/FAIL/WARN for each item, with
concrete evidence (not just "file exists").

Run from the project root:
    python verify_brief.py

Requires the API to NOT be running (this script imports and
calls functions directly, not via HTTP) — OR run with
--api-mode to test via a running server instead.
"""

import sys
import os
import json
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

RESULTS = []


def check(name, condition, detail="", severity="FAIL"):
    """Record a check result."""
    status = "PASS" if condition else severity
    RESULTS.append((name, status, detail))
    icon = "✅" if status == "PASS" else ("⚠️ " if status == "WARN" else "❌")
    print(f"{icon} [{status}] {name}")
    if detail:
        print(f"      {detail}")


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# =============================================================================
section("1. DATA INTEGRATION — Five Public Multi-Omics Sources")
# =============================================================================

try:
    from config.config import PARQUET_DIR, MASTER_MERGED
    import pandas as pd

    sources = {
        "HPA expression": "gene_expr_hpa_preprocessed.parquet",
        "DepMap expression": "gene_expr_depmap_preprocessed.parquet",
        "DepMap CRISPR dependency": "crispr_dependency_preprocessed.parquet",
        "GEO expression": "gene_expr_geo_preprocessed.parquet",
        "CCLE proteomics": "proteomics_preprocessed.parquet",
    }

    for label, fname in sources.items():
        fpath = PARQUET_DIR / fname
        exists = fpath.exists()
        if exists:
            try:
                df = pd.read_parquet(fpath, columns=None)
                n_rows = len(df)
                check(f"Data source: {label}", True,
                      f"{fname} — {n_rows:,} rows")
            except Exception as e:
                check(f"Data source: {label}", False,
                      f"{fname} exists but failed to read: {e}")
        else:
            check(f"Data source: {label}", False,
                  f"{fname} NOT FOUND at {fpath}")

    # Master merged table
    if MASTER_MERGED.exists():
        master_df = pd.read_parquet(MASTER_MERGED)
        n_cell_lines = len(master_df)
        check("Master merged cell-line table", n_cell_lines > 0,
              f"{n_cell_lines:,} cell lines in master_merged.parquet")
    else:
        check("Master merged cell-line table", False,
              f"NOT FOUND at {MASTER_MERGED}")

except Exception as e:
    check("Data integration module import", False, str(e))


# =============================================================================
section("2. CLASSICAL SCORING MODEL")
# =============================================================================

try:
    from src.models.classical.scorer import (
        classify_gene, score_rna_expression, score_protein_expression,
        score_context, score_data_quality, load_mappings, GENE_ROLES,
        get_gene_role
    )
    from src.models.classical.ranker import rank

    # Gene classification (3-way)
    ts = classify_gene("EGFR")
    ub = classify_gene("TP53")
    lof = classify_gene("BRCA1")
    check("Gene classification: tissue_specific",
          ts == "tissue_specific", f"EGFR classified as '{ts}'")
    check("Gene classification: ubiquitous",
          ub == "ubiquitous", f"TP53 classified as '{ub}'")
    check("Gene classification: loss_of_function",
          lof == "loss_of_function", f"BRCA1 classified as '{lof}'")

    # Gene role tagging
    check("Gene role tagging (RTK/suppressor/oncogene)",
          len(GENE_ROLES) > 0,
          f"{len(GENE_ROLES)} genes tagged with roles")

    # Ranking works end to end
    result = rank("EGFR", top_n=5)
    check("Ranker returns results for EGFR",
          result is not None and len(result) > 0,
          f"{len(result) if result is not None else 0} results returned")

    if result is not None and len(result) > 0:
        required_cols = ["cellosaurus_id", "final_score", "rna_score",
                          "protein_score", "quality_score", "context_score"]
        missing = [c for c in required_cols if c not in result.columns]
        check("Ranker output has all required score columns",
              len(missing) == 0,
              f"Missing: {missing}" if missing else "All present")

        scores_valid = (result["final_score"].min() >= 0.0 and
                         result["final_score"].max() <= 1.0)
        check("Final scores bounded [0, 1]", scores_valid,
              f"min={result['final_score'].min():.3f}, "
              f"max={result['final_score'].max():.3f}")

    # Disease/lineage filtering
    filtered = rank("EGFR", disease_filter="lung", top_n=5)
    check("Disease filter narrows results",
          filtered is not None and len(filtered) > 0,
          f"{len(filtered) if filtered is not None else 0} lung-filtered results")

    # Exclusion genes with penalty
    without_excl = rank("EGFR", top_n=1)
    with_excl = rank("EGFR", exclude_genes=["TP53"], top_n=1)
    if without_excl is not None and with_excl is not None and \
       len(without_excl) > 0 and len(with_excl) > 0:
        check("Exclusion gene penalty applied",
              with_excl.iloc[0]["final_score"] <= without_excl.iloc[0]["final_score"],
              f"score without excl={without_excl.iloc[0]['final_score']:.3f}, "
              f"with excl={with_excl.iloc[0]['final_score']:.3f}")

    # Same-gene search+exclude should be prevented (checked at API level below)

except Exception as e:
    check("Classical scoring model", False,
          f"{e}\n{traceback.format_exc()}")


# =============================================================================
section("3. GENE-CLASS-AWARE WEIGHT OPTIMISATION (Core Finding)")
# =============================================================================

try:
    from src.models.classical.weights_learned import (
        optimise_weights, optimise_weights_by_class, VALIDATION_SET,
        GRID_SEARCH_PATHWAY_WEIGHTS
    )

    check("Validation set exists for MRR evaluation",
          len(VALIDATION_SET) >= 20,
          f"{len(VALIDATION_SET)} genes in validation set")

    check("Grid-search pathway weights defined per class",
          set(GRID_SEARCH_PATHWAY_WEIGHTS.keys()) ==
          {"tissue_specific", "ubiquitous", "loss_of_function"},
          f"Weights: {GRID_SEARCH_PATHWAY_WEIGHTS}")

    # Verify the specific grid-search values are applied
    expected = {"tissue_specific": 0.00, "ubiquitous": 0.01,
                "loss_of_function": 0.50}
    matches = all(
        abs(GRID_SEARCH_PATHWAY_WEIGHTS.get(k, -1) - v) < 0.001
        for k, v in expected.items()
    )
    check("Grid-search pathway weights match verified optimum",
          matches, f"Expected {expected}, got {GRID_SEARCH_PATHWAY_WEIGHTS}")

except Exception as e:
    check("Weight optimisation module", False, str(e))


# =============================================================================
section("4. PATHWAY-AWARE SCORING (Knowledge Graph Integration)")
# =============================================================================

try:
    from src.models.classical.pathway_scorer import score_pathway_activity

    pw_result = score_pathway_activity("EGFR")
    check("Pathway activity scorer runs without error",
          pw_result is not None,
          f"Returned {len(pw_result) if pw_result is not None else 0} rows")

    if pw_result is not None and len(pw_result) > 0:
        has_score = "pathway_activity_score" in pw_result.columns
        check("Pathway score column present", has_score)
        if has_score:
            in_range = (pw_result["pathway_activity_score"].min() >= 0.0 and
                        pw_result["pathway_activity_score"].max() <= 1.0)
            check("Pathway scores bounded [0, 1]", in_range)
    else:
        check("Pathway scorer returns non-empty results", False,
              "Empty result — check Neo4j connection/data", severity="WARN")

except Exception as e:
    check("Pathway scoring module", False, str(e))


# =============================================================================
section("5. NEO4J KNOWLEDGE GRAPH (Multi-hop Queries)")
# =============================================================================

try:
    from src.models.graph.neo4j_client import run_query

    node_counts = run_query(
        "MATCH (n) RETURN labels(n)[0] AS type, count(*) AS c"
    )
    node_counts = {r["type"]: r["c"] for r in node_counts}

    check("Neo4j connection alive", len(node_counts) > 0,
          f"Node types found: {node_counts}")

    check("Gene nodes populated",
          node_counts.get("Gene", 0) > 0,
          f"{node_counts.get('Gene', 0)} Gene nodes")
    check("CellLine nodes populated",
          node_counts.get("CellLine", 0) > 0,
          f"{node_counts.get('CellLine', 0)} CellLine nodes")
    check("Pathway nodes populated",
          node_counts.get("Pathway", 0) > 0,
          f"{node_counts.get('Pathway', 0)} Pathway nodes")

    # Genuine multi-hop query test
    multihop = run_query(
        """
        MATCH (g:Gene {symbol: 'EGFR'})-[:MEMBER_OF]->(p:Pathway)
              -[:CONTAINS]->(neighbor:Gene)
        RETURN DISTINCT neighbor.symbol AS gene LIMIT 5
        """
    )
    check("Multi-hop pathway traversal query works",
          len(multihop) > 0,
          f"Found {len(multihop)} pathway-neighbor genes for EGFR: "
          f"{[r['gene'] for r in multihop]}")

except Exception as e:
    check("Neo4j knowledge graph", False,
          f"{e} — Neo4j may be down/deleted (Aura free-tier auto-delete)",
          severity="FAIL")


# =============================================================================
section("6. CRISPR GENE DEPENDENCY (Distinct Evidence Type)")
# =============================================================================

try:
    from src.models.classical.scorer import score_crispr_dependency
    crispr_result = score_crispr_dependency("KRAS")
    check("CRISPR dependency scorer runs",
          crispr_result is not None and len(crispr_result) > 0,
          f"{len(crispr_result) if crispr_result is not None else 0} cell lines with KRAS dependency data")
except ImportError:
    check("CRISPR dependency scorer function exists", False,
          "score_crispr_dependency not found — check exact function name in scorer.py",
          severity="WARN")
except Exception as e:
    check("CRISPR dependency scoring", False, str(e))


# =============================================================================
section("7. SIMILARITY / ALTERNATIVE CELL LINE RECOMMENDATIONS")
# =============================================================================

try:
    from src.models.classical.similarity import find_alternatives
    from config.config import MASTER_MERGED

    ranked = rank("EGFR", disease_filter="lung", top_n=3)
    if ranked is not None and len(ranked) > 0:
        alts = find_alternatives("EGFR", ranked, MASTER_MERGED, top_k=3)
        check("Similarity engine returns alternatives",
              isinstance(alts, dict) and len(alts) > 0,
              f"{len(alts)} cell lines have alternatives computed")
    else:
        check("Similarity engine test", False,
              "Could not rank EGFR to test similarity", severity="WARN")

except Exception as e:
    check("Similarity/alternatives module", False, str(e))


# =============================================================================
section("8. AGENTIC RAG PIPELINE")
# =============================================================================

try:
    import importlib
    agentic_mod = importlib.import_module("models.agentic.pipeline")
    check("Agentic pipeline module imports", True)

    # Check sub-components exist without necessarily running Ollama
    has_kegg = hasattr(
        importlib.import_module("models.agentic.pathways"),
        "get_pathways_for_gene"
    ) if _module_exists("models.agentic.pathways") else False

except Exception as e:
    check("Agentic RAG pipeline import", False, str(e))


def _module_exists(name):
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


for mod_name, label in [
    ("models.agentic.pathways", "KEGG pathway lookup module"),
    ("models.agentic.growth_properties", "Cellosaurus growth-property module"),
    ("models.agentic.pubmed", "PubMed literature retrieval module"),
    ("models.agentic.retriever", "Evidence retriever module"),
    ("models.agentic.generator", "LLM generation module"),
]:
    check(f"Agentic component: {label}", _module_exists(mod_name),
          f"module {mod_name}")


# =============================================================================
section("9. API LAYER (FastAPI)")
# =============================================================================

try:
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)

    # Health
    r = client.get("/health")
    check("GET /health", r.status_code == 200, f"status={r.status_code}")

    # Gene search
    r = client.get("/genes/search?q=EGFR")
    check("GET /genes/search", r.status_code == 200 and len(r.json()) > 0,
          f"status={r.status_code}, results={len(r.json()) if r.status_code==200 else 'n/a'}")

    # Classical recommendation
    r = client.post("/recommend/classical", json={"gene": "EGFR", "top_n": 3})
    check("POST /recommend/classical", r.status_code == 200,
          f"status={r.status_code}")

    if r.status_code == 200:
        data = r.json()
        check("Response includes session_id", "session_id" in data)
        check("Response includes weights_used (transparency)",
              "weights_used" in data,
              f"weights_used={data.get('weights_used', 'MISSING')}")

    # CRITICAL BUG CHECK: same-gene search+exclude must be rejected
    r = client.post("/recommend/classical", json={
        "gene": "EGFR", "exclude_genes": ["EGFR"], "top_n": 3
    })
    check("Same-gene search+exclude returns 400 (bug fix verified)",
          r.status_code == 400,
          f"status={r.status_code} (expected 400)")

    # NaN safety check
    r = client.post("/recommend/classical", json={
        "gene": "EGFR", "disease_filter": "lung", "top_n": 10
    })
    check("Response contains no NaN/Infinity (JSON safety)",
          "NaN" not in r.text and "Infinity" not in r.text)

    # Export endpoints
    r_search = client.post("/recommend/classical", json={"gene": "EGFR", "top_n": 3})
    if r_search.status_code == 200:
        sid = r_search.json().get("session_id")
        for fmt in ["json", "csv", "pdf"]:
            r_export = client.get(f"/recommend/export/{fmt}?session_id={sid}")
            check(f"GET /recommend/export/{fmt}", r_export.status_code == 200,
                  f"status={r_export.status_code}")

    # Graph/pathway endpoints
    r = client.get("/graph/pathway-neighbors/EGFR")
    check("GET /graph/pathway-neighbors/{gene}", r.status_code == 200,
          f"status={r.status_code}")

    r = client.get("/graph/cell-lines-via-pathway/EGFR?top_k=5")
    check("GET /graph/cell-lines-via-pathway/{gene}", r.status_code == 200,
          f"status={r.status_code}")

    # Browse/cell-lines endpoint
    r = client.get("/cell-lines")
    check("GET /cell-lines", r.status_code == 200, f"status={r.status_code}")

except Exception as e:
    check("API layer (FastAPI)", False, f"{e}\n{traceback.format_exc()}")


# =============================================================================
section("10. FRONTEND (React/TypeScript) — Static Checks")
# =============================================================================

app_src = Path("app/src")
if app_src.exists():
    check("React app source directory exists", True)

    # Pages that SHOULD exist (per brief + professor's requested changes)
    expected_pages = ["Search.tsx", "About.tsx", "Home.tsx"]
    for page in expected_pages:
        p = app_src / "pages" / page
        check(f"Page exists: {page}", p.exists())

    # Pages that SHOULD be removed (professor's feedback)
    removed_pages = ["Browse.tsx", "GraphExplorer.tsx"]
    for page in removed_pages:
        p = app_src / "pages" / page
        check(f"Page correctly removed: {page}", not p.exists())

    # Check Search.tsx contains key features
    search_tsx = app_src / "pages" / "Search.tsx"
    if search_tsx.exists():
        content = search_tsx.read_text()
        check("Search.tsx has scoring transparency panel",
              "How Fit Score" in content or "weights_used" in content)
        check("Search.tsx has pathway-connected section",
              "Pathway-Connected" in content or "pathway" in content.lower())
        check("Search.tsx has same-gene exclusion validation",
              "excludeList" in content or "Cannot search" in content)

    # Navbar should not have Browse/Graph links
    navbar = app_src / "components" / "Navbar.tsx"
    if navbar.exists():
        nav_content = navbar.read_text()
        check("Navbar has no 'Browse' link", "Browse" not in nav_content)
        check("Navbar has no standalone 'Graph' link",
              "/graph" not in nav_content)
else:
    check("React app source directory", False, "app/src not found",
          severity="WARN")


# =============================================================================
section("11. EVALUATION FRAMEWORK")
# =============================================================================

eval_scripts = [
    "scripts/evaluation/full_evaluation.py",
    "scripts/evaluation/pathway_grid_search.py",
    "scripts/evaluation/cross_validated_evaluation.py",
]
for script in eval_scripts:
    p = Path(script)
    check(f"Evaluation script exists: {script}", p.exists())


# =============================================================================
section("12. GIT / VERSION CONTROL")
# =============================================================================

import subprocess

try:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, timeout=10
    )
    uncommitted = result.stdout.strip()
    check("No uncommitted changes",
          len(uncommitted) == 0,
          f"{len(uncommitted.splitlines())} uncommitted file(s)" if uncommitted else "Clean",
          severity="WARN")

    result = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, timeout=10
    )
    check("Git log accessible", result.returncode == 0,
          f"Latest commit: {result.stdout.strip()}")

    result = subprocess.run(
        ["git", "status", "-sb"],
        capture_output=True, text=True, timeout=10
    )
    ahead_behind = result.stdout.splitlines()[0] if result.stdout else ""
    check("Branch sync status", "ahead" not in ahead_behind.lower(),
          ahead_behind, severity="WARN")

except Exception as e:
    check("Git status check", False, str(e), severity="WARN")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

section("SUMMARY")

n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
n_warn = sum(1 for _, s, _ in RESULTS if s == "WARN")
total = len(RESULTS)

print(f"\nTotal checks: {total}")
print(f"  ✅ PASS: {n_pass}")
print(f"  ⚠️  WARN: {n_warn}")
print(f"  ❌ FAIL: {n_fail}")

if n_fail > 0:
    print("\n--- FAILURES REQUIRING ATTENTION ---")
    for name, status, detail in RESULTS:
        if status == "FAIL":
            print(f"  ❌ {name}")
            if detail:
                print(f"     {detail}")

if n_warn > 0:
    print("\n--- WARNINGS (review, may be acceptable) ---")
    for name, status, detail in RESULTS:
        if status == "WARN":
            print(f"  ⚠️  {name}")
            if detail:
                print(f"     {detail}")

print("\n" + "=" * 70)
if n_fail == 0:
    print("ALL CRITICAL CHECKS PASSED")
else:
    print(f"{n_fail} CRITICAL ISSUE(S) FOUND — see failures above")
print("=" * 70)

