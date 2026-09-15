import json
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import OUTPUTS_DIR
from src.models.graph.neo4j_client import run_query

# Precomputed, same pattern as outputs/production_weights.json
# (api/main.py::_warm_learned_weights) — a live personalized-PageRank call
# is cheap ONCE the graph is in memory (~10ms), but exporting the graph
# from Neo4j fresh per cold process, plus doing that on the critical path
# of a user-facing API request, is exactly the cost that file was already
# introduced to avoid. Regenerate via precompute_rwr_scores() whenever
# VALIDATION_SET changes or the graph is re-ingested.
RWR_SCORES_FILE = OUTPUTS_DIR / "rwr_scores.json"

# ─────────────────────────────────────────────────────────────────────────────
# Random Walk with Restart (RWR) — Köhler et al. 2008, "Walking the
# interactome for prioritization of candidate disease genes". Implemented
# via networkx.pagerank's personalization vector, which IS random-walk-
# with-restart under a different name (same algorithm, standard library).
#
# Tested standalone first (44-gene validation set): RWR ALONE is worse than
# production (MRR 0.1358 vs 0.1547, not significant, p=0.557) — it has no
# mechanism analogous to production's mutation-weight-0.8 LOF handling or
# any ubiquitous-specific signal. But blended IN ADDITION to the existing
# formula (small weight, rescaling the rest down proportionally — same
# construction as copy_number_scorer.apply_amplification_copy_number_weight
# / weights_learned._apply_lof_mutation_weight), it showed genuine lift in
# ALL THREE classes via grid search: tissue_specific +0.0937 MRR (rwr=0.24),
# ubiquitous +0.0448 (rwr=0.28, n=4 genes — small sample), loss_of_function
# +0.0880 (rwr=0.38). See RWR_WEIGHT_BY_CLASS below.
#
# Graph: undirected (see _load_graph) — biological "relatedness" isn't
# strictly directional for a walk (a cell line expressing a pathway member
# is as informative walked backward as forward), though HAS_MUTATION really
# is gene->cell_line, not symmetric information — a modeling simplification,
# not an obviously-correct default.
# ─────────────────────────────────────────────────────────────────────────────

RWR_WEIGHT_BY_CLASS = {
    "tissue_specific":  0.24,
    "ubiquitous":       0.28,
    "loss_of_function": 0.38,
}
RWR_RESTART_PROB = 0.15  # alpha = 1 - restart_prob, per Köhler et al.'s typical range (0.15-0.30)


def apply_rwr_weight(weights: dict, gene_class: str) -> dict:
    """
    Rescale an already-resolved weights dict (learned tissue_specific/
    ubiquitous weights, or the LOF mutation-primary vector, or the
    amplification copy_number-primary vector) so it sums to
    (1 - rwr_weight), then add "rwr" at the class's grid-verified constant.
    Same rescale-by-(1-w) construction used everywhere else tonight.
    """
    w = RWR_WEIGHT_BY_CLASS.get(gene_class, 0.0)
    if w <= 0:
        return weights
    scale = 1.0 - w
    out = {k: v * scale for k, v in weights.items()}
    out["rwr"] = w
    return out


_GRAPH: nx.Graph | None = None
_RWR_CACHE: dict[str, pd.DataFrame] = {}


def _load_graph() -> nx.Graph:
    """
    Export the Neo4j graph to an undirected NetworkX graph once per
    process (Gene/CellLine/Pathway nodes, MEMBER_OF/CONTAINS/EXPRESSED_IN/
    HAS_MUTATION edges) and cache it. Small graph (~3,700 nodes, ~12,000
    edges) — the export itself takes well under a second; RWR's own
    per-gene pagerank call is ~10ms once the graph is loaded, so this
    caching is what keeps repeated score_rwr() calls cheap, not the walk
    itself.
    """
    global _GRAPH
    if _GRAPH is not None:
        return _GRAPH

    G = nx.Graph()
    for label in ["Gene", "CellLine", "Pathway"]:
        rows = run_query(
            f"MATCH (n:{label}) RETURN n.symbol AS symbol, n.cellosaurus_id AS cvcl, "
            f"n.pathway_id AS pid, n.name AS name"
        )
        for r in rows:
            if label == "Gene":
                nid = f"Gene:{r['symbol']}"
            elif label == "CellLine":
                nid = f"CellLine:{r['cvcl']}"
            else:
                nid = f"Pathway:{r['pid']}"
            G.add_node(nid, type=label)

    edge_queries = {
        "MEMBER_OF":    ("MATCH (g:Gene)-[:MEMBER_OF]->(p:Pathway) RETURN g.symbol AS a, p.pathway_id AS b", "Gene", "Pathway"),
        "CONTAINS":     ("MATCH (p:Pathway)-[:CONTAINS]->(g:Gene) RETURN p.pathway_id AS a, g.symbol AS b", "Pathway", "Gene"),
        "EXPRESSED_IN": ("MATCH (g:Gene)-[:EXPRESSED_IN]->(c:CellLine) RETURN g.symbol AS a, c.cellosaurus_id AS b", "Gene", "CellLine"),
        "HAS_MUTATION": ("MATCH (g:Gene)-[:HAS_MUTATION]->(c:CellLine) RETURN g.symbol AS a, c.cellosaurus_id AS b", "Gene", "CellLine"),
    }
    for rel, (q, ta, tb) in edge_queries.items():
        for r in run_query(q):
            na, nb = f"{ta}:{r['a']}", f"{tb}:{r['b']}"
            if G.has_node(na) and G.has_node(nb):
                G.add_edge(na, nb, rel=rel)

    _GRAPH = G
    return G


_PRECOMPUTED_RWR: dict[str, dict[str, float]] | None = None


def _load_precomputed_rwr() -> dict[str, dict[str, float]]:
    """Load outputs/rwr_scores.json once per process: {gene: {cvcl: score}}."""
    global _PRECOMPUTED_RWR
    if _PRECOMPUTED_RWR is not None:
        return _PRECOMPUTED_RWR
    if RWR_SCORES_FILE.exists():
        with open(RWR_SCORES_FILE, encoding="utf-8") as f:
            _PRECOMPUTED_RWR = json.load(f)
    else:
        _PRECOMPUTED_RWR = {}
    return _PRECOMPUTED_RWR


def _compute_rwr_live(gene: str) -> pd.DataFrame:
    """
    Random-walk-with-restart score of `gene` across all cell lines, via a
    LIVE personalized PageRank call from the gene's graph node. Only
    reached when `gene` isn't in the precomputed file — score_rwr() below
    is the normal entry point.

    Returns: DataFrame [cellosaurus_id, rwr_score] — rwr_score is the raw
    steady-state visit probability's PERCENTILE RANK across cell lines
    (0-1), same _pct_rank convention as every other scorer in this project
    (raw pagerank magnitudes are tiny, ~1e-4 to 1e-2, and not on a scale
    comparable to the other 0-1 score columns otherwise).

    A gene absent from the graph, or with no CellLine-type nodes reachable,
    returns an empty frame (0.0 after the caller's fillna, same convention
    as copy_number_score / mutation_impact_score for missing data).
    """
    empty = pd.DataFrame({
        "cellosaurus_id": pd.Series(dtype="object"),
        "rwr_score":      pd.Series(dtype="float64"),
    })

    G = _load_graph()
    seed = f"Gene:{gene}"
    if not G.has_node(seed):
        return empty

    personalization = {n: 0.0 for n in G.nodes()}
    personalization[seed] = 1.0
    scores = nx.pagerank(
        G, alpha=1 - RWR_RESTART_PROB, personalization=personalization,
        max_iter=100, tol=1e-6,
    )

    cvcl_to_raw = {
        n.split(":", 1)[1]: s for n, s in scores.items()
        if G.nodes[n]["type"] == "CellLine"
    }
    if not cvcl_to_raw:
        return empty

    raw = pd.Series(cvcl_to_raw)
    return pd.DataFrame({
        "cellosaurus_id": raw.index,
        "rwr_score": raw.rank(pct=True, method="average").values,
    }).reset_index(drop=True)


def score_rwr(gene: str) -> pd.DataFrame:
    """
    RWR score of `gene` across all cell lines. Reads outputs/rwr_scores.json
    (precomputed offline via precompute_rwr_scores(), same pattern as
    production_weights.json) first; falls back to a live pagerank call,
    LOUDLY (not silently — a cold live call means the precomputed file is
    stale or missing this gene), if the gene isn't in that file.

    Returns: DataFrame [cellosaurus_id, rwr_score]. Cached per gene.
    """
    if gene in _RWR_CACHE:
        return _RWR_CACHE[gene]

    precomputed = _load_precomputed_rwr()
    if gene in precomputed:
        raw = precomputed[gene]
        result = pd.DataFrame({
            "cellosaurus_id": list(raw.keys()),
            "rwr_score": list(raw.values()),
        })
    else:
        print(f"[rwr_scorer] WARNING: {gene} not in {RWR_SCORES_FILE} — "
              f"computing live (graph export + pagerank, not the ~8min class "
              f"of cost, but still avoidable by regenerating the file).")
        result = _compute_rwr_live(gene)

    _RWR_CACHE[gene] = result
    return result


def precompute_rwr_scores(genes: list[str], verbose: bool = True) -> dict[str, dict[str, float]]:
    """
    Compute RWR scores for every gene in `genes` and return
    {gene: {cvcl: score}}, ready to json.dump to RWR_SCORES_FILE.
    Standalone entry point — run once offline, not from the API.
    """
    out: dict[str, dict[str, float]] = {}
    for i, gene in enumerate(genes, 1):
        if verbose:
            print(f"  [{i}/{len(genes)}] {gene}...")
        df = _compute_rwr_live(gene)
        out[gene] = dict(zip(df["cellosaurus_id"], df["rwr_score"].astype(float)))
    return out


if __name__ == "__main__":
    from src.models.classical.weights_learned import VALIDATION_SET

    print(f"Precomputing RWR scores for {len(VALIDATION_SET)} VALIDATION_SET genes...")
    scores = precompute_rwr_scores(list(VALIDATION_SET.keys()))
    RWR_SCORES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RWR_SCORES_FILE, "w", encoding="utf-8") as f:
        json.dump(scores, f)
    print(f"Wrote {RWR_SCORES_FILE} ({sum(len(v) for v in scores.values())} gene-cellline entries)")

