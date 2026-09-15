import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import OUTPUTS_DIR
from src.models.classical.rwr_scorer import _load_graph

# ─────────────────────────────────────────────────────────────────────────────
# Metapath-guided walk — Dong et al. 2017, "metapath2vec: Scalable
# Representation Learning for Heterogeneous Networks" is the reference, but
# this is explicitly NOT the full academic algorithm (no learned
# embeddings, no random sampling along metapaths) — a deliberately
# simplified, deterministic middle ground, same framing used when this was
# first tested on EGFR tonight: walk each defined metapath explicitly,
# count/score distinct metapath instances connecting the gene to each cell
# line, weighting direct paths more heavily than 2-hop pathway-mediated
# ones.
#
# Metapaths (matching this project's actual schema):
#   Gene -[HAS_MUTATION]->      CellLine                          (direct)
#   Gene -[EXPRESSED_IN]->      CellLine                          (direct)
#   Gene -[MEMBER_OF]-> Pathway -[CONTAINS]-> Gene -[EXPRESSED_IN]-> CellLine
#                                                                  (2-hop, pathway-mediated)
#
# Same graph as rwr_scorer.py (imported, not re-exported — one Neo4j export
# serves both scorers). Precomputed to outputs/metapath_scores.json, same
# pattern as rwr_scores.json — the graph traversal itself is cheap (no
# training, no iterative walk), so this is mainly for consistency with the
# rest of tonight's "precompute graph-derived scores once" convention.
# ─────────────────────────────────────────────────────────────────────────────

METAPATH_SCORES_FILE = OUTPUTS_DIR / "metapath_scores.json"

DIRECT_WEIGHT = 1.0
PATHWAY_WEIGHT = 0.2  # 2-hop pathway-mediated path, weighted down — same
                       # constants used in tonight's original EGFR test

_METAPATH_CACHE: dict[str, pd.DataFrame] = {}


def _compute_metapath_live(gene: str) -> pd.DataFrame:
    """
    Deterministic metapath-guided score of `gene` across all cell lines.

    Returns: DataFrame [cellosaurus_id, metapath_score] — the raw weighted
    instance count's PERCENTILE RANK across cell lines (0-1), same
    _pct_rank convention as rwr_score/ssgsea_score (raw counts aren't on a
    comparable scale to the other 0-1 score columns otherwise, and here
    they're also highly skewed — most cell lines score exactly 0).
    """
    empty = pd.DataFrame({
        "cellosaurus_id": pd.Series(dtype="object"),
        "metapath_score": pd.Series(dtype="float64"),
    })

    G = _load_graph()
    seed = f"Gene:{gene}"
    if not G.has_node(seed):
        return empty

    scores: dict[str, float] = {}

    # Direct metapaths: Gene -[EXPRESSED_IN|HAS_MUTATION]-> CellLine
    for nbr in G.neighbors(seed):
        if G.nodes[nbr]["type"] != "CellLine":
            continue
        rel = G.edges[seed, nbr]["rel"]
        if rel in ("EXPRESSED_IN", "HAS_MUTATION"):
            scores[nbr] = scores.get(nbr, 0.0) + DIRECT_WEIGHT

    # Pathway-mediated: Gene -[MEMBER_OF]-> Pathway -[CONTAINS]-> Gene2
    # -[EXPRESSED_IN]-> CellLine
    for p in G.neighbors(seed):
        if G.nodes[p]["type"] != "Pathway" or G.edges[seed, p]["rel"] != "MEMBER_OF":
            continue
        for g2 in G.neighbors(p):
            if G.nodes[g2]["type"] != "Gene" or g2 == seed:
                continue
            if G.edges[p, g2]["rel"] != "CONTAINS":
                continue
            for cl in G.neighbors(g2):
                if G.nodes[cl]["type"] != "CellLine":
                    continue
                if G.edges[g2, cl]["rel"] != "EXPRESSED_IN":
                    continue
                scores[cl] = scores.get(cl, 0.0) + PATHWAY_WEIGHT

    if not scores:
        return empty

    cvcl_to_raw = {n.split(":", 1)[1]: s for n, s in scores.items()}
    raw = pd.Series(cvcl_to_raw)
    return pd.DataFrame({
        "cellosaurus_id": raw.index,
        "metapath_score": raw.rank(pct=True, method="average").values,
    }).reset_index(drop=True)


_PRECOMPUTED_METAPATH: dict[str, dict[str, float]] | None = None


def _load_precomputed_metapath() -> dict[str, dict[str, float]]:
    global _PRECOMPUTED_METAPATH
    if _PRECOMPUTED_METAPATH is not None:
        return _PRECOMPUTED_METAPATH
    if METAPATH_SCORES_FILE.exists():
        with open(METAPATH_SCORES_FILE, encoding="utf-8") as f:
            _PRECOMPUTED_METAPATH = json.load(f)
    else:
        _PRECOMPUTED_METAPATH = {}
    return _PRECOMPUTED_METAPATH


def score_metapath(gene: str) -> pd.DataFrame:
    """
    Metapath-guided score of `gene` across all cell lines. Reads
    outputs/metapath_scores.json (precomputed via precompute_metapath_scores(),
    same pattern as rwr_scores.json) first; falls back to a live
    computation, loudly, if the gene isn't in that file.
    """
    if gene in _METAPATH_CACHE:
        return _METAPATH_CACHE[gene]

    precomputed = _load_precomputed_metapath()
    if gene in precomputed:
        raw = precomputed[gene]
        result = pd.DataFrame({
            "cellosaurus_id": list(raw.keys()),
            "metapath_score": list(raw.values()),
        })
    else:
        print(f"[metapath_scorer] WARNING: {gene} not in {METAPATH_SCORES_FILE} — computing live.")
        result = _compute_metapath_live(gene)

    _METAPATH_CACHE[gene] = result
    return result


def precompute_metapath_scores(genes: list[str], verbose: bool = True) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for i, gene in enumerate(genes, 1):
        if verbose:
            print(f"  [{i}/{len(genes)}] {gene}...")
        df = _compute_metapath_live(gene)
        out[gene] = dict(zip(df["cellosaurus_id"], df["metapath_score"].astype(float)))
    return out


if __name__ == "__main__":
    from src.models.classical.weights_learned import VALIDATION_SET

    print(f"Precomputing metapath scores for {len(VALIDATION_SET)} VALIDATION_SET genes...")
    scores = precompute_metapath_scores(list(VALIDATION_SET.keys()))
    METAPATH_SCORES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(METAPATH_SCORES_FILE, "w", encoding="utf-8") as f:
        json.dump(scores, f)
    print(f"Wrote {METAPATH_SCORES_FILE} ({sum(len(v) for v in scores.values())} gene-cellline entries)")

