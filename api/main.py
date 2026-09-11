"""
CellSelector Omics — FastAPI backend.

Run:  uvicorn api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
import asyncio
import io
import json
import math
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
import sys

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CELL_LINE_LOOKUP, MASTER_MERGED, OUTPUTS_DIR, PARQUET_DIR
from api.models import AgenticRequest, ClassicalRequest
from models.agentic.growth_properties import get_growth_properties
from models.classical.ranker import FIXED_WEIGHTS, rank
from models.classical.scorer import classify_gene, get_gene_role
from models.classical.similarity import DATASET_CITATIONS, find_alternatives
from models.graph.queries import (
    graph_to_json,
    query_best_cell_lines_via_pathway,
    query_pathway_neighbor_genes,
)

# ── Source parquets (gene search) ─────────────────────────────────────────────
_PARQUET_SOURCES: dict[str, Path] = {
    "hpa":        PARQUET_DIR / "gene_expr_hpa_preprocessed.parquet",
    "depmap":     PARQUET_DIR / "gene_expr_depmap_preprocessed.parquet",
    "geo":        PARQUET_DIR / "gene_expr_geo_preprocessed.parquet",
    "proteomics": PARQUET_DIR / "gene_expr_ccle_proteomics_preprocessed.parquet",
}

# ── Session store (keyed by UUID, max 100 entries) ────────────────────────────
_MAX_SESSIONS = 100
_session_store: OrderedDict = OrderedDict()


def _store_session(data: dict) -> str:
    sid = str(uuid.uuid4())
    _session_store[sid] = data
    while len(_session_store) > _MAX_SESSIONS:
        _session_store.popitem(last=False)
    return sid


# ── Learned-weight cache ──────────────────────────────────────────────────────
# Cached per gene class, not one global set — pathway activity's value
# genuinely varies by class (see weights_learned.optimise_weights_by_class),
# so the weights actually applied to a query depend on classify_gene(gene).
_learned_weights_by_class_cache: dict | None = None
_learned_weights_lock = asyncio.Lock()


# Precomputed once offline (see the regeneration command in
# _warm_learned_weights' docstring below) so startup doesn't pay the
# ~8 minute optimise_weights_by_class(include_mutation=True) cost.
_PRODUCTION_WEIGHTS_FILE = OUTPUTS_DIR / "production_weights.json"


async def _warm_learned_weights() -> dict:
    """
    Populate BOTH learned-weight caches that read this — this module's own,
    and ranker's _CACHED_LEARNED_WEIGHTS_BY_CLASS (the agentic pipeline
    calls rank() with no weights, which reads ranker's cache directly, not
    this one) — from outputs/production_weights.json, a near-instant file
    read.

    ⚠️ REGENERATION REQUIRED: outputs/production_weights.json is a frozen
    snapshot of optimise_weights_by_class(include_mutation=True). It MUST be
    regenerated any time weights_learned.VALIDATION_SET changes (genes
    added/removed/reclassified) — otherwise the API silently serves stale
    weights fit on an old gene set. Regenerate with:

        python3 -c "
        from models.classical.weights_learned import optimise_weights_by_class
        import json
        weights = optimise_weights_by_class(include_mutation=True)
        clean = {cls: ({k: float(v) for k, v in w.items()} if w else None)
                 for cls, w in weights.items()}
        with open('outputs/production_weights.json', 'w') as f:
            json.dump(clean, f, indent=2)
        "

    then commit the updated file.

    Falls back to computing live (~8 min, blocking startup) only if the
    file is missing — loudly, not silently — so a forgotten regeneration
    degrades to correct-but-slow rather than crashing.
    """
    global _learned_weights_by_class_cache
    async with _learned_weights_lock:
        if _learned_weights_by_class_cache is None:
            import models.classical.ranker as _ranker

            if _PRODUCTION_WEIGHTS_FILE.exists():
                with open(_PRODUCTION_WEIGHTS_FILE, encoding="utf-8") as f:
                    weights = json.load(f)
            else:
                print(f"[startup] WARNING: {_PRODUCTION_WEIGHTS_FILE} not found — "
                      f"computing weights live (~8 min). Run the precompute "
                      f"script (see _warm_learned_weights docstring) and commit "
                      f"the file to avoid this on every cold start.")
                from functools import partial

                from models.classical.weights_learned import optimise_weights_by_class

                weights = await asyncio.to_thread(
                    partial(optimise_weights_by_class, include_mutation=True)
                )

            _learned_weights_by_class_cache = weights
            _ranker._CACHED_LEARNED_WEIGHTS_BY_CLASS = weights
    return _learned_weights_by_class_cache


async def _get_learned_weights_for_gene(gene: str) -> dict:
    by_class = _learned_weights_by_class_cache or await _warm_learned_weights()
    gene_class = classify_gene(gene)
    return by_class.get(gene_class) or by_class["tissue_specific"]


# ── Lifespan: load heavy data once ───────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.master_merged    = pd.read_parquet(MASTER_MERGED)
    app.state.cell_line_lookup = pd.read_parquet(CELL_LINE_LOOKUP)

    def _load_gene_set(path: Path) -> set:
        df = pd.read_parquet(path, columns=["gene_symbol"])
        # .unique() deduplicates in C before Python set() sees the values
        return set(df["gene_symbol"].dropna().unique())

    gene_sets: dict[str, set] = {}
    tasks = {
        name: asyncio.to_thread(_load_gene_set, path)
        for name, path in _PARQUET_SOURCES.items()
        if path.exists()
    }
    for name in _PARQUET_SOURCES:
        if name in tasks:
            gene_sets[name] = await tasks[name]
        else:
            gene_sets[name] = set()
    app.state.gene_sets = gene_sets

    eval_path = OUTPUTS_DIR / "model_evaluation.json"
    if eval_path.exists():
        with open(eval_path, encoding="utf-8") as f:
            app.state.eval_stats = json.load(f)
    else:
        app.state.eval_stats = None

    # Warm the per-gene-class learned weights BEFORE accepting requests.
    # Near-instant: loads the precomputed outputs/production_weights.json
    # (see _warm_learned_weights' docstring) rather than re-running the
    # ~8 min SLSQP + score precompute live.
    print("[startup] Loading per-gene-class learned weights...")
    _t0 = time.time()
    await _warm_learned_weights()
    print(f"[startup] Learned weights ready in {time.time() - _t0:.0f}s — accepting requests")

    yield


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CellSelector Omics API",
    description="Multi-omics cell line recommendation system for AstraZeneca",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. GET /health
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health(request: Request):
    return {
        "status":     "ok",
        "version":    "1.0.0",
        "cell_lines": len(request.app.state.master_merged),
        "datasets":   4,
        "models":     ["classical", "agentic"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. GET /genes/search?q=EGFR
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/genes/search")
async def search_gene(q: str = Query(..., min_length=1), request: Request = None):
    gene      = q.strip().upper()
    gene_sets = request.app.state.gene_sets
    sources   = {name: (gene in gs) for name, gs in gene_sets.items()}
    found     = any(sources.values())

    total_cl = 0
    if found:
        full = await asyncio.to_thread(rank, gene, None, None, None)
        total_cl = len(full)

    return {
        "gene":                    gene,
        "found":                   found,
        "sources":                 sources,
        "total_cell_lines_with_data": total_cl,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apply_exclusion_penalties(
    df: pd.DataFrame,
    exclude_genes: list[str],
) -> tuple[pd.DataFrame, dict]:
    """
    Apply multiplicative RNA-expression penalties for excluded genes.
    Returns (modified_df, {excl_gene: {cvcl: score}}).
    Penalty formula: final_score *= (1 - excl_rna_score * 0.5)
    """
    if not exclude_genes:
        return df, {}

    from models.classical.scorer import load_mappings, score_rna_expression
    hpa_to_cvcl, _, gsm_to_cvcl = load_mappings()

    excl_scores: dict[str, dict] = {}
    df = df.copy()
    for excl_gene in exclude_genes:
        excl_rna = score_rna_expression(excl_gene, hpa_to_cvcl, gsm_to_cvcl)
        cvcl_map = dict(zip(excl_rna["cellosaurus_id"], excl_rna["rna_score"]))
        excl_scores[excl_gene] = cvcl_map

        col = f"_excl_{excl_gene}"
        df[col] = df["cellosaurus_id"].map(lambda c: float(cvcl_map.get(c, 0.0)))
        df["final_score"] = (df["final_score"] * (1 - df[col] * 0.5)).clip(0.0, 1.0)

    return df.sort_values("final_score", ascending=False), excl_scores


def _safe_float(v) -> float:
    try:
        f = float(v)
        return 0.0 if math.isnan(f) else f
    except (TypeError, ValueError):
        return 0.0


def _sanitize_for_json(obj):
    """Recursively replace NaN/Inf floats with None so JSON serialisation never fails."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _build_classical_result(
    rank_pos: int,
    row: pd.Series,
    gene: str,
    alts: list[dict],
    excl_scores: dict,
    exclude_genes: list[str],
) -> dict:
    cvcl = row["cellosaurus_id"]

    exclusion_warnings = []
    for excl_gene in exclude_genes:
        score = _safe_float(row.get(f"_excl_{excl_gene}", 0.0))
        if score > 0.5:
            exclusion_warnings.append({
                "gene":    excl_gene,
                "score":   round(score, 3),
                "message": (
                    f"{excl_gene} also expressed (score {score:.2f}) — "
                    f"may confound {gene} experiments"
                ),
            })

    return {
        "rank":               rank_pos,
        "cellosaurus_id":     cvcl,
        "official_name":      str(row.get("official_name") or cvcl),
        "final_score":        round(_safe_float(row.get("final_score")), 4),
        "rna_score":          round(_safe_float(row.get("rna_score")), 4),
        "protein_score":      round(_safe_float(row.get("protein_score")), 4),
        "quality_score":      round(_safe_float(row.get("quality_score")), 4),
        "context_score":      round(_safe_float(row.get("context_score")), 4),
        "geo_confirmation":   round(_safe_float(row.get("geo_confirmation")), 4),
        "pathway_activity_score":  round(_safe_float(row.get("pathway_activity_score")), 4),
        "pathway_genes_expressed": int(row.get("pathway_genes_expressed") or 0),
        "pathway_genes_total":     int(row.get("pathway_genes_total") or 0),
        "mutation_impact_score":   round(_safe_float(row.get("mutation_impact_score")), 4),
        "mutation_detail":         str(row.get("mutation_detail") or ""),
        "n_sources":          int(row.get("n_sources") or 0),
        "hpa_evidence":       row.get("hpa_evidence"),
        "depmap_evidence":    row.get("depmap_evidence"),
        "geo_evidence":       row.get("geo_evidence"),
        "protein_evidence":   row.get("protein_evidence"),
        "vs_next_rank":       row.get("vs_next_rank"),
        "quality_explanation": row.get("quality_explanation") or "",
        "context_explanation": row.get("context_explanation") or "",
        "gene_class":         row.get("gene_class"),
        "gene_role":          get_gene_role(gene),
        "growth_properties":  get_growth_properties(cvcl),
        "disease":            str(row.get("disease") or ""),
        "lineage":            str(row.get("lineage") or ""),
        "exclusion_warnings": exclusion_warnings,
        "alternatives": [
            {
                "official_name":     a["official_name"],
                "cellosaurus_id":    a["cellosaurus_id"],
                "similarity_score":  a["similarity_score"],
                "similarity_reason": a["similarity_reason"],
                "shared_data_types": a.get("shared_data_types", []),
                "note":              a.get("note"),
                "cellosaurus_url":   (
                    a.get("citations", {}).get("cellosaurus_url")
                    or f"https://www.cellosaurus.org/{a['cellosaurus_id']}"
                ),
            }
            for a in alts
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. POST /recommend/classical
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/recommend/classical")
async def recommend_classical(body: ClassicalRequest):
    if body.exclude_genes and body.gene.upper() in [g.upper() for g in body.exclude_genes]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot search for {body.gene} and exclude it simultaneously.",
        )

    t0 = time.time()

    # LOF genes always use the mutation-aware learned weights inside rank()
    # regardless of use_learned_weights — fetch them here too so weights_used
    # in the response reflects what actually ran.
    if body.use_learned_weights or classify_gene(body.gene) == "loss_of_function":
        weights = await _get_learned_weights_for_gene(body.gene)
    else:
        weights = FIXED_WEIGHTS

    # Run with top_n=None to capture total candidate count
    all_ranked = await asyncio.to_thread(
        rank, body.gene, body.disease_filter, body.lineage_filter, None, weights, body.use_learned_weights
    )

    if all_ranked is None or len(all_ranked) == 0:
        raise HTTPException(404, f"No expression data found for gene: {body.gene}")

    total_candidates = len(all_ranked)

    if body.exclude_genes:
        all_ranked, excl_scores = await asyncio.to_thread(
            _apply_exclusion_penalties, all_ranked, body.exclude_genes
        )
    else:
        excl_scores = {}

    top_results = all_ranked.head(body.top_n)

    try:
        alts_map = await asyncio.to_thread(
            find_alternatives,
            body.gene, top_results, MASTER_MERGED, 3,
            body.disease_filter, body.lineage_filter,
        )
    except Exception as exc:
        print(f"[warn] similarity: {exc}")
        alts_map = {}

    results = [
        _build_classical_result(
            i + 1, row, body.gene,
            alts_map.get(row["cellosaurus_id"], []),
            excl_scores, body.exclude_genes,
        )
        for i, (_, row) in enumerate(top_results.iterrows())
    ]

    execution_ms = int((time.time() - t0) * 1000)

    response = {
        "query":   body.model_dump(),
        "model":   "classical",
        "results": results,
        "weights_used": weights,
        "metadata": {
            "total_candidates":  total_candidates,
            "execution_time_ms": execution_ms,
        },
    }
    response = _sanitize_for_json(response)
    response["session_id"] = _store_session(response)
    return response


# ─────────────────────────────────────────────────────────────────────────────
# 4. POST /recommend/agentic
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/recommend/agentic")
async def recommend_agentic(body: AgenticRequest):
    if body.exclude_genes and body.gene.upper() in [g.upper() for g in body.exclude_genes]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot search for {body.gene} and exclude it simultaneously.",
        )

    from models.agentic.pipeline import run as pipeline_run

    t0 = time.time()

    pipeline_out = await asyncio.to_thread(
        pipeline_run,
        body.gene, body.disease_filter, body.lineage_filter, body.top_n,
        body.target_cellosaurus_id, body.exclude_genes or None,
    )

    # Reconstruct a minimal DataFrame for find_alternatives (needs cellosaurus_id column)
    pipe_results = pipeline_out.get("results", [])
    ranked_for_sim = pd.DataFrame({
        "cellosaurus_id": [r["cellosaurus_id"] for r in pipe_results]
    })

    try:
        alts_map = await asyncio.to_thread(
            find_alternatives,
            body.gene, ranked_for_sim, MASTER_MERGED, 3,
            body.disease_filter, body.lineage_filter,
        )
    except Exception as exc:
        print(f"[warn] similarity: {exc}")
        alts_map = {}

    results = []
    for r in pipe_results:
        cvcl = r.get("cellosaurus_id", "")
        alts = alts_map.get(cvcl, [])
        results.append({
            **r,
            "growth_properties": get_growth_properties(cvcl),
            "alternatives": [
                {
                    "official_name":     a["official_name"],
                    "cellosaurus_id":    a["cellosaurus_id"],
                    "similarity_score":  a["similarity_score"],
                    "similarity_reason": a["similarity_reason"],
                    "shared_data_types": a.get("shared_data_types", []),
                    "note":              a.get("note"),
                    "cellosaurus_url":   (
                        a.get("citations", {}).get("cellosaurus_url")
                        or f"https://www.cellosaurus.org/{a['cellosaurus_id']}"
                    ),
                }
                for a in alts
            ],
        })

    execution_ms = int((time.time() - t0) * 1000)

    response = {
        "query":   body.model_dump(),
        "model":   "agentic",
        "results": results,
        "comparative_summary": pipeline_out.get("comparative_summary", ""),
        "dataset_citations":   list(DATASET_CITATIONS.values()),
        "metadata": {
            "execution_time_ms":     execution_ms,
            "llm_model":             body.ollama_model,
            "pubmed_papers_retrieved": sum(
                len(r.get("literature", []))
                for r in results
                if isinstance(r.get("literature"), list)
            ),
        },
    }
    response = _sanitize_for_json(response)
    response["session_id"] = _store_session(response)

    resp = JSONResponse(content=response)
    resp.headers["X-Processing-Time"] = str(execution_ms)
    return resp


# ─────────────────────────────────────────────────────────────────────────────
# 5. GET /recommend/export/{format}
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/recommend/export/{fmt}")
async def export_results(
    fmt: str,
    session_id: str = Query(..., description="session_id returned by /recommend endpoints"),
):
    if session_id not in _session_store:
        raise HTTPException(404, "Session not found. Run /recommend/classical or /recommend/agentic first.")

    data    = _session_store[session_id]
    results = data.get("results", [])
    gene    = data.get("query", {}).get("gene", "unknown")
    fmt     = fmt.lower()

    # ── JSON ──────────────────────────────────────────────────────────────────
    if fmt == "json":
        content = json.dumps(data, indent=2, ensure_ascii=False)
        return StreamingResponse(
            io.BytesIO(content.encode()),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="cellselector_{gene}.json"'},
        )

    # ── CSV ───────────────────────────────────────────────────────────────────
    elif fmt == "csv":
        rows = []
        for r in results:
            alts = r.get("alternatives", [])
            sc   = r.get("scores", r)  # agentic nests scores; classical has them flat
            rows.append({
                "rank":             r.get("rank"),
                "cellosaurus_id":   r.get("cellosaurus_id"),
                "official_name":    r.get("official_name"),
                "final_score":      sc.get("final_score", r.get("final_score")),
                "rna_score":        sc.get("rna_score",   r.get("rna_score")),
                "protein_score":    sc.get("protein_score", r.get("protein_score")),
                "quality_score":    sc.get("quality_score", r.get("quality_score")),
                "disease":          r.get("disease", ""),
                "lineage":          r.get("lineage", ""),
                "n_sources":        r.get("n_sources", ""),
                "gene_class":       r.get("gene_class", ""),
                "top_alternative":  alts[0]["official_name"]   if alts else "",
                "similarity_score": alts[0]["similarity_score"] if alts else "",
            })
        df  = pd.DataFrame(rows)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return StreamingResponse(
            io.BytesIO(buf.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="cellselector_{gene}.csv"'},
        )

    # ── PDF ───────────────────────────────────────────────────────────────────
    elif fmt == "pdf":
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.units import cm
            from reportlab.platypus import (
                Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
            )
        except ImportError:
            raise HTTPException(500, "reportlab not installed. Run: pip install reportlab")

        buf  = io.BytesIO()
        doc  = SimpleDocTemplate(buf, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
        styl = getSampleStyleSheet()
        body = []

        body.append(Paragraph("CellSelector Omics — Recommendation Report", styl["Title"]))
        body.append(Spacer(1, 0.4 * cm))

        q = data.get("query", {})
        for line in [
            f"<b>Gene:</b> {q.get('gene', '')}",
            f"<b>Disease filter:</b> {q.get('disease_filter') or '—'}",
            f"<b>Lineage filter:</b> {q.get('lineage_filter') or '—'}",
            f"<b>Model:</b> {data.get('model', '')}",
            f"<b>Results:</b> {q.get('top_n', '')} requested",
        ]:
            body.append(Paragraph(line, styl["Normal"]))
        body.append(Spacer(1, 0.5 * cm))

        body.append(Paragraph("Results", styl["Heading2"]))
        tbl_data = [["#", "Cell Line", "Score", "RNA", "Protein", "Quality", "Disease"]]
        for r in results:
            sc = r.get("scores", r)
            tbl_data.append([
                str(r.get("rank", "")),
                str(r.get("official_name", ""))[:32],
                f"{_safe_float(sc.get('final_score', r.get('final_score'))):.3f}",
                f"{_safe_float(sc.get('rna_score',   r.get('rna_score'))):.3f}",
                f"{_safe_float(sc.get('protein_score', r.get('protein_score'))):.3f}",
                f"{_safe_float(sc.get('quality_score', r.get('quality_score'))):.3f}",
                str(r.get("disease", ""))[:28],
            ])

        tbl = Table(tbl_data, repeatRows=1, hAlign="LEFT")
        tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0),  colors.HexColor("#0F172A")),
            ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
            ("FONTSIZE",     (0, 0), (-1, 0),  9),
            ("FONTSIZE",     (0, 1), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
            ("ALIGN",        (1, 0), (1, -1),  "LEFT"),
            ("TOPPADDING",   (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ]))
        body.append(tbl)
        body.append(Spacer(1, 0.5 * cm))

        # Top result justification (agentic only)
        if results and results[0].get("justification"):
            body.append(Paragraph("Top Result — AI Justification", styl["Heading2"]))
            just = results[0]["justification"]
            for line in (just if isinstance(just, str) else json.dumps(just)).split("\n"):
                if line.strip():
                    body.append(Paragraph(line.strip()[:200], styl["Normal"]))
            body.append(Spacer(1, 0.4 * cm))

        body.append(Paragraph("Data Sources", styl["Heading2"]))
        for k, v in DATASET_CITATIONS.items():
            body.append(Paragraph(f"[{k}] {v['name']}: {v['citation']}", styl["Normal"]))

        doc.build(body)
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="cellselector_{gene}.pdf"'},
        )

    else:
        raise HTTPException(400, f"Unsupported format '{fmt}'. Use: json, csv, pdf")


# ─────────────────────────────────────────────────────────────────────────────
# 6. GET /cell-lines  (list / search)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/cell-lines")
def list_cell_lines(
    search: str | None = Query(None, description="Filter by name, disease or lineage"),
    limit: int = Query(100, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    request: Request = None,
):
    master = request.app.state.master_merged
    lookup = request.app.state.cell_line_lookup

    merged = master.merge(
        lookup[["cellosaurus_id", "disease", "lineage"]],
        on="cellosaurus_id",
        how="left",
    )
    for col in ("has_mutations", "has_fusions"):
        if col not in merged.columns:
            merged[col] = False

    merged = merged.drop_duplicates("cellosaurus_id")

    if search:
        s = search.strip().lower()
        mask = (
            merged["official_name"].fillna("").str.lower().str.contains(s, regex=False)
            | merged["disease"].fillna("").str.lower().str.contains(s, regex=False)
            | merged["lineage"].fillna("").str.lower().str.contains(s, regex=False)
        )
        merged = merged[mask]

    total   = int(len(merged))
    page_df = merged.iloc[offset : offset + limit]

    def _bool(v) -> bool:
        try:
            return bool(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else False
        except Exception:
            return False

    def _clean(v) -> str:
        s = str(v) if v is not None else ""
        return "" if s in ("nan", "None", "NaN") else s.strip()

    results = [
        {
            "cellosaurus_id":  _clean(row.get("cellosaurus_id")),
            "official_name":   _clean(row.get("official_name")),
            "disease":         _clean(row.get("disease")),
            "lineage":         _clean(row.get("lineage")),
            "confidence":      round(min(1.0, max(0.0, _safe_float(row.get("confidence")))), 3),
            "evidence_count":  int(row.get("evidence_count") or 0),
            "has_hpa_expr":    _bool(row.get("has_hpa_expr")),
            "has_depmap_expr": _bool(row.get("has_depmap_expr")),
            "has_geo_expr":    _bool(row.get("has_geo_expr")),
            "has_proteomics":  _bool(row.get("has_proteomics")),
            "has_mutations":   _bool(row.get("has_mutations")),
            "has_fusions":     _bool(row.get("has_fusions")),
        }
        for _, row in page_df.iterrows()
    ]

    return {"total": total, "results": results}


# ─────────────────────────────────────────────────────────────────────────────
# 7. GET /cell-lines/{cellosaurus_id}
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/cell-lines/{cellosaurus_id}")
def cell_line_profile(cellosaurus_id: str, request: Request):
    master = request.app.state.master_merged
    lookup = request.app.state.cell_line_lookup

    row = master[master["cellosaurus_id"] == cellosaurus_id]
    if len(row) == 0:
        raise HTTPException(404, f"Cell line {cellosaurus_id} not found")
    row = row.iloc[0]

    lkp_row = lookup[lookup["cellosaurus_id"] == cellosaurus_id]

    def _str_or_none(df_row, col):
        if len(df_row) == 0 or col not in df_row.columns:
            return None
        v = str(df_row[col].iloc[0])
        return None if v in ("nan", "", "None") else v

    coverage_cols = [
        "has_hpa_expr", "has_depmap_expr", "has_geo_expr", "has_proteomics",
        "has_metabolomics", "has_mirna", "has_mutations", "has_fusions",
    ]
    genomic_cols = ["MSIScore", "Ploidy", "CIN", "Aneuploidy"]

    def _nullable_float(v):
        try:
            f = float(v)
            return None if math.isnan(f) else round(f, 4)
        except (TypeError, ValueError):
            return None

    return {
        "cellosaurus_id": cellosaurus_id,
        "official_name":  str(row.get("official_name") or cellosaurus_id),
        "disease":        _str_or_none(lkp_row, "disease"),
        "lineage":        _str_or_none(lkp_row, "lineage"),
        "evidence_count": int(row.get("evidence_count") or 0),
        "data_coverage":  {c: bool(row.get(c, False)) for c in coverage_cols},
        "genomic_features": {
            c: _nullable_float(row.get(c)) for c in genomic_cols
        },
        "cellosaurus_url": f"https://www.cellosaurus.org/{cellosaurus_id}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. GET /stats
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/stats")
def stats(request: Request):
    master     = request.app.state.master_merged
    eval_stats = request.app.state.eval_stats

    coverage_cols = [
        "has_hpa_expr", "has_depmap_expr", "has_geo_expr", "has_proteomics",
        "has_metabolomics", "has_mirna", "has_mutations", "has_fusions",
    ]
    coverage = {
        col: int(master[col].fillna(False).astype(bool).sum())
        if col in master.columns else 0
        for col in coverage_cols
    }

    validation = None
    if eval_stats:
        validation = {
            "mrr_fixed":        eval_stats.get("fixed_mrr"),
            "mrr_learned":      eval_stats.get("learned_mrr"),
            "hit_at_10":        eval_stats.get("fixed_hit_rate_at_10"),
            "validation_genes": eval_stats.get("n_genes", 25),
        }

    return {
        "total_cell_lines": len(master),
        "data_coverage":    coverage,
        "validation":       validation,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 9. Knowledge graph — multi-hop queries over gene/pathway/cell_line nodes,
# backed by Neo4j (models/graph/ingest.py populates it; see that module's
# docstring — it's a one-time/periodic ETL, not run per request). Each
# request is still a network round-trip to Aura, so it still runs off the
# event loop thread via asyncio.to_thread, matching /recommend/classical.
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/graph/explore/{gene}")
async def explore_graph(gene: str, disease_filter: str = Query(None)):
    """
    Full knowledge graph subgraph for a gene as nodes+edges JSON,
    for visualization.

    disease_filter is accepted but currently ignored — TODO: CellLine nodes
    now carry disease/lineage (see ingest.py), graph_to_json() just doesn't
    filter on it yet. query_best_cell_lines_via_pathway() does.
    """
    try:
        result = await asyncio.to_thread(graph_to_json, gene.strip().upper())
        return _sanitize_for_json(result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/graph/pathway-neighbors/{gene}")
async def pathway_neighbors(gene: str):
    """
    Genuine 2-hop graph query: genes sharing a pathway with the target gene.
    """
    try:
        result = await asyncio.to_thread(query_pathway_neighbor_genes, gene.strip().upper())
        return _sanitize_for_json({
            "gene":      gene.strip().upper(),
            "neighbors": result,
            "count":     len(result),
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/graph/cell-lines-via-pathway/{gene}")
async def cell_lines_via_pathway(
    gene: str,
    disease_filter: str = Query(None),
    top_k: int = Query(5),
):
    """
    Multi-hop query: best cell lines connected to gene directly OR via
    pathway-neighbor genes, optionally restricted to a disease
    (case-insensitive substring match against CellLine.disease).
    """
    try:
        result = await asyncio.to_thread(
            query_best_cell_lines_via_pathway, gene.strip().upper(), disease_filter, top_k
        )
        return _sanitize_for_json({
            "gene":    gene.strip().upper(),
            "results": result,
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
