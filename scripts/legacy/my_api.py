"""
Standalone API serving Karen's own recommendation engine.
Returns HPA / DepMap / GEO / Proteomics format with Expression, Confidence,
Sources and Evidence - exactly matching the original dashboard.
Run:  py -m uvicorn my_api:app --port 8000
"""
import os, math
from functools import reduce
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PQ = "outputs/parquet/"

# ---------- Load data once at startup ----------
print("Loading data...")
lk = pd.read_parquet("outputs/cell_line_lookup.parquet")
hpa_to_id = dict(zip(lk["hpa_name"].dropna(), lk.loc[lk["hpa_name"].notna(), "cellosaurus_id"]))
META = lk.set_index("cellosaurus_id")[["disease", "lineage"]].to_dict("index")
samp = pd.read_csv("data/nomenclature/9_DepMap_sample_info.csv", low_memory=False)
ach_to_id = dict(zip(samp["DepMap_ID"], samp["RRID"]))
geo = pd.read_csv("data/nomenclature/10_GEOInfo.txt", sep="\t", low_memory=False)
gsm_to_id = dict(zip(geo["Geo_accession"], geo["Cellosaurus_ID"]))
m = pd.read_parquet("outputs/master_with_confidence.parquet")
keep = [k for k in ["cellosaurus_id", "official_name", "confidence", "evidence_count"] if k in m.columns]
CONF = m[keep].drop_duplicates("cellosaurus_id")
print("Data loaded.")

# ---------- Engine (from the dashboard) ----------
def score_source(path, gene, which):
    idmap = {"hpa": hpa_to_id, "depmap": None, "geo": gsm_to_id, "prot": ach_to_id}[which]
    col = "cellosaurus_id" if which == "depmap" else "original_id"
    df = pd.read_parquet(path, columns=[col, "gene_symbol", "expression_value"])
    rows = df[df["gene_symbol"] == gene]
    if len(rows) == 0:
        return None
    per = rows.groupby(col)["expression_value"].mean().reset_index()
    per["score"] = per["expression_value"] / per["expression_value"].max()
    per["cellosaurus_id"] = per[col].map(idmap) if idmap is not None else per[col]
    per = per.dropna(subset=["cellosaurus_id"])
    return per.groupby("cellosaurus_id")["score"].max().reset_index()

def recommend(gene, disease_filter=None, top_n=10, exclude_genes=None):
    parts = {}
    for path, which in [(PQ+"gene_expr_hpa_preprocessed.parquet","hpa"),
                        (PQ+"gene_expr_depmap_preprocessed.parquet","depmap"),
                        (PQ+"gene_expr_geo_preprocessed.parquet","geo"),
                        (PQ+"gene_expr_ccle_proteomics_preprocessed.parquet","prot")]:
        s = score_source(path, gene, which)
        if s is not None:
            parts[which] = s.rename(columns={"score": which})
    if not parts:
        return None, 0
    c = reduce(lambda a, b: a.merge(b, on="cellosaurus_id", how="outer"), parts.values())
    for w in ["hpa","depmap","geo","prot"]:
        if w not in c:
            c[w] = float("nan")
    c["expr_score"] = c[["hpa","depmap","geo","prot"]].mean(axis=1, skipna=True)
    c["n_sources"] = c[["hpa","depmap","geo","prot"]].notna().sum(axis=1)
    r = c.merge(CONF, on="cellosaurus_id", how="left")
    r["disease"] = r["cellosaurus_id"].map(lambda x: (META.get(x) or {}).get("disease", ""))
    r["lineage"] = r["cellosaurus_id"].map(lambda x: (META.get(x) or {}).get("lineage", ""))
    if disease_filter:
        mask = (r["disease"].fillna("").str.lower().str.contains(disease_filter.lower()) |
                r["lineage"].fillna("").str.lower().str.contains(disease_filter.lower()))
        r = r[mask]
    r["final_score"] = r["expr_score"] * r["confidence"]
    # ---- Exclusion: remove cells that strongly express any excluded gene ----
    if exclude_genes:
        for eg in exclude_genes:
            eg = eg.strip().upper()
            if not eg:
                continue
            # score the excluded gene's expression across cells (reuse score_source)
            excl_parts = []
            for path, which in [(PQ+"gene_expr_hpa_preprocessed.parquet","hpa"),
                                (PQ+"gene_expr_depmap_preprocessed.parquet","depmap"),
                                (PQ+"gene_expr_geo_preprocessed.parquet","geo"),
                                (PQ+"gene_expr_ccle_proteomics_preprocessed.parquet","prot")]:
                es = score_source(path, eg, which)
                if es is not None:
                    excl_parts.append(es.rename(columns={"score": "e_"+which}))
            if excl_parts:
                em = reduce(lambda a, b: a.merge(b, on="cellosaurus_id", how="outer"), excl_parts)
                ecols = [col for col in em.columns if col.startswith("e_")]
                em["excl_expr"] = em[ecols].mean(axis=1, skipna=True)
                em = em[["cellosaurus_id", "excl_expr"]]
                r = r.merge(em, on="cellosaurus_id", how="left")
                # remove cells that express the excluded gene above threshold (0.5)
                r = r[~(r["excl_expr"] > 0.5)]
                r = r.drop(columns=["excl_expr"])
    r = r.sort_values("final_score", ascending=False)
    return r.head(top_n), len(r)

# ---------- API ----------
app = FastAPI(title="CellLineFinder (own engine)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Req(BaseModel):
    gene: str
    disease_filter: str | None = None
    top_n: int = 10
    exclude_genes: list[str] = []

def _f(v):
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return float(v)
    except (ValueError, TypeError):
        return None

@app.get("/health")
def health():
    return {"status": "ok", "engine": "own", "cell_lines": len(CONF)}

@app.post("/recommend/classical")
def recommend_endpoint(req: Req):
    r, total = recommend(req.gene.upper(), req.disease_filter, req.top_n, req.exclude_genes)
    if r is None or len(r) == 0:
        raise HTTPException(404, f"No data for {req.gene}")
    results = []
    for i, (_, row) in enumerate(r.iterrows(), 1):
        evc = row.get("evidence_count")
        evc_out = int(evc) if evc is not None and not (isinstance(evc, float) and math.isnan(evc)) else None
        results.append({
            "rank": i,
            "cellosaurus_id": row.get("cellosaurus_id"),
            "official_name": str(row.get("official_name") or row.get("cellosaurus_id")),
            "final_score": _f(row.get("final_score")),
            "expression_score": _f(row.get("expr_score")),
            "confidence": _f(row.get("confidence")),
            "n_sources": int(row.get("n_sources") or 0),
            "evidence_count": evc_out,
            "hpa": _f(row.get("hpa")),
            "depmap": _f(row.get("depmap")),
            "geo": _f(row.get("geo")),
            "proteomics": _f(row.get("prot")),
            "disease": str(row.get("disease") or ""),
            "lineage": str(row.get("lineage") or ""),
        })
    return {"results": results, "metadata": {"total_candidates": int(total)}}

