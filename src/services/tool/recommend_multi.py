import pandas as pd
from functools import reduce

print("Loading data and lookups...")

lk = pd.read_parquet("outputs/cell_line_lookup.parquet")
hpa_to_id = dict(zip(lk["hpa_name"].dropna(), lk.loc[lk["hpa_name"].notna(),"cellosaurus_id"]))
samp = pd.read_csv("data/nomenclature/9_DepMap_sample_info.csv", low_memory=False)
ach_to_id = dict(zip(samp["DepMap_ID"], samp["RRID"]))
geo_info = pd.read_csv("data/nomenclature/10_GEOInfo.txt", sep="\t", low_memory=False)
gsm_to_id = dict(zip(geo_info["Geo_accession"], geo_info["Cellosaurus_ID"]))

CONF = pd.read_parquet("outputs/master_with_confidence.parquet")[["cellosaurus_id","official_name","confidence"]].drop_duplicates("cellosaurus_id")
PQ = "outputs/parquet/"


def _score_source(path, gene, id_map, id_col="original_id"):
    df = pd.read_parquet(path, columns=[id_col,"gene_symbol","expression_value"])
    rows = df[df["gene_symbol"] == gene]
    if len(rows) == 0:
        return None
    per = rows.groupby(id_col)["expression_value"].mean().reset_index()
    per["score"] = per["expression_value"] / per["expression_value"].max()
    if id_map is not None:
        per["cellosaurus_id"] = per[id_col].map(id_map)
    else:
        per["cellosaurus_id"] = per[id_col]
    per = per.dropna(subset=["cellosaurus_id"])
    return per.groupby("cellosaurus_id")["score"].max().reset_index()


def make_reason(gene, es, ns, cf):
    strength = "very strongly" if es>=0.7 else "strongly" if es>=0.4 else "moderately" if es>=0.2 else "weakly"
    conf_word = "high confidence" if cf>=0.8 else "moderate confidence" if cf>=0.5 else "lower confidence"
    return f"Expresses {gene} {strength}, backed by {int(ns)} dataset(s) ({conf_word})."


def recommend(gene, top_n=10):
    parts = []
    h = _score_source(PQ+"gene_expr_hpa_preprocessed.parquet", gene, hpa_to_id)
    if h is not None: parts.append(h.rename(columns={"score":"hpa"}))
    d = _score_source(PQ+"gene_expr_depmap_preprocessed.parquet", gene, None, id_col="cellosaurus_id")
    if d is not None: parts.append(d.rename(columns={"score":"depmap"}))
    g = _score_source(PQ+"gene_expr_geo_preprocessed.parquet", gene, gsm_to_id)
    if g is not None: parts.append(g.rename(columns={"score":"geo"}))
    p = _score_source(PQ+"gene_expr_ccle_proteomics_preprocessed.parquet", gene, ach_to_id)
    if p is not None: parts.append(p.rename(columns={"score":"proteomics"}))
    if not parts:
        print("No expression data for", gene)
        return None
    combined = reduce(lambda a,b: a.merge(b, on="cellosaurus_id", how="outer"), parts)
    score_cols = [c for c in combined.columns if c != "cellosaurus_id"]
    combined["expr_score"] = combined[score_cols].mean(axis=1, skipna=True)
    combined["n_sources"] = combined[score_cols].notna().sum(axis=1)
    result = combined.merge(CONF, on="cellosaurus_id", how="left")
    result["final_score"] = result["expr_score"] * result["confidence"]
    result = result.sort_values("final_score", ascending=False).head(top_n)
    result["reason"] = result.apply(lambda r: make_reason(gene, r["expr_score"], r["n_sources"], r["confidence"]), axis=1)
    return result[["official_name","final_score","reason"]]


if __name__ == "__main__":
    for gene in ["EGFR"]:
        print("\n" + "="*70)
        print("Top cell lines for", gene)
        r = recommend(gene)
        if r is not None:
            for _, row in r.iterrows():
                print(f"\n{row['official_name']}  (score {row['final_score']:.2f})")
                print(f"   {row['reason']}")

