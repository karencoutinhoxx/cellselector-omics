# -*- coding: utf-8 -*-
import pandas as pd
from functools import reduce

print("Loading data and lookups...")

lk = pd.read_parquet("outputs/cell_line_lookup.parquet")
hpa_to_id = dict(zip(lk["hpa_name"].dropna(), lk.loc[lk["hpa_name"].notna(),"cellosaurus_id"]))
samp = pd.read_csv("data/nomenclature/9_DepMap_sample_info.csv", low_memory=False)
ach_to_id = dict(zip(samp["DepMap_ID"], samp["RRID"]))
geo_info = pd.read_csv("data/nomenclature/10_GEOInfo.txt", sep="\t", low_memory=False)
gsm_to_id = dict(zip(geo_info["Geo_accession"], geo_info["Cellosaurus_ID"]))

MASTER = pd.read_parquet("outputs/master_with_confidence.parquet")
prop_cols = ["has_mutations","has_fusions"]
MASTER["has_prop"] = MASTER[prop_cols].any(axis=1) if all(c in MASTER for c in prop_cols) else False
CONF = MASTER[["cellosaurus_id","official_name","confidence","has_prop"]].drop_duplicates("cellosaurus_id")
PQ = "outputs/parquet/"


def _score_source(path, gene, id_map, id_col="original_id"):
    df = pd.read_parquet(path, columns=[id_col,"gene_symbol","expression_value"])
    rows = df[df["gene_symbol"] == gene]
    if len(rows) == 0:
        return None
    per = rows.groupby(id_col)["expression_value"].mean().reset_index()
    per["score"] = per["expression_value"] / per["expression_value"].max()
    per["cellosaurus_id"] = per[id_col].map(id_map) if id_map is not None else per[id_col]
    per = per.dropna(subset=["cellosaurus_id"])
    return per.groupby("cellosaurus_id")["score"].max().reset_index()


def recommend(gene, top_n=5):
    parts = []
    for path, idmap, col in [
        (PQ+"gene_expr_hpa_preprocessed.parquet", hpa_to_id, "original_id"),
        (PQ+"gene_expr_depmap_preprocessed.parquet", None, "cellosaurus_id"),
        (PQ+"gene_expr_geo_preprocessed.parquet", gsm_to_id, "original_id"),
        (PQ+"gene_expr_ccle_proteomics_preprocessed.parquet", ach_to_id, "original_id"),
    ]:
        s = _score_source(path, gene, idmap, col)
        if s is not None: parts.append(s.rename(columns={"score":path}))
    if not parts:
        print("No expression data for", gene); return None
    combined = reduce(lambda a,b: a.merge(b, on="cellosaurus_id", how="outer"), parts)
    sc = [c for c in combined.columns if c != "cellosaurus_id"]
    combined["expr_score"] = combined[sc].mean(axis=1, skipna=True)
    combined["n_sources"] = combined[sc].notna().sum(axis=1)
    r = combined.merge(CONF, on="cellosaurus_id", how="left")
    r["final_score"] = r["expr_score"] * r["confidence"]
    r = r.sort_values("final_score", ascending=False).head(top_n)

    print("\n" + "="*70)
    print("TOP CELL LINES FOR " + gene)
    print("="*70)
    for _, row in r.iterrows():
        strength = "very strongly" if row["expr_score"]>=0.7 else "strongly" if row["expr_score"]>=0.4 else "moderately"
        profile = "complete model (expression + mutation data)" if row["has_prop"] else "expression only (no mutation data)"
        print("\n" + str(row["official_name"]) + "  -  fit score " + format(row["final_score"], ".2f"))
        print("   Expresses " + gene + " " + strength + ", across " + str(int(row["n_sources"])) + " of 4 datasets.")
        print("   Profile: " + profile + ".")
        if not row["has_prop"]:
            print("   Note: if you need a mutation model, check a complete-model cell instead.")
    return r


if __name__ == "__main__":
    recommend("EGFR")

