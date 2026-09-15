# -*- coding: utf-8 -*-
import pandas as pd
from functools import reduce

print("Loading engine and data...")

lk = pd.read_parquet("outputs/cell_line_lookup.parquet")
hpa_to_id = dict(zip(lk["hpa_name"].dropna(), lk.loc[lk["hpa_name"].notna(),"cellosaurus_id"]))
samp = pd.read_csv("data/nomenclature/9_DepMap_sample_info.csv", low_memory=False)
ach_to_id = dict(zip(samp["DepMap_ID"], samp["RRID"]))
geo_info = pd.read_csv("data/nomenclature/10_GEOInfo.txt", sep="\t", low_memory=False)
gsm_to_id = dict(zip(geo_info["Geo_accession"], geo_info["Cellosaurus_ID"]))
CONF = pd.read_parquet("outputs/master_with_confidence.parquet")[["cellosaurus_id","official_name","confidence"]].drop_duplicates("cellosaurus_id")
PQ = "outputs/parquet/"

def _score(path, gene, idmap, col="original_id"):
    df = pd.read_parquet(path, columns=[col,"gene_symbol","expression_value"])
    rows = df[df["gene_symbol"] == gene]
    if len(rows)==0: return None
    per = rows.groupby(col)["expression_value"].mean().reset_index()
    per["score"] = per["expression_value"]/per["expression_value"].max()
    per["cellosaurus_id"] = per[col].map(idmap) if idmap is not None else per[col]
    per = per.dropna(subset=["cellosaurus_id"])
    return per.groupby("cellosaurus_id")["score"].max().reset_index()

def recommend(gene, top_n=5):
    parts=[]
    for path,idmap,col in [
        (PQ+"gene_expr_hpa_preprocessed.parquet",hpa_to_id,"original_id"),
        (PQ+"gene_expr_depmap_preprocessed.parquet",None,"cellosaurus_id"),
        (PQ+"gene_expr_geo_preprocessed.parquet",gsm_to_id,"original_id"),
        (PQ+"gene_expr_ccle_proteomics_preprocessed.parquet",ach_to_id,"original_id")]:
        s=_score(path,gene,idmap,col)
        if s is not None: parts.append(s.rename(columns={"score":path}))
    if not parts: return None
    c=reduce(lambda a,b:a.merge(b,on="cellosaurus_id",how="outer"),parts)
    sc=[x for x in c.columns if x!="cellosaurus_id"]
    c["expr_score"]=c[sc].mean(axis=1,skipna=True)
    r=c.merge(CONF,on="cellosaurus_id",how="left")
    r["final_score"]=r["expr_score"]*r["confidence"]
    return r.sort_values("final_score",ascending=False).head(top_n)

known = {
    "EGFR":["A-431"],
    "ERBB2":["SK-BR-3","AU565"],
    "MYCN":["IMR-32","Kelly","SK-N-BE(2)"],
    "MET":["EBC-1","Hs 746T"],
    "KIT":["Kasumi-1","GIST882"],
}

print("\n" + "="*60)
print("VALIDATION: does the tool rank known cell lines highly?")
print("="*60)
hits=0; total=0
for gene, expected in known.items():
    r = recommend(gene, top_n=5)
    total += 1
    if r is None:
        print(f"\n{gene}: no data"); continue
    top_names = r["official_name"].tolist()
    found = [e for e in expected if e in top_names]
    mark = "HIT" if found else "miss"
    if found: hits += 1
    print(f"\n{gene}: [{mark}]")
    print(f"   expected one of: {expected}")
    print(f"   tool top 5:      {top_names}")
    if found:
        rank = top_names.index(found[0]) + 1
        print(f"   -> found {found[0]} at rank {rank}")

print("\n" + "="*60)
print(f"RESULT: {hits} of {total} genes had a known cell line in the top 5")
print("="*60)

