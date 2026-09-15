import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import (
    CELLOSAURUS,
    CELL_LINE_LOOKUP,
    GEO_INFO,
    HPA_DESC,
    OMICS_PROFILES,
    SAMPLE_INFO,
)

print("Loading all sources...")
samp  = pd.read_csv(SAMPLE_INFO,  low_memory=False)
hpa   = pd.read_csv(HPA_DESC,     sep="\t", low_memory=False)
geo   = pd.read_csv(GEO_INFO,     sep="\t", low_memory=False)
cello = pd.read_csv(CELLOSAURUS,  low_memory=False)

# ---- Collect every cell line ID from all three sources ----
depmap_ids = set(samp["RRID"].dropna())
hpa_ids = set(hpa["Cellosaurus ID"].dropna())
geo_ids = set(geo["Cellosaurus_ID"].dropna())
all_ids = depmap_ids | hpa_ids | geo_ids
print("Total unique cell lines across all sources:", len(all_ids))

# ---- Base table: one row per unique cell line ----
m = pd.DataFrame({"cellosaurus_id": sorted(all_ids)})
m["in_depmap"] = m["cellosaurus_id"].isin(depmap_ids)
m["in_hpa"] = m["cellosaurus_id"].isin(hpa_ids)
m["in_geo"] = m["cellosaurus_id"].isin(geo_ids)
m["evidence_count"] = m[["in_depmap","in_hpa","in_geo"]].sum(axis=1)

# ---- Add DepMap details ----
dep = samp[["RRID","DepMap_ID","stripped_cell_line_name","primary_disease","lineage"]]
dep = dep.rename(columns={"RRID":"cellosaurus_id","stripped_cell_line_name":"depmap_name","primary_disease":"disease"})
dep = dep.dropna(subset=["cellosaurus_id"]).drop_duplicates("cellosaurus_id")
m = m.merge(dep, on="cellosaurus_id", how="left")

# ---- Add HPA name ----
hn = hpa[["Cellosaurus ID","Cell line"]].rename(columns={"Cellosaurus ID":"cellosaurus_id","Cell line":"hpa_name"})
hn = hn.dropna(subset=["cellosaurus_id"]).drop_duplicates("cellosaurus_id")
m = m.merge(hn, on="cellosaurus_id", how="left")

# ---- Add GEO name ----
gn = geo[["Cellosaurus_ID","cell_line"]].rename(columns={"Cellosaurus_ID":"cellosaurus_id","cell_line":"geo_name"})
gn = gn.dropna(subset=["cellosaurus_id"]).drop_duplicates("cellosaurus_id")
m = m.merge(gn, on="cellosaurus_id", how="left")

# ---- Add official name, synonyms, and human check from Cellosaurus ----
cs = cello[["Accession (CVCL_xxxx)","Identifier (cell line name)","Synonyms","Species of origin"]]
cs = cs.rename(columns={
    "Accession (CVCL_xxxx)":"cellosaurus_id",
    "Identifier (cell line name)":"official_name",
    "Synonyms":"synonyms",
    "Species of origin":"species",
})
m = m.merge(cs, on="cellosaurus_id", how="left")
m["is_human"] = m["species"].astype(str).str.contains("Homo sapiens", na=False)

# ---- Add data-type map from file 8 ----
print("Loading OmicsProfiles (file 8)...")
prof = pd.read_csv(OMICS_PROFILES, low_memory=False)
prof["has"] = True
datatypes = prof.pivot_table(index="ModelID", columns="Datatype", values="has", aggfunc="any", fill_value=False)
datatypes = datatypes.reset_index().rename(columns={"ModelID":"DepMap_ID","rna":"has_rna","wes":"has_wes","wgs":"has_wgs"})
m = m.merge(datatypes, on="DepMap_ID", how="left")
m["datatype_count"] = m[["has_rna","has_wes","has_wgs"]].sum(axis=1)

# ---- Save ----
m.to_parquet(CELL_LINE_LOOKUP)
print("\nFinal table:", m.shape[0], "cell lines,", m.shape[1], "columns")
print("Matched to Cellosaurus:", m["official_name"].notna().sum())
print("Human cell lines:", m["is_human"].sum())
print("\nEvidence count across all cells:")
print(m["evidence_count"].value_counts().sort_index().to_string())
print("\nSaved to outputs/cell_line_lookup.parquet")
