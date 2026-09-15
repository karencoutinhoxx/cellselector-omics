import pandas as pd

# ============================================================
# MERGE SCRIPT
# Spine = nomenclature lookup table (shared cell line IDs).
# Each member's cleaned data is mapped onto it by cellosaurus_id.
# ============================================================

# ---- Spine: the nomenclature lookup ----
print("Loading spine (nomenclature lookup)...")
master = pd.read_parquet("outputs/cell_line_lookup.parquet")
master = master[["cellosaurus_id", "DepMap_ID", "official_name", "evidence_count"]].copy()

# ---- ID translation maps (from DepMap sample info) ----
samp = pd.read_csv("data/nomenclature/9_DepMap_sample_info.csv", low_memory=False)
dep_to_cvcl = dict(zip(samp["DepMap_ID"], samp["RRID"]))
ccle_to_cvcl = dict(zip(samp["CCLE_Name"], samp["RRID"]))
# ============================================================
# MEMBER 4 - non gene expression
# ============================================================
print("Merging Member 4 (non gene expression)...")

# File 1: genome signatures (keyed by DepMap ModelID) - attach features
sig = pd.read_csv("data/member4/OmicsGlobalSignatures_clean.tsv", sep="\t", low_memory=False)
sig["cellosaurus_id"] = sig["ModelID"].map(dep_to_cvcl)
sig = sig.dropna(subset=["cellosaurus_id"]).drop(columns=["ModelID"]).drop_duplicates("cellosaurus_id")
master = master.merge(sig, on="cellosaurus_id", how="left")

# File 2: metabolomics (cell lines are columns = DepMap IDs) - add coverage flag
met = pd.read_csv("data/member4/CCLE_metabolomics_wide.tsv", sep="\t", nrows=1, low_memory=False)
met_cvcls = {dep_to_cvcl.get(c) for c in met.columns if c.startswith("ACH-")}
master["has_metabolomics"] = master["cellosaurus_id"].isin(met_cvcls)

# File 3: miRNA (cell lines are columns = CCLE names) - add coverage flag
mir = pd.read_csv("data/member4/CCLE_miRNA_wide.tsv", sep="\t", nrows=1, low_memory=False)
mir_cvcls = {ccle_to_cvcl.get(c) for c in mir.columns if c != "miRNA"}
master["has_mirna"] = master["cellosaurus_id"].isin(mir_cvcls)

# ============================================================
# MEMBER 2 - gene expression (all four files)
# ============================================================
print("Merging Member 2 (gene expression)...")
pq = "outputs/parquet/"

# proteomics: original_id = ACH ids -> map via dep_to_cvcl
prot = pd.read_parquet(pq+"gene_expr_ccle_proteomics_preprocessed.parquet", columns=["original_id"])
prot_cells = set(prot["original_id"].dropna().map(dep_to_cvcl).dropna())
master["has_proteomics"] = master["cellosaurus_id"].isin(prot_cells)

# HPA: original_id = free-text names -> map via hpa_name in the lookup
hpa_out = pd.read_parquet(pq+"gene_expr_hpa_preprocessed.parquet", columns=["original_id"])
hpa_lookup = pd.read_parquet("outputs/cell_line_lookup.parquet")[["cellosaurus_id","hpa_name"]].dropna()
hpa_name_to_cvcl = dict(zip(hpa_lookup["hpa_name"], hpa_lookup["cellosaurus_id"]))
hpa_cells = set(hpa_out["original_id"].dropna().map(hpa_name_to_cvcl).dropna())
master["has_hpa_expr"] = master["cellosaurus_id"].isin(hpa_cells)

# DepMap: already has cellosaurus_id column -> use directly
dep_out = pd.read_parquet(pq+"gene_expr_depmap_preprocessed.parquet", columns=["cellosaurus_id"])
master["has_depmap_expr"] = master["cellosaurus_id"].isin(set(dep_out["cellosaurus_id"].dropna()))

# GEO: original_id = GSM codes -> map via GEO info (Geo_accession -> Cellosaurus_ID)
geo_info = pd.read_csv("data/nomenclature/10_GEOInfo.txt", sep="\t", low_memory=False)
gsm_to_cvcl = dict(zip(geo_info["Geo_accession"], geo_info["Cellosaurus_ID"]))
geo_out = pd.read_parquet(pq+"gene_expr_geo_preprocessed.parquet", columns=["original_id"])
geo_cells = set(geo_out["original_id"].dropna().map(gsm_to_cvcl).dropna())
master["has_geo_expr"] = master["cellosaurus_id"].isin(geo_cells)
# ============================================================
# MEMBER 3 - gene properties (mutations + fusions)
# ============================================================
print("Merging Member 3 (gene properties)...")

# mutations: ProfileID = PR- ids -> chain PR- → ACH- → CVCL_
prof_df = pd.read_csv("data/nomenclature/8_DepMap_OmicsProfiles.csv",
                      usecols=["ProfileID", "ModelID"], low_memory=False)
pr_to_ach = dict(zip(prof_df["ProfileID"], prof_df["ModelID"]))
mut = pd.read_csv("data/member3/mutation_gene_features.csv", low_memory=False, usecols=["ProfileID"])
mut_cells = set(
    mut["ProfileID"].dropna()
                    .map(pr_to_ach)      # PR- → ACH-
                    .dropna()
                    .map(dep_to_cvcl)    # ACH- → CVCL_
                    .dropna()
)
master["has_mutations"] = master["cellosaurus_id"].isin(mut_cells)

# fusions: ModelID = ACH ids -> map via dep_to_cvcl
fus = pd.read_csv("data/member3/fusion_gene_features.csv", low_memory=False, usecols=["ModelID"])
fus_cells = set(fus["ModelID"].dropna().map(dep_to_cvcl).dropna())
master["has_fusions"] = master["cellosaurus_id"].isin(fus_cells)
# ---- Save the merged master table ----
master.to_parquet("outputs/master_merged.parquet")

print("\nMerged master table:", master.shape[0], "cells,", master.shape[1], "columns")
print("Cells with signature features:", master["MSIScore"].notna().sum())
print("Cells with metabolomics:", master["has_metabolomics"].sum())
print("Cells with miRNA:", master["has_mirna"].sum())
print("\nSaved to outputs/master_merged.parquet")
