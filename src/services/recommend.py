import pandas as pd

GENE = "TP53"
print(f"Building recommendations for gene: {GENE}")

# ---- 1. Gene expression scores (uses cell names) ----
scores = pd.read_parquet("outputs/gene_scores.parquet")

# ---- 2. Map cell names to cellosaurus IDs using the lookup (nomenclature bridge) ----
lookup = pd.read_parquet("outputs/cell_line_lookup.parquet")[["cellosaurus_id","hpa_name"]].dropna()
name_to_id = dict(zip(lookup["hpa_name"], lookup["cellosaurus_id"]))
scores["cellosaurus_id"] = scores["original_id"].map(name_to_id)
scores = scores.dropna(subset=["cellosaurus_id"])

# ---- 3. Bring in confidence scores ----
conf = pd.read_parquet("outputs/master_with_confidence.parquet")[["cellosaurus_id","official_name","confidence"]]
merged = scores.merge(conf, on="cellosaurus_id", how="left")

# ---- 4. Final recommendation score: expression strength x confidence ----
merged["final_score"] = merged["expr_score"] * merged["confidence"]

# ---- 5. Rank and show ----
merged = merged.sort_values("final_score", ascending=False)
merged.to_parquet("outputs/recommendations.parquet")

print(f"\nTop 10 recommended cell lines for {GENE}:")
print(merged[["official_name","expr_score","confidence","final_score"]].head(10).to_string(index=False))
