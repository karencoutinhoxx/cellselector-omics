import pandas as pd
import sys

# ---- Which gene are we scoring for? ----
# (later this comes from user input; for now, set it here to test)
GENE = "TP53"

print(f"Scoring cell lines for gene: {GENE}")

# ---- Load expression data (Ishaan's HPA output) ----
expr = pd.read_parquet("outputs/parquet/gene_expr_hpa_preprocessed.parquet",
                       columns=["original_id","gene_symbol","expression_value"])

# ---- Keep only rows for our gene ----
gene_rows = expr[expr["gene_symbol"] == GENE].copy()
print(f"Found {len(gene_rows)} cell lines with expression data for {GENE}")

# ---- Average expression per cell (in case of duplicates) ----
per_cell = gene_rows.groupby("original_id")["expression_value"].mean().reset_index()

# ---- Score 0 to 1: strongest expression = 1.0 ----
maxval = per_cell["expression_value"].max()
per_cell["expr_score"] = per_cell["expression_value"] / maxval

# ---- Sort best to worst ----
per_cell = per_cell.sort_values("expr_score", ascending=False)

# ---- Save and show ----
per_cell.to_parquet("outputs/gene_scores.parquet")
print(f"\nTop 10 cell lines expressing {GENE}:")
print(per_cell.head(10).to_string(index=False))

