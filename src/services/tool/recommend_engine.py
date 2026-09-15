import pandas as pd

# ---- Load everything once (so it's ready for any gene) ----
print("Loading data...")
EXPR = pd.read_parquet("outputs/parquet/gene_expr_hpa_preprocessed.parquet",
                       columns=["original_id","gene_symbol","expression_value"])
LOOKUP = pd.read_parquet("outputs/cell_line_lookup.parquet")[["cellosaurus_id","hpa_name"]].dropna()
NAME_TO_ID = dict(zip(LOOKUP["hpa_name"], LOOKUP["cellosaurus_id"]))
CONF = pd.read_parquet("outputs/master_with_confidence.parquet")[["cellosaurus_id","official_name","confidence"]]


def recommend(gene, top_n=10):
    """Given a gene, return the top ranked cell lines with confidence."""
    # 1. filter expression to this gene
    rows = EXPR[EXPR["gene_symbol"] == gene]
    if len(rows) == 0:
        print(f"No expression data found for gene: {gene}")
        return None

    # 2. average expression per cell, score 0-1
    per_cell = rows.groupby("original_id")["expression_value"].mean().reset_index()
    per_cell["expr_score"] = per_cell["expression_value"] / per_cell["expression_value"].max()

    # 3. map cell names to cellosaurus IDs (nomenclature bridge)
    per_cell["cellosaurus_id"] = per_cell["original_id"].map(NAME_TO_ID)
    per_cell = per_cell.dropna(subset=["cellosaurus_id"])

    # 4. attach confidence
    result = per_cell.merge(CONF, on="cellosaurus_id", how="left")

    # 5. final score = expression x confidence
    result["final_score"] = result["expr_score"] * result["confidence"]

    # 6. rank and return top N
    result = result.sort_values("final_score", ascending=False)
    return result[["official_name","expr_score","confidence","final_score"]].head(top_n)


# ---- Test it with a few genes ----
if __name__ == "__main__":
    for g in ["TP53", "BRCA1", "EGFR"]:
        print(f"\n{'='*50}\nTop cell lines for {g}:")
        r = recommend(g)
        if r is not None:
            print(r.to_string(index=False))
