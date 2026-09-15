import pandas as pd

# Load the merged master table
t = pd.read_parquet("outputs/master_merged.parquet")

# The coverage columns (which data types each cell has)
cov_cols = [c for c in t.columns if c.startswith("has_")]

# Step 1: count how many data types each cell has
t["datatype_coverage"] = t[cov_cols].sum(axis=1)

# Step 2: build the confidence score (0 to 1)
t["cov_norm"] = t["datatype_coverage"] / len(cov_cols)
t["evi_norm"] = t["evidence_count"] / 3
t["confidence"] = (t["cov_norm"] + t["evi_norm"]) / 2

# Save the table with confidence added
t.to_parquet("outputs/master_with_confidence.parquet")

# Show the result
print("Confidence added to", len(t), "cell lines")
print("\nMost confident cells:")
print(t.sort_values("confidence", ascending=False)[["official_name","evidence_count","datatype_coverage","confidence"]].head(5).to_string(index=False))
print("\nLeast confident cells:")
print(t.sort_values("confidence")[["official_name","evidence_count","datatype_coverage","confidence"]].head(5).to_string(index=False))
