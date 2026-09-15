import pandas as pd

# Load the merged table
table = pd.read_parquet("outputs/master_merged.parquet")

print("Shape:", table.shape[0], "rows,", table.shape[1], "columns")

# Export a CSV copy you can open in Excel
table.to_csv("outputs/master_merged_view.csv", index=False)
print("Saved a CSV copy: outputs/master_merged_view.csv")
