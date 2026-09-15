import pandas as pd

# Open the table I built
table = pd.read_parquet("outputs/cell_line_lookup.parquet")

# How big is it?
print("My table has", table.shape[0], "rows and", table.shape[1], "columns.")
print()

# What columns does it have?
print("Columns:", table.columns.tolist())
print()

# Show the first 10 rows
print("First 10 rows:")
print(table.head(10).to_string())
