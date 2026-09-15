import pandas as pd

# Open the Cellosaurus file
print("Opening Cellosaurus... this may take a few seconds.")
cello = pd.read_csv("data/7_cellosaurus.csv", low_memory=False)

# How big is it?
print("\nThe file has", cello.shape[0], "rows and", cello.shape[1], "columns.")

# What are the column names?
print("\nColumn names:")
print(cello.columns.tolist())

# Show the first 5 rows
print("\nFirst 5 rows:")
print(cello.head())
