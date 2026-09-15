import sys
import re
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import FILES, PARQUET_DIR, setup_dirs

# Output file name for this preprocessor
OUTPUT_NAME = "gene_expr_ccle_proteomics_preprocessed.parquet"

# "UPID (GENE)" or "UPID-isoform (GENE)" pattern
_COL_RE = re.compile(r"^([A-Z0-9]+-?\d*)\s+\(([^)]+)\)$")

STANDARD_COLS = [
    "original_id",
    "original_id_type",
    "gene_id",
    "gene_symbol",
    "expression_value",
    "source",
    "units",
    "date_processed",
]


def parse_protein_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a mapping table from raw column name → (gene_id, gene_symbol).
    Columns that do not match the 'UPID (GENE)' pattern are tracked separately.
    Returns a tidy DataFrame of (raw_col, gene_id, gene_symbol).
    """
    records = []
    failed = []
    for col in df.columns:
        m = _COL_RE.match(col)
        if m:
            records.append({"raw_col": col, "gene_id": m.group(1), "gene_symbol": m.group(2)})
        else:
            failed.append(col)
            records.append({"raw_col": col, "gene_id": col, "gene_symbol": None})

    mapping = pd.DataFrame(records)
    n_ok = len(records) - len(failed)
    print(f"Columns parsed successfully : {n_ok:,} / {len(df.columns):,}")
    if failed:
        print(f"Columns failed to parse    : {len(failed)}")
        for c in failed:
            print(f"  - {c!r}")
    else:
        print("Columns failed to parse    : 0")
    return mapping, failed


def preprocess_file4() -> pd.DataFrame:
    setup_dirs()

    src_path = FILES["ms_ccle"]
    print(f"\nLoading: {src_path.name}")
    df = pd.read_csv(src_path, index_col=0)
    print(f"Raw shape                  : {df.shape[0]:,} rows × {df.shape[1]:,} columns")

    # Parse column names
    print()
    col_map, failed_cols = parse_protein_columns(df)

    # Melt wide → long
    df.index.name = "original_id"
    df = df.reset_index()
    long = df.melt(id_vars="original_id", var_name="raw_col", value_name="expression_value")
    print(f"\nRows before NaN drop       : {len(long):,}")

    # Drop NaN expression values (expected ~27.8% for mass spec)
    long = long.dropna(subset=["expression_value"])
    print(f"Rows after NaN drop        : {len(long):,}")
    total = df.shape[0] * (df.shape[1] - 1)  # exclude original_id col
    pct = 100 * len(long) / total
    print(f"% values retained          : {pct:.1f}%")

    # Merge in gene_id / gene_symbol from mapping
    long = long.merge(col_map[["raw_col", "gene_id", "gene_symbol"]], on="raw_col", how="left")
    long = long.drop(columns="raw_col")

    # Add standard metadata columns
    long["original_id_type"] = "ACH"
    long["source"] = "CCLE_proteomics"
    long["units"] = "protein_intensity_MS"
    long["date_processed"] = date.today().strftime("%Y-%m-%d")

    # Enforce column order
    long = long[STANDARD_COLS]

    print(f"Final shape                : {long.shape[0]:,} rows × {long.shape[1]} columns")
    print(f"\nColumn order: {long.columns.tolist()}")
    print(f"\nSample output (5 rows):")
    print(long.head(5).to_string(index=False))

    # Save
    out_path = PARQUET_DIR / OUTPUT_NAME
    long.to_parquet(out_path, index=False)
    print(f"\nSaved → {out_path}")

    return long


if __name__ == "__main__":
    df = preprocess_file4()

