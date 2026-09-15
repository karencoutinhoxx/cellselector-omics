import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import FILES, PARQUET_DIR, setup_dirs

OUTPUT_NAME = "gene_expr_hpa_preprocessed.parquet"

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

VALUE_COLS = ["TPM", "pTPM", "nTPM"]

# Read dtypes: float32 halves memory vs float64 (TPM values have 1 d.p. precision)
READ_DTYPES = {
    "Gene":       "string",
    "Gene name":  "string",
    "Cell line":  "string",
    "TPM":        "float32",
    "pTPM":       "float32",
    "nTPM":       "float32",
}

RENAME = {
    "Gene":      "gene_id",
    "Gene name": "gene_symbol",
    "Cell line": "original_id",
}


def _process_chunk(chunk: pd.DataFrame, today: str) -> pd.DataFrame:
    chunk = chunk.rename(columns=RENAME)

    long = chunk.melt(
        id_vars=["gene_id", "gene_symbol", "original_id"],
        value_vars=VALUE_COLS,
        var_name="units",
        value_name="expression_value",
    )

    # Low-cardinality string columns → Categorical so parquet uses dictionary encoding
    long["original_id_type"] = pd.Categorical(["freetext"] * len(long), categories=["freetext"])
    long["source"]           = pd.Categorical(["HPA_RNA"] * len(long),  categories=["HPA_RNA"])
    long["date_processed"]   = pd.Categorical([today] * len(long),       categories=[today])
    long["units"]            = pd.Categorical(long["units"],              categories=VALUE_COLS)

    return long[STANDARD_COLS]


def preprocess_file1() -> pd.DataFrame:
    t0 = time.perf_counter()
    setup_dirs()

    src_path = FILES["hpa"]
    out_path = PARQUET_DIR / OUTPUT_NAME
    today = date.today().strftime("%Y-%m-%d")

    size_gb = src_path.stat().st_size / 1_073_741_824
    print(f"\nSource     : {src_path.name}  ({size_gb:.2f} GB)")
    print(f"Strategy   : chunked load (500k rows/chunk) → per-chunk melt → stream to Parquet")

    reader = pd.read_csv(
        src_path,
        sep="\t",
        dtype=READ_DTYPES,
        chunksize=500_000,
    )

    writer = None
    schema = None
    rows_in = 0
    rows_out = 0
    n_chunks = 0
    unique_cell_lines: set[str] = set()
    unique_genes: set[str] = set()

    print()
    for chunk in reader:
        n_chunks += 1
        rows_in += len(chunk)
        unique_cell_lines.update(chunk["Cell line"].dropna().tolist())
        unique_genes.update(chunk["Gene"].dropna().tolist())

        long = _process_chunk(chunk, today)
        rows_out += len(long)

        table = pa.Table.from_pandas(long, preserve_index=False)

        if writer is None:
            schema = table.schema
            writer = pq.ParquetWriter(out_path, schema, compression="snappy")

        writer.write_table(table)

        if n_chunks % 10 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  chunk {n_chunks:>3}  |  {rows_in:>12,} rows in  |  {rows_out:>12,} rows out  |  {elapsed:>6.1f}s elapsed")

    if writer:
        writer.close()

    elapsed = time.perf_counter() - t0
    parquet_mb = out_path.stat().st_size / 1_048_576

    # Load a tiny slice for display without pulling all 72M rows into RAM
    sample = pd.read_parquet(out_path).head(5)

    print(f"\n{'─'*60}")
    print(f"Raw shape              : 24,315,372 × 6")
    print(f"Chunks processed       : {n_chunks}")
    print(f"Unique cell lines      : {len(unique_cell_lines):,}")
    print(f"Unique genes           : {len(unique_genes):,}")
    print(f"Rows after melt        : {rows_out:,}  (expected ~72,946,116 = 24,315,372 × 3)")
    print(f"Final shape            : {rows_out:,} rows × 8 columns")
    print(f"Parquet size on disk   : {parquet_mb:.1f} MB")
    print(f"Time taken             : {elapsed:.1f}s")
    print(f"\nColumn order: {STANDARD_COLS}")
    print(f"\nSample output (5 rows):")
    print(sample.to_string(index=False))
    print(f"\nSaved → {out_path}")

    return sample


if __name__ == "__main__":
    df = preprocess_file1()

