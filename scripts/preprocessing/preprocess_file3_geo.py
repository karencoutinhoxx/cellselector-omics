import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import FILES, PARQUET_DIR, setup_dirs

OUTPUT_NAME = "gene_expr_geo_preprocessed.parquet"

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

# How many transposed rows (GSM samples) to melt at once
MELT_CHUNK = 200


def preprocess_file3() -> pd.DataFrame:
    t0 = time.perf_counter()
    setup_dirs()

    src_path = FILES["geo"]
    out_path = PARQUET_DIR / OUTPUT_NAME
    today = date.today().strftime("%Y-%m-%d")

    size_gb = src_path.stat().st_size / 1_073_741_824
    print(f"\nSource  : {src_path.name}  ({size_gb:.2f} GB)")

    # ── Step 1: read column names to build an explicit dtype map ─────────────
    # Using a dtype dict keeps Gene as string index while forcing all 3,267
    # GSM expression columns to float32 (~260 MB vs ~520 MB for float64).
    print("Reading column headers …")
    header_cols = pd.read_csv(src_path, sep="\t", nrows=0).columns.tolist()
    gene_col   = header_cols[0]          # "Gene"
    gsm_cols   = header_cols[1:]         # 3,267 GSM accessions
    dtype_map  = {c: "float32" for c in gsm_cols}
    # gene_col will become the index (string) — no entry needed in dtype_map

    # ── Step 2: full load ─────────────────────────────────────────────────────
    print(f"Loading full file as float32 (expect ~260 MB in RAM) …")
    df = pd.read_csv(src_path, sep="\t", index_col=0, dtype=dtype_map)

    raw_rows, raw_cols = df.shape          # 19,914 genes × 3,267 samples
    print(f"Raw shape (genes × samples)        : {raw_rows:,} × {raw_cols:,}")

    unique_genes = df.index.tolist()       # ENSG IDs (from row index)

    # ── Step 3: transpose ─────────────────────────────────────────────────────
    # df.T on a homogeneous float32 frame returns a copy (~260 MB more → ~520 MB peak).
    # After this: rows = GSM samples, columns = ENSG gene IDs.
    print("Transposing (rows → GSM samples, cols → ENSG genes) …")
    df = df.T                              # 3,267 × 19,914
    tr_rows, tr_cols = df.shape
    print(f"Shape after transpose (samples × genes): {tr_rows:,} × {tr_cols:,}")

    unique_gsm = df.index.tolist()         # GSM accessions

    # ── Step 3b: build ENSG → gene_symbol lookup from HPA preprocessed parquet ─
    print("Building ENSG → gene_symbol lookup from HPA parquet …")
    hpa_path = PARQUET_DIR / "gene_expr_hpa_preprocessed.parquet"
    hpa_sym  = pd.read_parquet(hpa_path, columns=["gene_id", "gene_symbol"])
    hpa_sym  = hpa_sym.drop_duplicates("gene_id").dropna(subset=["gene_symbol"])
    ensg_to_symbol: dict[str, str] = dict(zip(hpa_sym["gene_id"], hpa_sym["gene_symbol"]))
    print(f"  {len(ensg_to_symbol):,} ENSG → symbol pairs loaded from HPA")

    # ── Step 4: chunk-melt → stream to Parquet ───────────────────────────────
    # Process MELT_CHUNK GSM rows at a time to keep melt output manageable
    # (200 × 19,914 ≈ 4M rows per chunk).
    print(f"\nMelting in chunks of {MELT_CHUNK} GSM rows, streaming to Parquet …")

    writer         = None
    rows_out       = 0
    n_chunks       = 0
    symbols_filled = 0
    symbols_nan    = 0

    for start in range(0, tr_rows, MELT_CHUNK):
        end   = min(start + MELT_CHUNK, tr_rows)
        chunk = df.iloc[start:end].copy()

        chunk.index.name = "original_id"
        chunk = chunk.reset_index()

        long = chunk.melt(
            id_vars=["original_id"],
            var_name="gene_id",
            value_name="expression_value",
        )

        # Categorical columns → dictionary encoding in Parquet (tiny footprint)
        n = len(long)
        long["original_id_type"] = pd.Categorical(["GSM"] * n,               categories=["GSM"])
        long["gene_symbol"]      = long["gene_id"].map(ensg_to_symbol)        # backfilled from HPA
        long["source"]           = pd.Categorical(["GEO_expression"] * n,    categories=["GEO_expression"])
        long["units"]            = pd.Categorical(["unknown_expression"] * n, categories=["unknown_expression"])
        long["date_processed"]   = pd.Categorical([today] * n,                categories=[today])

        symbols_filled += int(long["gene_symbol"].notna().sum())
        symbols_nan    += int(long["gene_symbol"].isna().sum())

        long = long[STANDARD_COLS]
        rows_out += len(long)

        table = pa.Table.from_pandas(long, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
        writer.write_table(table)

        n_chunks += 1
        if n_chunks % 20 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  chunk {n_chunks:>3}  GSM {start + 1}–{end}  |  {rows_out:>12,} rows written  |  {elapsed:>6.1f}s")

    if writer:
        writer.close()

    elapsed    = time.perf_counter() - t0
    parquet_mb = out_path.stat().st_size / 1_048_576
    expected   = raw_rows * raw_cols

    sample = pd.read_parquet(out_path).head(5)

    print(f"\n{'─'*62}")
    print(f"Raw shape before transpose : {raw_rows:,} × {raw_cols:,}")
    print(f"Shape after transpose      : {tr_rows:,} × {tr_cols:,}")
    print(f"Unique GSM samples         : {len(unique_gsm):,}")
    print(f"Unique genes               : {len(unique_genes):,}")
    print(f"Expected row count         : {expected:,}  (19,914 × 3,267)")
    print(f"Actual row count           : {rows_out:,}")
    print(f"Gene symbols filled        : {symbols_filled:,} / {rows_out:,} ({100*symbols_filled/rows_out:.1f}%)")
    print(f"Gene symbols remaining NaN : {symbols_nan:,} ({100*symbols_nan/rows_out:.1f}%)")
    print(f"Final shape                : {rows_out:,} rows × 8 columns")
    print(f"Parquet size on disk       : {parquet_mb:.1f} MB")
    print(f"Time taken                 : {elapsed:.1f}s")
    print(f"\nColumn order: {STANDARD_COLS}")
    print(f"\nSample output (5 rows):")
    print(sample.to_string(index=False))
    print(f"\nSaved → {out_path}")

    return sample


if __name__ == "__main__":
    df = preprocess_file3()

