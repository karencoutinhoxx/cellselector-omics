import re
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import PROJECT_ROOT, PARQUET_DIR, SAMPLE_INFO, setup_dirs

CRISPR_DIR = PROJECT_ROOT / "data" / "crispr"
DEPENDENCY_CSV = CRISPR_DIR / "CRISPRGeneDependency.csv"
SCREEN_MAP_CSV = CRISPR_DIR / "CRISPRScreenMap.csv"

OUTPUT_NAME = "crispr_dependency_depmap_preprocessed.parquet"
COL_CHUNK   = 2000   # gene columns loaded per CSV pass

# CONCEPTUAL NOTE: CRISPR dependency scores measure gene ESSENTIALITY
# (probability a cell line requires the gene to survive), not EXPRESSION.
# `evidence_type` marks this explicitly so downstream consumers never
# conflate a "highly dependent" gene with a "highly expressed" one, and
# the value column is named `dependency_score` rather than the
# `expression_value` used by the expression preprocessors.
EVIDENCE_TYPE = "gene_dependency"
SOURCE        = "DepMap_CRISPR"
UNITS         = "dependency_probability"   # Chronos-derived probability, range 0-1

STANDARD_COLS = [
    "original_id",
    "original_id_type",
    "gene_id",
    "gene_symbol",
    "dependency_score",
    "evidence_type",
    "source",
    "units",
    "date_processed",
    "cellosaurus_id",
]

# "SYMBOL (entrez_id)", e.g. "A1BG (1)"  → symbol + Entrez ID
_SYMBOL_ENTREZ_RE = re.compile(r"^(.+?)\s+\((\d+)\)$")


def parse_gene_columns(col_names: list[str]) -> tuple[dict, dict, list]:
    """
    Returns:
      symbol_map : raw_col → gene_symbol
      entrez_map : raw_col → gene_id (Entrez ID as string)
      failed     : columns that didn't match "SYMBOL (entrez_id)"
    """
    symbol_map: dict[str, str] = {}
    entrez_map: dict[str, str] = {}
    failed: list[str] = []

    for col in col_names:
        m = _SYMBOL_ENTREZ_RE.match(col)
        if m:
            symbol_map[col] = m.group(1)
            entrez_map[col] = m.group(2)
        else:
            symbol_map[col] = None
            entrez_map[col] = col
            failed.append(col)

    return symbol_map, entrez_map, failed


def preprocess_crispr_dependency() -> pd.DataFrame:
    t0 = time.perf_counter()
    setup_dirs()

    out_path = PARQUET_DIR / OUTPUT_NAME
    today    = date.today().strftime("%Y-%m-%d")

    size_gb = DEPENDENCY_CSV.stat().st_size / 1_073_741_824
    print(f"\nSource  : {DEPENDENCY_CSV.name}  ({size_gb:.2f} GB)")

    # ── Step 1: ScreenID → ModelID (ACH-) → cellosaurus_id (CVCL_) ────────────
    # CRISPRGeneDependency.csv is already indexed by ModelID (ACH-) in this
    # release, but we still build the ScreenID → ModelID map from
    # CRISPRScreenMap.csv and fall back to it for any row ID that isn't
    # already an ACH- ID, so the script keeps working if a future export
    # goes back to indexing by ScreenID.
    print("Loading CRISPR screen map...")
    screen_map = pd.read_csv(SCREEN_MAP_CSV)
    screen_to_ach = dict(zip(screen_map["ScreenID"], screen_map["ModelID"]))

    print("Building ModelID (ACH-) → cellosaurus_id (CVCL_) map...")
    samp_df = pd.read_csv(SAMPLE_INFO, usecols=["DepMap_ID", "RRID"], low_memory=False)
    ach_to_cvcl = {row.DepMap_ID: row.RRID for row in samp_df.itertuples() if pd.notna(row.RRID)}

    def resolve_ach(raw_id: str) -> str:
        return raw_id if str(raw_id).startswith("ACH-") else screen_to_ach.get(raw_id, raw_id)

    # ── Step 2: read column names only ────────────────────────────────────────
    print("Reading column headers …")
    all_cols  = pd.read_csv(DEPENDENCY_CSV, nrows=0).columns.tolist()
    id_col    = all_cols[0]          # "Unnamed: 0"  → ModelID (ACH-) rows
    gene_cols = all_cols[1:]
    n_genes   = len(gene_cols)
    n_chunks  = -(-n_genes // COL_CHUNK)   # ceil div

    print(f"ID column   : '{id_col}'")
    print(f"Gene columns: {n_genes:,}")
    print(f"Col chunks  : {n_chunks}  ({COL_CHUNK} gene cols each)")

    # ── Step 3: pre-parse all column names in one pass (pure string ops) ──────
    symbol_map, entrez_map, failed_cols = parse_gene_columns(gene_cols)

    print(f"\nColumn name parsing:")
    print(f"  SYMBOL (entrez_id) matched : {n_genes - len(failed_cols):,} / {n_genes:,}")
    if failed_cols:
        print(f"  Failed to parse            : {len(failed_cols)}")
        for c in failed_cols[:20]:
            print(f"    - {c!r}")

    # ── Step 4: column-chunked CSV load → melt → stream to Parquet ───────────
    print(f"\nColumn-chunked processing ({n_chunks} passes over the CSV) …")

    writer         = None
    rows_out       = 0
    unique_row_ids = None

    for chunk_idx, start in enumerate(range(0, n_genes, COL_CHUNK), 1):
        t_chunk    = time.perf_counter()
        chunk_cols = gene_cols[start:start + COL_CHUNK]
        dtype_map  = {c: "float32" for c in chunk_cols}

        df = pd.read_csv(
            DEPENDENCY_CSV,
            usecols=[id_col] + chunk_cols,
            dtype=dtype_map,
        )

        if unique_row_ids is None:
            unique_row_ids = df[id_col].tolist()

        df = df.rename(columns={id_col: "original_id"})

        long = df.melt(
            id_vars=["original_id"],
            var_name="_raw_col",
            value_name="dependency_score",
        )
        long = long.dropna(subset=["dependency_score"])

        long["gene_symbol"] = long["_raw_col"].map(symbol_map)
        long["gene_id"]     = long["_raw_col"].map(entrez_map)
        long = long.drop(columns="_raw_col")

        n = len(long)
        long["original_id_type"] = pd.Categorical(["ACH"] * n,          categories=["ACH"])
        long["evidence_type"]    = pd.Categorical([EVIDENCE_TYPE] * n,  categories=[EVIDENCE_TYPE])
        long["source"]           = pd.Categorical([SOURCE] * n,         categories=[SOURCE])
        long["units"]            = pd.Categorical([UNITS] * n,          categories=[UNITS])
        long["date_processed"]   = pd.Categorical([today] * n,          categories=[today])
        long["cellosaurus_id"]   = long["original_id"].map(resolve_ach).map(ach_to_cvcl)

        long = long[STANDARD_COLS]
        rows_out += n

        table = pa.Table.from_pandas(long, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
        writer.write_table(table)

        chunk_t = time.perf_counter() - t_chunk
        total_t = time.perf_counter() - t0
        print(
            f"  chunk {chunk_idx:>3}/{n_chunks}"
            f"  cols {start + 1:>6}–{start + len(chunk_cols):<6}"
            f"  |  {n:>9,} rows"
            f"  |  {chunk_t:>5.1f}s/chunk"
            f"  |  {total_t:>6.1f}s total"
        )

    if writer:
        writer.close()

    elapsed    = time.perf_counter() - t0
    parquet_mb = out_path.stat().st_size / 1_048_576
    expected   = len(unique_row_ids) * n_genes

    resolved_ach   = [resolve_ach(r) for r in unique_row_ids]
    ach_hits       = sum(1 for r in unique_row_ids if str(r).startswith("ACH-") or r in screen_to_ach)
    cvcl_hits      = sum(1 for a in resolved_ach if ach_to_cvcl.get(a) is not None)
    ach_no_cvcl    = ach_hits - cvcl_hits

    sample = pd.read_parquet(out_path).head(5)

    print(f"\n{'─'*66}")
    print(f"Total gene columns       : {n_genes:,}")
    print(f"Columns parsed           : {n_genes - len(failed_cols):,} / {n_genes:,}")
    print(f"Column chunks processed  : {n_chunks}")
    print(f"Unique screens/models    : {len(unique_row_ids):,}")
    print(f"Resolved → ACH-          : {ach_hits:,} / {len(unique_row_ids):,}")
    print(f"ACH- → CVCL_ mapped      : {cvcl_hits:,} / {ach_hits:,}"
          + (f"  ({ach_no_cvcl} failed)" if ach_no_cvcl else "  (all mapped)"))
    print(f"Expected row count (max) : {expected:,}  ({len(unique_row_ids):,} × {n_genes:,})")
    print(f"Actual row count         : {rows_out:,}  (NaN dependency scores dropped)")
    print(f"Final shape              : {rows_out:,} rows × {len(STANDARD_COLS)} columns")
    print(f"Parquet size on disk     : {parquet_mb:.1f} MB")
    print(f"Time taken               : {elapsed:.1f}s")
    print(f"\nColumn order: {STANDARD_COLS}")
    print(f"\nSample output (5 rows):")
    print(sample.to_string(index=False))
    print(f"\nSaved → {out_path}")

    return sample


if __name__ == "__main__":
    df = preprocess_crispr_dependency()

