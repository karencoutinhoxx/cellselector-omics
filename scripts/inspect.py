import sys as _sys

# This file is named 'inspect.py', which shadows the stdlib 'inspect' module when
# scripts/ sits on sys.path[0]. Strip it out before any other import so that
# libraries like pyarrow/cloudpickle/dataclasses can find the real stdlib inspect.
_sys.path = [p for p in _sys.path if not (p and p.rstrip("/").endswith("/scripts"))]

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import DOCS_DIR, PARQUET_DIR, setup_dirs

# ── Separator detection ────────────────────────────────────────────────────────

_SEPS = {"\t": "TAB", ",": "COMMA", "|": "PIPE", ";": "SEMICOLON"}
_SEP_NAMES = {"tab": "\t", "comma": ",", "pipe": "|", "semicolon": ";"}


def _sniff_sep(path: Path, override: str | None) -> tuple[str, str]:
    if override:
        sep = _SEP_NAMES.get(override.lower(), override)
        return sep, _SEPS.get(sep, repr(sep))
    suffix = path.suffix.lower()
    if suffix == ".tsv":
        return "\t", "TAB"
    if suffix == ".csv":
        return ",", "COMMA"
    with open(path, encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
    counts = {sep: first_line.count(sep) for sep in _SEPS}
    best = max(counts, key=counts.get)
    return (best, _SEPS[best]) if counts[best] > 0 else (",", "COMMA")


# ── ID pattern detection ───────────────────────────────────────────────────────

_UNIPROT_RE = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9](-\d+)?$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}(-\d+)?$")


def _hints_from_values(series: pd.Series) -> list[str]:
    if pd.api.types.is_float_dtype(series) or pd.api.types.is_integer_dtype(series):
        try:
            non_null = series.dropna()
            if len(non_null) > 0 and (non_null == non_null.astype(int)).all():
                return ["numeric — possible Entrez gene ID"]
        except (ValueError, OverflowError):
            pass
        return []
    sample = series.dropna().astype(str).head(50)
    if len(sample) == 0:
        return []
    hints = []
    if sample.str.startswith("ENSG").any():
        hints.append("Ensembl ENSG")
    if sample.str.startswith("ACH-").any():
        hints.append("DepMap ACH- cell line")
    if sample.str.startswith("CVCL_").any():
        hints.append("Cellosaurus CVCL_")
    if sample.str.startswith("GSM").any():
        hints.append("GEO GSM sample")
    if sample.str.startswith("PR-").any():
        hints.append("DepMap PR- profile")
    if sample.apply(lambda s: bool(_UNIPROT_RE.match(s))).mean() > 0.4:
        hints.append("UniProt accession")
    if not hints:
        hgnc_frac = sample.str.match(r"^[A-Z][A-Z0-9]{1,19}$").mean()
        if hgnc_frac > 0.65:
            hints.append("possible HGNC symbol")
    if sample.str.match(r"^\d+$").mean() > 0.8:
        hints.append("numeric — possible Entrez gene ID")
    return hints


def _hints_from_col_names(columns: pd.Index) -> list[str]:
    """Detect ID patterns in the column headers themselves."""
    sample = [str(c) for c in columns[:200]]
    hints = []
    if sum(1 for c in sample if c.startswith("ENSG")) / len(sample) > 0.3:
        hints.append("bare Ensembl ENSG")
    elif sum(1 for c in sample if "ENSG" in c) / len(sample) > 0.3:
        hints.append("ENSG embedded (e.g. SYMBOL (ENSG…))")
    if sum(1 for c in sample if c.startswith("GSM")) / len(sample) > 0.3:
        hints.append("GEO GSM")
    if sum(1 for c in sample if c.startswith("ACH-")) / len(sample) > 0.3:
        hints.append("DepMap ACH-")
    if sum(1 for c in sample if c.startswith("PR-")) / len(sample) > 0.3:
        hints.append("DepMap PR-")
    # UniProt-like prefix before space/paren
    up_frac = sum(1 for c in sample if bool(_UNIPROT_RE.match(c.split(" ")[0].split("(")[0]))) / len(sample)
    if up_frac > 0.3:
        hints.append("UniProt accession (possible 'UPID (GENE)' format)")
    return hints


# ── Main inspection logic ──────────────────────────────────────────────────────

def inspect(file_path: Path, sep_override: str | None, nrows: int, save: bool, log: bool) -> dict:
    if not file_path.exists():
        print(f"ERROR: file not found: {file_path}")
        return {"file": str(file_path), "status": "missing"}

    size_mb = file_path.stat().st_size / 1_048_576
    sep, sep_label = _sniff_sep(file_path, sep_override)

    print(f"\n{'='*72}")
    print(f"  {file_path.name}")
    print(f"{'='*72}")
    print(f"Path      : {file_path}")
    print(f"Size      : {size_mb:.1f} MB")
    print(f"Separator : {sep_label}")

    try:
        df = pd.read_csv(file_path, sep=sep, low_memory=False)
    except Exception as exc:
        print(f"ERROR loading: {exc}")
        return {"file": file_path.name, "status": "error", "error": str(exc)}

    rows, cols = df.shape
    print(f"Shape     : {rows:,} rows × {cols:,} columns")

    print(f"\n── Columns ({cols}) " + "─" * max(0, 52 - len(str(cols))))
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:>5}. {col}")

    print("\n── Dtypes " + "─" * 60)
    for col, dtype in df.dtypes.items():
        print(f"  {str(dtype):<12} {col}")

    print(f"\n── First {nrows} rows " + "─" * 55)
    print(df.head(nrows).to_string())

    first_col = df.columns[0]
    sample_vals = df[first_col].head(10).tolist()
    print(f"\n── First column: '{first_col}' " + "─" * max(0, 42 - len(first_col)))
    print(f"  Sample values (10): {sample_vals}")

    print("\n── Missing values " + "─" * 52)
    missing = df.isnull().sum()
    total_missing = int(missing.sum())
    any_missing = False
    for col, n in missing.items():
        if n > 0:
            print(f"  {col}: {n:,} ({100 * n / rows:.1f}%)")
            any_missing = True
    if not any_missing:
        print("  (none)")
    total_cells = rows * cols
    print(f"  Total: {total_missing:,} / {total_cells:,} ({100 * total_missing / total_cells:.1f}%)")

    print("\n── ID hints — column names " + "─" * 44)
    col_name_hints = _hints_from_col_names(df.columns)
    if col_name_hints:
        for h in col_name_hints:
            print(f"  {h}")
    else:
        print("  (no recognised patterns in column headers)")

    print("\n── ID hints — column values (object/int columns only) " + "─" * 17)
    all_value_hints: dict[str, list[str]] = {}
    check_cols = list(df.select_dtypes(include=["object"]).columns)
    # Also check purely-integer-valued float columns (Entrez check)
    check_cols += [c for c in df.select_dtypes(include=["number"]).columns
                   if df[c].dropna().shape[0] > 0]
    for col in check_cols:
        hints = _hints_from_values(df[col])
        if hints:
            all_value_hints[col] = hints
            print(f"  '{col}' → {', '.join(hints)}")
    if not all_value_hints:
        print("  (no recognised ID patterns in values)")

    # Parquet
    parquet_path = None
    if save:
        setup_dirs()
        stem = file_path.stem.replace(" ", "_")
        parquet_path = PARQUET_DIR / f"{stem}.parquet"
        if parquet_path.exists():
            print(f"\nParquet already exists, skipping: {parquet_path.name}")
        else:
            df.to_parquet(parquet_path, index=False)
            print(f"\nSaved Parquet → {parquet_path}")

    result = {
        "file": file_path.name,
        "status": "ok",
        "size_mb": round(size_mb, 1),
        "shape": (rows, cols),
        "sep": sep_label,
        "first_col": first_col,
        "first_col_sample": sample_vals[:5],
        "col_name_hints": col_name_hints,
        "value_hints": all_value_hints,
        "missing_total": total_missing,
        "parquet": str(parquet_path) if parquet_path else None,
    }

    if log:
        _append_log(result)

    return result


def _append_log(r: dict):
    log_path = DOCS_DIR / "progress_log.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows, cols = r["shape"]
    col_hints = ", ".join(r["col_name_hints"]) or "none"
    val_hints = "; ".join(
        f"{col}: {', '.join(hints)}" for col, hints in r["value_hints"].items()
    ) or "none"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n## Inspect — {r['file']} — {timestamp}\n")
        f.write(f"- Shape: {rows:,} × {cols:,} | Size: {r['size_mb']} MB | Sep: {r['sep']}\n")
        f.write(f"- First col: `{r['first_col']}` → sample: `{r['first_col_sample']}`\n")
        f.write(f"- Missing: {r['missing_total']:,} total\n")
        f.write(f"- Column name hints: {col_hints}\n")
        f.write(f"- Value hints: {val_hints}\n")
    print(f"\nProgress log updated → {log_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect a tabular data file: shape, columns, dtypes, ID hints."
    )
    parser.add_argument("--file",  required=True, help="Path to file to inspect")
    parser.add_argument("--sep",   default=None,  help="Separator override: tab/comma/pipe/semicolon or literal")
    parser.add_argument("--nrows", type=int, default=5, help="Rows to show in preview (default: 5)")
    parser.add_argument("--save",  action="store_true", help="Save DataFrame to Parquet in PARQUET_DIR")
    parser.add_argument("--log",   action="store_true", help="Append summary to docs/progress_log.md")
    args = parser.parse_args()

    inspect(
        file_path=Path(args.file),
        sep_override=args.sep,
        nrows=args.nrows,
        save=args.save,
        log=args.log,
    )

