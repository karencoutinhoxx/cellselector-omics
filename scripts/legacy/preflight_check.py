"""
Pre-flight check before implementing mutation scoring.

Verifies:
1. All bugs fixed earlier today are STILL fixed (load_mappings, 
   GENE_CLASSES overlap, FIXED_WEIGHTS sum, bare except)
2. Neo4j is alive and populated (didn't get auto-deleted again)
3. RRF and LambdaMART modules from the last task work correctly
   and didn't introduce new bugs
4. Git status is clean / everything committed
5. mutation_gene_features.csv has the exact structure the new 
   mutation scoring prompt assumes (columns, resolution rate)
6. Re-runs the general bug-pattern sweep to catch anything new
"""

import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

RESULTS = []

def check(name, condition, detail="", severity="FAIL"):
    status = "PASS" if condition else severity
    RESULTS.append((name, status, detail))
    icon = "PASS" if status == "PASS" else ("WARN" if status == "WARN" else "FAIL")
    print(f"[{icon}] {name}")
    if detail:
        print(f"      {detail}")

def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# =============================================================================
section("1. VERIFY load_mappings() FIX IS STILL INTACT")
# =============================================================================

try:
    from src.models.classical.scorer import load_mappings
    import pandas as pd

    hpa_to_cvcl, ach_to_cvcl, gsm_to_cvcl = load_mappings()

    check("hpa_to_cvcl non-empty", len(hpa_to_cvcl) > 0,
          f"{len(hpa_to_cvcl)} entries")
    check("gsm_to_cvcl non-empty", len(gsm_to_cvcl) > 0,
          f"{len(gsm_to_cvcl)} entries")

    lkp = pd.read_parquet("outputs/cell_line_lookup.parquet",
                           columns=["cellosaurus_id", "hpa_name"])
    lkp_valid = lkp.dropna(subset=["hpa_name"])
    ground_truth = dict(zip(lkp_valid["hpa_name"], lkp_valid["cellosaurus_id"]))
    mismatches = sum(1 for k, v in ground_truth.items()
                      if hpa_to_cvcl.get(k) != v)
    check("HPA mapping matches ground truth (no regression)",
          mismatches == 0,
          f"{mismatches}/{len(ground_truth)} mismatches "
          f"(should be 0 -- this was 1126/1198 before the fix)")

    geo = pd.read_csv("data/nomenclature/10_GEOInfo.txt", sep="\t",
                       usecols=["Geo_accession", "Cellosaurus_ID"],
                       low_memory=False)
    geo_valid = geo.dropna(subset=["Geo_accession", "Cellosaurus_ID"])
    geo_truth = dict(zip(geo_valid["Geo_accession"], geo_valid["Cellosaurus_ID"]))
    geo_mismatches = sum(1 for k, v in geo_truth.items()
                          if gsm_to_cvcl.get(k) != v)
    check("GEO mapping matches ground truth (no regression)",
          geo_mismatches == 0,
          f"{geo_mismatches}/{len(geo_truth)} mismatches "
          f"(should be 0 -- this was 2534/3159 before the fix)")

except Exception as e:
    check("load_mappings() regression check", False, str(e))


# =============================================================================
section("2. VERIFY GENE_CLASSES OVERLAP FIX IS STILL INTACT")
# =============================================================================

try:
    from src.models.classical.scorer import GENE_CLASSES
    ub = set(GENE_CLASSES.get("ubiquitous", []))
    lof = set(GENE_CLASSES.get("loss_of_function", []))
    ts = set(GENE_CLASSES.get("tissue_specific", []))

    overlap_ub_lof = ub & lof
    overlap_ub_ts = ub & ts
    overlap_lof_ts = lof & ts

    check("No overlap: ubiquitous / loss_of_function",
          len(overlap_ub_lof) == 0, f"Overlap: {overlap_ub_lof}")
    check("No overlap: ubiquitous / tissue_specific",
          len(overlap_ub_ts) == 0, f"Overlap: {overlap_ub_ts}")
    check("No overlap: loss_of_function / tissue_specific",
          len(overlap_lof_ts) == 0, f"Overlap: {overlap_lof_ts}")

except Exception as e:
    check("GENE_CLASSES overlap regression check", False, str(e))


# =============================================================================
section("3. VERIFY FIXED_WEIGHTS SUMS TO 1.0")
# =============================================================================

try:
    from src.models.classical.scorer import FIXED_WEIGHTS
    total = sum(FIXED_WEIGHTS.values())
    check("FIXED_WEIGHTS sums to 1.0", abs(total - 1.0) < 0.001,
          f"Sum = {total:.4f}, weights = {FIXED_WEIGHTS}")
except Exception as e:
    check("FIXED_WEIGHTS regression check", False, str(e))


# =============================================================================
section("4. VERIFY my_api.py BARE EXCEPT FIX")
# =============================================================================

try:
    content = Path("my_api.py").read_text()
    check("No bare 'except:' remaining in my_api.py",
          "except:" not in content,
          "Searched for literal 'except:' pattern")
except Exception as e:
    check("my_api.py regression check", False, str(e), severity="WARN")


# =============================================================================
section("5. NEO4J STILL ALIVE AND POPULATED")
# =============================================================================

try:
    from src.models.graph.neo4j_client import run_query
    node_counts = run_query(
        "MATCH (n) RETURN labels(n) AS labs, count(*) AS c"
    )
    counts_dict = {}
    for r in node_counts:
        labs = r["labs"]
        key = labs[0] if labs else "unknown"
        counts_dict[key] = r["c"]
    check("Neo4j connection alive", len(counts_dict) > 0,
          f"{counts_dict}")
    check("Gene/CellLine/Pathway nodes all present",
          all(k in counts_dict for k in ["Gene", "CellLine", "Pathway"]),
          f"{counts_dict}")
except Exception as e:
    check("Neo4j connectivity", False,
          f"{e} -- may need to recreate Aura instance again "
          f"(known to auto-delete on inactivity)")


# =============================================================================
section("6. RRF AND LAMBDAMART MODULES -- NO NEW BUGS")
# =============================================================================

try:
    from src.models.classical.rrf_ranker import compute_rrf_score
    import pandas as pd
    test_df = pd.DataFrame({
        "rna_score": [0.9, 0.5, 0.1],
        "protein_score": [0.8, 0.6, 0.2],
    })
    rrf = compute_rrf_score(test_df, ["rna_score", "protein_score"], k=60)
    check("RRF scorer runs on synthetic data", len(rrf) == 3,
          f"scores: {rrf.tolist()}")
    check("RRF scores are monotonic with input rank",
          rrf.iloc[0] > rrf.iloc[1] > rrf.iloc[2],
          f"scores: {rrf.tolist()} (should be descending)")
except Exception as e:
    check("RRF module check", False, str(e))

try:
    from src.models.classical.lambdamart_ranker import FEATURE_COLUMNS
    check("LambdaMART module imports", True,
          f"Feature columns: {FEATURE_COLUMNS}")
    import lightgbm
    check("lightgbm installed", True, f"version {lightgbm.__version__}")
except Exception as e:
    check("LambdaMART module check", False, str(e))


# =============================================================================
section("7. RE-RUN GENERAL BUG PATTERN SWEEP")
# =============================================================================

try:
    scan_path = Path("bug_pattern_scan.py")
    if scan_path.exists():
        result = subprocess.run(
            [sys.executable, "bug_pattern_scan.py"],
            capture_output=True, text=True, timeout=60
        )
        output = result.stdout

        has_new_weight_issues = (
            "WEIGHTS_DONT_SUM_TO_1" in output
            and "FIXED_WEIGHTS" in output
        )
        check("No FIXED_WEIGHTS regression in bug scan",
              not has_new_weight_issues,
              "Bug scan output checked for FIXED_WEIGHTS sum issue")

        has_bare_except = "my_api.py" in output and "BARE_EXCEPT" in output
        check("No bare-except regression in bug scan",
              not has_bare_except,
              "Bug scan output checked for my_api.py bare except")

        print("\n--- Full bug_pattern_scan.py output tail ---")
        print(output[-3000:])
    else:
        check("bug_pattern_scan.py exists", False,
              "Not found -- copy it back into project root to re-run",
              severity="WARN")

except Exception as e:
    check("Bug pattern re-scan", False, str(e), severity="WARN")


# =============================================================================
section("8. GIT STATUS -- CONFIRM EVERYTHING COMMITTED")
# =============================================================================

try:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, timeout=10
    )
    uncommitted = result.stdout.strip()
    check("No uncommitted changes",
          len(uncommitted) == 0,
          f"Uncommitted files:\n{uncommitted}" if uncommitted else "Clean",
          severity="WARN")

    result = subprocess.run(
        ["git", "log", "--oneline", "-5"],
        capture_output=True, text=True, timeout=10
    )
    check("Recent commit history accessible", result.returncode == 0,
          f"\n{result.stdout}")

except Exception as e:
    check("Git status check", False, str(e), severity="WARN")


# =============================================================================
section("9. MUTATION DATA FILE -- VERIFY STRUCTURE BEFORE BUILDING ON IT")
# =============================================================================

try:
    import pandas as pd
    mut_path = Path("data/member3/mutation_gene_features.csv")

    check("mutation_gene_features.csv exists", mut_path.exists(),
          str(mut_path))

    if mut_path.exists():
        df_sample = pd.read_csv(mut_path, nrows=1000)

        expected_cols = [
            "HugoSymbol", "ProteinChange", "VariantType",
            "MolecularConsequence", "Hotspot", "LikelyLoF",
            "VepClinSig", "OncogeneHighImpact"
        ]
        missing_cols = [c for c in expected_cols if c not in df_sample.columns]
        check("Expected columns present",
              len(missing_cols) == 0,
              f"Missing: {missing_cols}" if missing_cols
              else f"All present: {expected_cols}")

        score_cols = [c for c in df_sample.columns
                      if "revel" in c.lower() or "alphamissense" in c.lower()]
        check("Pathogenicity score column(s) found",
              len(score_cols) > 0,
              f"Found: {score_cols}")

        id_chain_cols = [c for c in df_sample.columns
                          if "profile" in c.lower() or "model" in c.lower()]
        check("ID chain columns (ProfileID/ModelID) present",
              len(id_chain_cols) > 0,
              f"Found: {id_chain_cols}")

        print(f"\n  Full column list ({len(df_sample.columns)} total):")
        print(f"  {list(df_sample.columns)}")

        result = subprocess.run(
            ["wc", "-l", str(mut_path)],
            capture_output=True, text=True
        )
        check("Full row count (informational)",
              True,
              result.stdout.strip(), severity="WARN")

except Exception as e:
    check("Mutation data structure check", False, str(e))


# =============================================================================
section("10. VERIFY EXISTING ID-RESOLUTION LOGIC IS REUSABLE")
# =============================================================================

try:
    result = subprocess.run(
        ["grep", "-rn", "-l", "ProfileID", "models/", "scripts/"],
        capture_output=True, text=True
    )
    files_with_profileid = result.stdout.strip().split("\n") if result.stdout.strip() else []
    check("Found existing ProfileID resolution logic to reuse",
          len(files_with_profileid) > 0,
          f"Files: {files_with_profileid}", severity="WARN")

except Exception as e:
    check("ID resolution logic search", False, str(e), severity="WARN")


# =============================================================================
# SUMMARY
# =============================================================================

section("SUMMARY")

n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
n_warn = sum(1 for _, s, _ in RESULTS if s == "WARN")

print(f"\nTotal checks: {len(RESULTS)}")
print(f"  PASS: {n_pass}")
print(f"  WARN: {n_warn}")
print(f"  FAIL: {n_fail}")

if n_fail > 0:
    print("\n--- MUST FIX BEFORE PROCEEDING ---")
    for name, status, detail in RESULTS:
        if status == "FAIL":
            print(f"  FAIL: {name}")
            if detail:
                print(f"        {detail}")

if n_warn > 0:
    print("\n--- REVIEW (may be fine) ---")
    for name, status, detail in RESULTS:
        if status == "WARN":
            print(f"  WARN: {name}")
            if detail:
                print(f"        {detail}")

print("\n" + "=" * 70)
if n_fail == 0:
    print("CLEAN -- safe to proceed with mutation scoring implementation")
else:
    print(f"{n_fail} ISSUE(S) MUST BE FIXED FIRST")
print("=" * 70)

