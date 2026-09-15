import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import numpy as np

from src.models.classical.scorer import load_mappings, rank_sort
from src.models.classical.weights_learned import VALIDATION_SET, _build_name_to_cvcl, _precompute_scores
from src.models.classical.rrf_ranker import compute_rrf_score
from src.models.classical.lambdamart_ranker import (
    build_training_data, train_lambdamart, score_with_lambdamart, FEATURE_COLUMNS
)
from scripts.evaluation.cross_validated_evaluation import RESULTS_PATH

# ─────────────────────────────────────────────────────────────────────────────
# Compares Reciprocal Rank Fusion and LambdaMART against the existing
# weighted-sum LOO-CV baseline (Config A, MRR=0.2421 on the corrected
# mappings) using the SAME 25-gene validation set and the SAME reciprocal-
# rank computation convention as cross_validated_evaluation.py, so the
# numbers are directly comparable.
#
# RRF (Configs D/E) has no free parameters fit from data — k is a constant
# chosen a priori, not learned — so there is nothing to hold out and refit
# per fold. It's evaluated directly on all 25 genes, which is the correct
# way to test a parameter-free method, not a shortcut around LOO-CV. Only
# LambdaMART (Config F) has actual learned parameters (tree structure) and
# therefore genuinely needs a fresh model trained per fold on the other 24
# genes, which is what's implemented below.
# ─────────────────────────────────────────────────────────────────────────────


def paired_bootstrap_test(rrs_a, rrs_b, n_bootstrap=10000, seed=42):
    """
    Paired bootstrap significance test: is the MRR difference between two
    methods (paired by gene) larger than chance?

    rrs_a, rrs_b: lists of per-gene reciprocal ranks for the two methods,
    same gene order (caller is responsible for aligning them — see
    _aligned_rrs below, which aligns by gene NAME, not position).

    Returns: (observed MRR difference A-B, p-value, (ci_low, ci_high))

    Note on the p-value: this resamples the OBSERVED paired differences
    (not a permutation under a null model), then reports the fraction of
    resamples crossing zero, doubled for a two-sided estimate. This is a
    standard, widely-used approximate bootstrap hypothesis test (see e.g.
    Efron & Tibshirani), not a formal permutation test — worth keeping in
    mind given N=25 genes is a small sample for either approach.
    """
    rng = np.random.default_rng(seed)
    rrs_a = np.array(rrs_a)
    rrs_b = np.array(rrs_b)
    n = len(rrs_a)

    observed_diff = rrs_a.mean() - rrs_b.mean()

    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diff = rrs_a[idx].mean() - rrs_b[idx].mean()
        diffs.append(diff)
    diffs = np.array(diffs)

    if observed_diff > 0:
        p_value = np.mean(diffs <= 0) * 2
    else:
        p_value = np.mean(diffs >= 0) * 2
    p_value = min(p_value, 1.0)

    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

    return observed_diff, p_value, (ci_low, ci_high)


def _aligned_rrs(per_gene_a: dict, per_gene_b: dict) -> tuple[list, list, list]:
    """
    Align two per-gene RR dicts by gene NAME (not position/order), so a
    mismatch between VALIDATION_SET's current gene set and whatever was
    saved to disk for Config A can't silently misalign the pairing — the
    exact bug class this whole project keeps tripping over.
    """
    common = sorted(set(per_gene_a) & set(per_gene_b))
    missing_a = set(per_gene_b) - set(per_gene_a)
    missing_b = set(per_gene_a) - set(per_gene_b)
    if missing_a or missing_b:
        print(f"  [warning] gene sets don't fully match — "
              f"{len(missing_a)} genes missing from A, {len(missing_b)} missing "
              f"from B; comparing only the {len(common)} genes both have.")
    return common, [per_gene_a[g] for g in common], [per_gene_b[g] for g in common]


def compute_rr(df, score_col, known_cvcls):
    """Reciprocal rank for one gene given a score column."""
    # Multi-key tie-break (scorer.rank_sort): RRF / LambdaMART scores can tie,
    # and for a fused mutation source the saturated-LOF blocks recur — raw
    # mutation_impact_score / rna_score / quality_score break ties before
    # cellosaurus_id. Also puts cellosaurus_id back as a column.
    df_sorted = rank_sort(df, score_col)
    for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
        if row["cellosaurus_id"] in known_cvcls:
            return 1.0 / i
    return 0.0


def run_comparison():
    print("Precomputing scores...")
    hpa, ach, gsm = load_mappings()
    scores_cache = _precompute_scores(VALIDATION_SET, hpa, ach, gsm)
    name_to_cvcl = _build_name_to_cvcl()
    all_genes = list(VALIDATION_SET.keys())

    def known_cvcls_for(gene):
        # _build_name_to_cvcl()'s keys are lowercased; VALIDATION_SET names
        # are mixed-case ("HCC827") — without .lower() this never matches.
        s = set()
        for name in VALIDATION_SET[gene]:
            cvcl = name_to_cvcl.get(name.lower())
            if cvcl:
                s.add(cvcl)
        return s

    def run_rrf_config(label, columns, k):
        print(f"\n=== {label} ===")
        per_gene: dict[str, float] = {}
        for gene in all_genes:
            if gene not in scores_cache or scores_cache[gene] is None:
                per_gene[gene] = 0.0
                print(f"  {gene:10s}  RR=0.000 (no precomputed data)")
                continue
            df = scores_cache[gene].copy()
            cols = [c for c in columns if c in df.columns]
            df["rrf_score"] = compute_rrf_score(df, cols, k=k)
            rr = compute_rr(df, "rrf_score", known_cvcls_for(gene))
            per_gene[gene] = rr
            print(f"  {gene:10s}  RR={rr:.3f}")
        mrr = float(np.mean(list(per_gene.values())))
        print(f"  MRR ({label}): {mrr:.4f}")
        return mrr, per_gene

    base_cols = ["rna_score", "protein_score", "quality_score", "context_score"]
    pathway_cols = base_cols + ["pathway_activity_score"]
    mutation_cols = base_cols + ["mutation_impact_score"]
    all_cols = base_cols + ["pathway_activity_score", "mutation_impact_score"]
    # rwr_score AND copy_number_score added as fusable sources — full
    # feature parity with production (rna, protein, quality, context,
    # mutation, copy_number, rwr). copy_number_score was the last missing
    # source (flagged earlier as out of scope, now closed) — it's 0.0 for
    # every gene except MYCN/ERBB2/FGFR1, so its effect on these configs'
    # MRR should be small, but included for genuine parity rather than
    # assumed negligible.
    mutation_rwr_cols = mutation_cols + ["rwr_score", "copy_number_score"]  # closest
    # RRF analog to production's actual formula shape: mutation for LOF,
    # copy_number for amplification genes, rwr for every class, no pathway
    # (production's ts/lof classes don't weight pathway)
    all_rwr_cols = all_cols + ["rwr_score", "copy_number_score"]  # fully maximal: every source

    # ============================================
    # CONFIG D: RRF (4 sources, no pathway), k=60
    # ============================================
    mrr_d, per_gene_d = run_rrf_config("CONFIG D: RRF (4 sources, k=60)", base_cols, k=60)

    # ============================================
    # CONFIG D2: RRF (4 sources, no pathway), k=10 — sensitivity check
    # ============================================
    mrr_d2, per_gene_d2 = run_rrf_config("CONFIG D2: RRF (4 sources, k=10, sensitivity)", base_cols, k=10)

    # ============================================
    # CONFIG E: RRF (5 sources, WITH pathway), k=60
    # ============================================
    mrr_e, per_gene_e = run_rrf_config("CONFIG E: RRF (5 sources w/ pathway, k=60)", pathway_cols, k=60)

    # ============================================
    # CONFIG E2: RRF (5 sources, WITH pathway), k=10 — sensitivity check
    # ============================================
    mrr_e2, per_gene_e2 = run_rrf_config("CONFIG E2: RRF (5 sources w/ pathway, k=10, sensitivity)", pathway_cols, k=10)

    # ============================================
    # CONFIG G: RRF (5 sources w/ mutation), k=60
    # ============================================
    mrr_g, per_gene_g = run_rrf_config("CONFIG G: RRF (5 sources w/ mutation, k=60)", mutation_cols, k=60)

    # ============================================
    # CONFIG H: RRF (6 sources: pathway + mutation), k=60
    # ============================================
    mrr_h, per_gene_h = run_rrf_config("CONFIG H: RRF (6 sources: pathway+mutation, k=60)", all_cols, k=60)

    # ============================================
    # CONFIG I: RRF (6 sources: mutation + rwr, no pathway), k=60
    # Closest RRF analog to production's actual formula shape.
    # ============================================
    mrr_i, per_gene_i = run_rrf_config("CONFIG I: RRF (6 sources: mutation+rwr, k=60)", mutation_rwr_cols, k=60)

    # ============================================
    # CONFIG J: RRF (7 sources: pathway + mutation + rwr), k=60
    # Fully maximal RRF config — every source production now has access to.
    # ============================================
    mrr_j, per_gene_j = run_rrf_config("CONFIG J: RRF (7 sources: pathway+mutation+rwr, k=60)", all_rwr_cols, k=60)

    # ============================================
    # CONFIG F: LambdaMART (LOO-CV — retrain per fold)
    # ============================================
    print("\n=== CONFIG F: LambdaMART (LOO-CV) ===")
    per_gene_f: dict[str, float] = {}
    lgb_errors = []
    for hold_out_gene in all_genes:
        train_genes = {g: v for g, v in VALIDATION_SET.items() if g != hold_out_gene}
        try:
            X, y, groups = build_training_data(train_genes, scores_cache, name_to_cvcl)
            # Count how many training queries have zero positive labels —
            # lambdarank gets no learning signal from an all-zero group.
            offset = 0
            zero_pos_groups = 0
            for g in groups:
                if y[offset:offset + g].sum() == 0:
                    zero_pos_groups += 1
                offset += g
            model = train_lambdamart(X, y, groups)
        except Exception as exc:
            lgb_errors.append((hold_out_gene, str(exc)))
            per_gene_f[hold_out_gene] = 0.0
            print(f"  {hold_out_gene:10s}  RR=0.000  [TRAINING FAILED: {exc}]")
            continue

        if hold_out_gene not in scores_cache or scores_cache[hold_out_gene] is None:
            per_gene_f[hold_out_gene] = 0.0
            print(f"  {hold_out_gene:10s}  RR=0.000 (no precomputed data)")
            continue

        test_df = scores_cache[hold_out_gene].copy()
        test_df["lgb_score"] = score_with_lambdamart(model, test_df)
        rr = compute_rr(test_df, "lgb_score", known_cvcls_for(hold_out_gene))
        per_gene_f[hold_out_gene] = rr
        flag = f"  [{zero_pos_groups}/{len(groups)} training queries had 0 positives]" if zero_pos_groups else ""
        print(f"  {hold_out_gene:10s}  RR={rr:.3f}{flag}")
    mrr_f = float(np.mean(list(per_gene_f.values())))
    print(f"\n  LOO-CV MRR (LambdaMART): {mrr_f:.4f}")
    if lgb_errors:
        print(f"  [{len(lgb_errors)} fold(s) failed to train — see above]")

    # ============================================
    # SIGNIFICANCE: each alternative config vs Config A
    # ============================================
    print("\n" + "=" * 60)
    print("PAIRED BOOTSTRAP SIGNIFICANCE TESTS (vs Config A)")
    print("=" * 60)

    if not RESULTS_PATH.exists():
        print(f"  [ERROR] {RESULTS_PATH} not found — run "
              f"cross_validated_evaluation.py first so Config A's per-gene "
              f"RRs are on disk. Skipping significance tests.")
        per_gene_a = None
        mrr_a_display = "0.2421  [reported, not verified this run]"
    else:
        with open(RESULTS_PATH) as f:
            saved = json.load(f)
        per_gene_a = saved["per_gene_rr"]["A"]
        mrr_a_display = f"{saved['cv_mrr']['A']:.4f}"

    sig_results = {}
    if per_gene_a is not None:
        for cfg_name, per_gene in [
            ("D  (RRF, 4 src, k=60)",       per_gene_d),
            ("D2 (RRF, 4 src, k=10)",       per_gene_d2),
            ("E  (RRF, 5 src +pathway)",    per_gene_e),
            ("E2 (RRF, 5 src +pathway k10)", per_gene_e2),
            ("G  (RRF, 5 src +mutation)",   per_gene_g),
            ("H  (RRF, 6 src path+mut)",    per_gene_h),
            ("I  (RRF, 6 src mut+rwr)",     per_gene_i),
            ("J  (RRF, 7 src path+mut+rwr)", per_gene_j),
            ("F  (LambdaMART)",            per_gene_f),
        ]:
            genes, rrs_alt, rrs_a = _aligned_rrs(per_gene, per_gene_a)
            diff, p_val, (ci_lo, ci_hi) = paired_bootstrap_test(rrs_alt, rrs_a)
            sig_results[cfg_name] = (diff, p_val, ci_lo, ci_hi)
            sig = "SIGNIFICANT" if p_val < 0.05 else "not significant"
            print(f"\n  Config {cfg_name} vs Config A  (n={len(genes)} genes)")
            print(f"    Observed MRR difference (alt - A): {diff:+.4f}")
            print(f"    95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
            print(f"    p-value: {p_val:.4f}  —  {sig} at α=0.05")

    # ============================================
    # SUMMARY
    # ============================================
    print("\n" + "=" * 60)
    print("RANKING METHOD COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  Config A (weighted-sum, no pathway, LOO-CV):    {mrr_a_display}")
    print(f"  Config D  (RRF, 4 sources, k=60):               {mrr_d:.4f}")
    print(f"  Config D2 (RRF, 4 sources, k=10):               {mrr_d2:.4f}")
    print(f"  Config E  (RRF, 5 sources w/ pathway, k=60):    {mrr_e:.4f}")
    print(f"  Config E2 (RRF, 5 sources w/ pathway, k=10):    {mrr_e2:.4f}")
    print(f"  Config G  (RRF, 5 sources w/ mutation, k=60):   {mrr_g:.4f}")
    print(f"  Config H  (RRF, 6 sources pathway+mutation):    {mrr_h:.4f}")
    print(f"  Config I  (RRF, 6 sources mutation+rwr):        {mrr_i:.4f}")
    print(f"  Config J  (RRF, 7 sources pathway+mutation+rwr):{mrr_j:.4f}")
    print(f"  Config F  (LambdaMART, LOO-CV, +rwr feature):   {mrr_f:.4f}")
    print()
    print("  NOTE: Configs D/D2/E/E2 (RRF) have no fitted parameters — k is")
    print("  a fixed constant, not learned — so they're evaluated directly")
    print("  on all 25 genes rather than via leave-one-out refitting; there")
    print("  is nothing for them to overfit to. Only Config F retrains per")
    print("  fold, since it's the only method with learned parameters.")
    if sig_results:
        print()
        print("  Significance vs Config A:")
        for cfg_name, (diff, p_val, ci_lo, ci_hi) in sig_results.items():
            sig = "*" if p_val < 0.05 else " "
            print(f"    [{sig}] {cfg_name:24s} diff={diff:+.4f}  p={p_val:.4f}")
        print("    (* = significant at α=0.05; none expected given N=25 and the")
        print("     effect sizes observed — see caveats in the written report)")


if __name__ == "__main__":
    run_comparison()

