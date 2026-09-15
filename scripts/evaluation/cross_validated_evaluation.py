import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import numpy as np

from config.config import OUTPUTS_DIR
from src.models.classical.scorer import classify_gene, load_mappings, rank_sort
from src.models.classical.copy_number_scorer import (
    AMPLIFICATION_COPY_NUMBER_WEIGHT,
    AMPLIFICATION_DRIVEN_GENES,
    apply_amplification_copy_number_weight,
)
from src.models.classical.rwr_scorer import apply_rwr_weight
from src.models.classical.ranker import WILD_TYPE_MUTATION_PENALTY, WILD_TYPE_PREFERRED_GENES
from src.models.classical.weights_learned import (
    VALIDATION_SET,
    _GRID_SEARCH_PATHWAY_WEIGHTS,
    _LOF_PRODUCTION_MUTATION_WEIGHT,
    _LOF_PRODUCTION_PATHWAY_WEIGHT,
    _build_name_to_cvcl,
    _precompute_scores,
    _run_optimisation,
)

# Per-gene RRs for A/B/C are persisted here so other scripts (e.g.
# compare_ranking_methods.py's significance tests) can reuse them without
# re-running this file's ~25-fold SLSQP optimization loops per config.
RESULTS_PATH = OUTPUTS_DIR / "cross_validated_loo_results.json"

# ─────────────────────────────────────────────────────────────────────────────
# Leave-one-out cross-validation. Every MRR reported so far (full_evaluation.py,
# pathway_grid_search.py, the SLSQP per-class fits) is computed IN-SAMPLE —
# weights fit on all 25 genes, then scored on those same 25 genes. That
# doesn't measure generalization; a weight vector can overfit a small,
# rank-based validation set. This holds out each gene in turn, refits
# weights on the other 24, and scores ONLY the held-out gene — the average
# reciprocal rank across all 25 folds is the cross-validated MRR.
#
# Reuses weights_learned._run_optimisation (11 starting points, including a
# pathway-heavy one) for every fold's optimization, rather than a fresh,
# more weakly-started optimizer. This session already found SLSQP can get
# stuck on the pathway dimension's step-function MRR surface when starting
# points are too sparse — running that risk silently across 75 separate
# optimizations (25 folds x 3 configs) would undermine exactly the rigor
# this script exists to add.
# ─────────────────────────────────────────────────────────────────────────────


def _reciprocal_rank_for_gene(
    gene: str, weights: dict, scores_cache: dict, name_to_cvcl: dict
) -> float:
    if gene not in scores_cache or scores_cache[gene] is None:
        return 0.0
    df = scores_cache[gene].copy()

    # NOTE: this formula does NOT include ranker.rank()'s tissue_specific
    # +0.05 mutation bonus — a pre-existing gap between this script's test
    # formula and production, predating this change, not fixed here (out
    # of scope). The wild-type penalty below IS added, since it's what
    # this specific change needs STEP 4's re-eval to actually reflect.
    wild_type_penalty = (
        WILD_TYPE_MUTATION_PENALTY * df.get("mutation_impact_score", 0.0)
        if gene in WILD_TYPE_PREFERRED_GENES else 0.0
    )
    df["test_score"] = (
        weights.get("rna", 0.0)     * df["rna_score"]
        + weights.get("protein", 0.0) * df["protein_score"]
        + weights.get("quality", 0.0) * df["quality_score"]
        + weights.get("context", 0.0) * df["context_score"]
        + weights.get("pathway", 0.0) * df.get("pathway_activity_score", 0.0)
        + weights.get("mutation", 0.0) * df.get("mutation_impact_score", 0.0)
        + weights.get("copy_number", 0.0) * df.get("copy_number_score", 0.0)
        + weights.get("rwr", 0.0) * df.get("rwr_score", 0.0)
        + df["geo_confirmation"]
        - wild_type_penalty
    ).clip(0.0, 1.0)

    # Multi-key tie-break (scorer.rank_sort): test_score is clipped to 1.0,
    # so mutated-LOF / strong-expression lines saturate — the raw
    # mutation_impact_score / rna_score / quality_score break those ties
    # before cellosaurus_id does. Also puts cellosaurus_id back as a column.
    df = rank_sort(df, "test_score")

    known = {
        name_to_cvcl[n.lower()] for n in VALIDATION_SET[gene] if n.lower() in name_to_cvcl
    }
    if not known:
        return 0.0

    for i, (_, row) in enumerate(df.iterrows(), 1):
        if row["cellosaurus_id"] in known:
            return 1.0 / i
    return 0.0


def leave_one_out_evaluation():
    print("Precomputing all scores (one-time)...")
    hpa, ach, gsm = load_mappings()
    scores_cache = _precompute_scores(VALIDATION_SET, hpa, ach, gsm)
    name_to_cvcl = _build_name_to_cvcl()

    all_genes = list(VALIDATION_SET.keys())
    gene_classes = {g: classify_gene(g) for g in all_genes}

    # ── Config A: global 4D, no pathway ───────────────────────────────────
    print("\n" + "=" * 60)
    print("CONFIG A: Global weights, no pathway (LOO-CV)")
    print("=" * 60)
    per_gene_a: dict[str, float] = {}
    for hold_out in all_genes:
        train = {g: VALIDATION_SET[g] for g in all_genes if g != hold_out}
        weights = _run_optimisation(train, scores_cache, label=f"A/-{hold_out}", include_pathway=False)
        rr = _reciprocal_rank_for_gene(hold_out, weights, scores_cache, name_to_cvcl)
        per_gene_a[hold_out] = rr
        print(f"  {hold_out:10s}  held-out RR={rr:.3f}")
    cv_mrr_a = float(np.mean(list(per_gene_a.values())))
    print(f"\n  LOO-CV MRR (no pathway): {cv_mrr_a:.4f}")

    # ── Config B: global 5D, with pathway ─────────────────────────────────
    print("\n" + "=" * 60)
    print("CONFIG B: Global weights, WITH pathway (LOO-CV)")
    print("=" * 60)
    per_gene_b: dict[str, float] = {}
    for hold_out in all_genes:
        train = {g: VALIDATION_SET[g] for g in all_genes if g != hold_out}
        weights = _run_optimisation(train, scores_cache, label=f"B/-{hold_out}", include_pathway=True)
        rr = _reciprocal_rank_for_gene(hold_out, weights, scores_cache, name_to_cvcl)
        per_gene_b[hold_out] = rr
        print(f"  {hold_out:10s}  held-out RR={rr:.3f}")
    cv_mrr_b = float(np.mean(list(per_gene_b.values())))
    print(f"\n  LOO-CV MRR (with pathway): {cv_mrr_b:.4f}")

    # ── Config C: per-class 4D baseline + grid pathway (ts/ub) OR
    #    mutation@0.80 (LOF) — matches production ranker.rank() ────────────
    print("\n" + "=" * 60)
    print("CONFIG C: Per-class weights + grid pathway / mutation (LOO-CV)")
    print("=" * 60)
    per_gene_c: dict[str, float] = {}
    for hold_out in all_genes:
        hold_out_class = gene_classes[hold_out]
        cls_train = {
            g: VALIDATION_SET[g] for g in all_genes
            if g != hold_out and gene_classes[g] == hold_out_class
        }

        if len(cls_train) >= 2:
            baseline_4d = _run_optimisation(
                cls_train, scores_cache, label=f"C/-{hold_out}", include_pathway=False
            )
        else:
            baseline_4d = {"rna": 0.4, "protein": 0.2, "quality": 0.3, "context": 0.1}

        # Same rescale construction as weights_learned's
        # _apply_grid_search_pathway_weights / _apply_lof_mutation_weight:
        # shrink the 4D baseline rather than tacking the extra term on top
        # (which would push the sum past 1.0).
        if hold_out_class == "loss_of_function":
            mw, pw = _LOF_PRODUCTION_MUTATION_WEIGHT, _LOF_PRODUCTION_PATHWAY_WEIGHT
            weights = {k: v * (1 - mw - pw) for k, v in baseline_4d.items()}
            weights["mutation"] = mw
            if pw > 0:
                weights["pathway"] = pw
            extra_desc = f"mut={mw:.2f}/path={pw:.2f}"
        elif hold_out in AMPLIFICATION_DRIVEN_GENES:
            # Same fixed-constant treatment as production ranker.rank() (see
            # copy_number_scorer.apply_amplification_copy_number_weight) —
            # rescales THIS FOLD's refit 4D baseline, not the full-sample one,
            # so it's a genuine held-out test, not a leak of the held-out
            # gene's own contribution to the baseline fit.
            weights = apply_amplification_copy_number_weight(baseline_4d)
            extra_desc = f"copy_number={AMPLIFICATION_COPY_NUMBER_WEIGHT:.2f}"
        else:
            pw = _GRID_SEARCH_PATHWAY_WEIGHTS.get(hold_out_class, 0.0)
            weights = {k: v * (1 - pw) for k, v in baseline_4d.items()}
            weights["pathway"] = pw
            extra_desc = f"path={pw:.2f}"

        # RWR blending, all classes — see rwr_scorer.RWR_WEIGHT_BY_CLASS /
        # apply_rwr_weight. Applied last, on top of whatever branch above
        # produced (mutation-primary LOF vector, copy_number-primary
        # amplification vector, or plain pathway-weighted vector) — same
        # order the grid search itself used to find these weights.
        weights = apply_rwr_weight(weights, hold_out_class)
        extra_desc += f" rwr={weights.get('rwr', 0.0):.2f}"

        rr = _reciprocal_rank_for_gene(hold_out, weights, scores_cache, name_to_cvcl)
        per_gene_c[hold_out] = rr
        print(f"  {hold_out:10s}  class={hold_out_class:18s}  {extra_desc:16s}  held-out RR={rr:.3f}")
    cv_mrr_c = float(np.mean(list(per_gene_c.values())))
    print(f"\n  LOO-CV MRR (per-class + grid pathway / mutation): {cv_mrr_c:.4f}")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CROSS-VALIDATED EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Config A (global, no pathway/mutation):        {cv_mrr_a:.4f}")
    print(f"  Config B (global, with pathway):               {cv_mrr_b:.4f}")
    print(f"  Config C (per-class: pathway ts/ub, mut@0.80+path@{_LOF_PRODUCTION_PATHWAY_WEIGHT:.2f} LOF): {cv_mrr_c:.4f}")
    print()
    print("  For reference only — NOT the same quantity: the earlier")
    print("  IN-SAMPLE (not cross-validated) Config 2 result from")
    print("  full_evaluation.py was 0.1826 (weights fit AND scored on the")
    print("  same 25 genes). It measures fit, not generalization; comparing")
    print("  it to cv_mrr_a is illustrative, not a like-for-like delta.")
    print()

    diff = cv_mrr_c - cv_mrr_a
    verb = "IMPROVES" if diff > 0 else ("DEGRADES" if diff < 0 else "leaves unchanged")
    print(f"  Per-class pathway+mutation scoring {verb} cross-validated MRR by {diff:+.4f}")

    # Per-class breakdown — the LOF class is where mutation scoring should move
    lof = [g for g in all_genes if gene_classes[g] == "loss_of_function"]
    if lof:
        lof_a = float(np.mean([per_gene_a[g] for g in lof]))
        lof_c = float(np.mean([per_gene_c[g] for g in lof]))
        print(f"  LOF-only ({len(lof)} genes): Config A {lof_a:.4f} -> Config C {lof_c:.4f}  ({lof_c - lof_a:+.4f})")

    results = {
        "cv_mrr":     {"A": cv_mrr_a, "B": cv_mrr_b, "C": cv_mrr_c},
        "per_gene_rr": {"A": per_gene_a, "B": per_gene_b, "C": per_gene_c},
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved per-gene results → {RESULTS_PATH}")

    return results


if __name__ == "__main__":
    leave_one_out_evaluation()

