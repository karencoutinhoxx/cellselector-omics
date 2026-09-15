import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.models.classical.ranker import rank
from src.models.classical.weights_learned import VALIDATION_SET, _build_name_to_cvcl

# ─────────────────────────────────────────────────────────────────────────────
# A/B comparison of scoring WITH and WITHOUT pathway_activity_score, isolating
# its effect on Mean Reciprocal Rank against the 25-gene validation set.
#
# Two bugs fixed from the original draft of this script, both of which would
# have made every result silently read 0.000 / no improvement rather than
# erroring, hiding the real problem:
#
# 1. ID-space mismatch: VALIDATION_SET's known-cell-line values are official
#    NAMES (e.g. "HCC827"), but rank()'s output is keyed by cellosaurus_id
#    (e.g. "CVCL_2063") — comparing them directly never matches. Reused
#    weights_learned._build_name_to_cvcl(), the same translation
#    optimise_weights() already relies on for its own MRR computation.
# 2. use_pathway was accepted but never actually used — evaluate_mrr always
#    called rank() the same way regardless of the flag, so "with" and
#    "without" would always be numerically identical. Wired it to
#    rank()'s include_pathway parameter (added alongside pathway_scorer.py).
# ─────────────────────────────────────────────────────────────────────────────

_NAME_TO_CVCL = _build_name_to_cvcl()


def evaluate_mrr(gene: str, known_names: list[str], use_pathway: bool = True) -> float:
    """Reciprocal rank of the first known cell line found in the ranking."""
    known_cvcls = {
        _NAME_TO_CVCL[n.lower()] for n in known_names if n.lower() in _NAME_TO_CVCL
    }
    if not known_cvcls:
        return 0.0

    ranked = rank(gene, top_n=None, include_pathway=use_pathway)
    if ranked is None or len(ranked) == 0:
        return 0.0

    for i, (_, row) in enumerate(ranked.iterrows(), 1):
        if row["cellosaurus_id"] in known_cvcls:
            return 1.0 / i
    return 0.0


def run_comparison() -> None:
    print("=== Scoring Comparison: With vs Without Pathway ===")
    print()

    results_with = []
    results_without = []

    for gene, known in VALIDATION_SET.items():
        mrr_with = evaluate_mrr(gene, known, use_pathway=True)
        mrr_without = evaluate_mrr(gene, known, use_pathway=False)
        results_with.append(mrr_with)
        results_without.append(mrr_without)

        diff = mrr_with - mrr_without
        marker = "+" if diff > 0 else ("-" if diff < 0 else "=")
        print(f"  {gene:10s}  with={mrr_with:.3f}  "
              f"without={mrr_without:.3f}  {marker}{abs(diff):.3f}")

    avg_with = sum(results_with) / len(results_with)
    avg_without = sum(results_without) / len(results_without)

    print()
    print(f"  Average MRR WITH pathway:    {avg_with:.4f}")
    print(f"  Average MRR WITHOUT pathway: {avg_without:.4f}")
    print(f"  Improvement: {avg_with - avg_without:+.4f}")


if __name__ == "__main__":
    run_comparison()

