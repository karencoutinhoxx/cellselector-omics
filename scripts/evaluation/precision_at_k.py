import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.classical.ranker import rank
from src.models.classical.scorer import classify_gene
from src.models.classical.weights_learned import VALIDATION_SET, _build_name_to_cvcl

# ─────────────────────────────────────────────────────────────────────────────
# Precision@k: does the correct answer appear anywhere in a realistic
# shortlist (top 1 / 5 / 10 / 20)? A binary per gene — 1 if any known-correct
# cell line is in the top k, else 0 — averaged within each gene class.
#
# Uses the PRODUCTION rank() pipeline (per-class weights, mutation@0.80 for
# LOF, etc.), scored in-sample on the 30-gene validation set. This is a
# more intuitive framing than MRR for a results summary; it is NOT a
# cross-validated number and should not be compared like-for-like with the
# LOO-CV MRR figures.
#
# Efficiency note: rank() is called ONCE per gene at k = max(K_VALUES) and
# the result sliced for each k, rather than re-ranking per k (each rank()
# call re-scores every candidate + hits Neo4j for pathway activity).
# ─────────────────────────────────────────────────────────────────────────────

K_VALUES = [1, 5, 10, 20]


def precision_at_k(gene: str, known_cvcls: set, k: int) -> int:
    """1 if any known-correct cell line is in this gene's top-k, else 0."""
    r = rank(gene, top_n=k)
    if r is None or len(r) == 0:
        return 0
    return 1 if any(c in known_cvcls for c in r["cellosaurus_id"]) else 0


def run_precision_evaluation():
    name_to_cvcl = _build_name_to_cvcl()  # keys are lowercased
    ks = K_VALUES
    kmax = max(ks)

    results_by_class: dict[str, dict[int, list[int]]] = {}
    per_gene: dict[str, dict[int, int]] = {}

    for gene in VALIDATION_SET:
        known = {
            name_to_cvcl[n.lower()]
            for n in VALIDATION_SET[gene]
            if n.lower() in name_to_cvcl
        }
        if not known:
            print(f"  [skip] {gene}: no validation cell line resolves")
            continue

        r = rank(gene, top_n=kmax)
        ordered = list(r["cellosaurus_id"]) if r is not None and len(r) else []

        gene_class = classify_gene(gene)
        results_by_class.setdefault(gene_class, {k: [] for k in ks})
        per_gene[gene] = {}
        for k in ks:
            hit = 1 if any(c in known for c in ordered[:k]) else 0
            results_by_class[gene_class][k].append(hit)
            per_gene[gene][k] = hit

    # ── per-gene table ────────────────────────────────────────────────────
    print("\nPer-gene hit@k (1 = a known line is in the top k):")
    print(f"  {'Gene':10s} {'Class':18s} " + " ".join(f"@{k:<3d}" for k in ks))
    for gene, hits in per_gene.items():
        print(f"  {gene:10s} {classify_gene(gene):18s} "
              + " ".join(f"{hits[k]:<4d}" for k in ks))

    # ── by class ──────────────────────────────────────────────────────────
    print("\nPrecision@k by gene class (production scoring, in-sample, 30-gene set):")
    print(f"  {'Class':20s} {'n':>3s}  " + "  ".join(f"P@{k:<4d}" for k in ks))
    overall = {k: [] for k in ks}
    for cls in ("tissue_specific", "loss_of_function", "ubiquitous"):
        data = results_by_class.get(cls)
        if not data:
            continue
        n = len(data[ks[0]])
        cells = []
        for k in ks:
            p = sum(data[k]) / n
            cells.append(f"{p:.3f}")
            overall[k].extend(data[k])
        print(f"  {cls:20s} {n:>3d}  " + "  ".join(f"{c:>6s}" for c in cells))

    print("\nOverall:")
    for k in ks:
        p = sum(overall[k]) / len(overall[k])
        print(f"  P@{k:<3d}: {p:.3f}   ({sum(overall[k])}/{len(overall[k])} genes)")


if __name__ == "__main__":
    run_precision_evaluation()

