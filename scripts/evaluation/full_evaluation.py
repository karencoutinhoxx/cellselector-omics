import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.models.classical.ranker import rank
from src.models.classical.scorer import FIXED_WEIGHTS, classify_gene
from src.models.classical.weights_learned import (
    VALIDATION_SET,
    _apply_grid_search_pathway_weights,
    _apply_lof_mutation_weight,
    _build_name_to_cvcl,
    _precompute_scores,
    _run_optimisation,
)
from src.models.classical.scorer import load_mappings

# ─────────────────────────────────────────────────────────────────────────────
# Comprehensive comparison across four scoring configurations:
#   1. Fixed weights, no pathway
#   2. Learned weights (global, 4-D), no pathway
#   3. Learned weights (global, 5-D), WITH pathway
#   4. Learned weights (per-gene-class, 5-D), WITH pathway
#
# All four report MRR via rank() — the actual production scoring pipeline —
# not weights_learned.py's own internal (independently reimplemented, faster
# but not identical) scoring path; that path is used only to LEARN the
# weights, matching how those weights are actually learned throughout this
# codebase.
#
# _precompute_scores() is called exactly ONCE and its result reused across
# every _run_optimisation() call below (global-4D, global-5D, and three
# per-class 5-D fits). Calling optimise_weights()/optimise_weights_by_class()
# as black boxes here would redundantly re-run the ~8min-for-25-genes
# precompute step (which includes a Neo4j pathway query per gene) up to five
# times over; sharing it cuts that to once.
# ─────────────────────────────────────────────────────────────────────────────


def _weights_for_gene(gene: str, weights_by_class: dict) -> dict:
    cls = classify_gene(gene)
    w = weights_by_class.get(cls)
    if w is None:
        # optimise_weights_by_class() returns None for loss_of_function when
        # the validation set has no dedicated LOF-only genes — same
        # tissue_specific fallback it documents.
        w = weights_by_class.get("tissue_specific")
    return w


def _reciprocal_rank(gene: str, known_names: list[str], weights: dict,
                      include_pathway: bool, name_to_cvcl: dict) -> float:
    known_cvcls = {
        name_to_cvcl[n.lower()] for n in known_names if n.lower() in name_to_cvcl
    }
    if not known_cvcls:
        return 0.0

    ranked = rank(gene, top_n=None, weights=weights, include_pathway=include_pathway)
    if ranked is None or len(ranked) == 0:
        return 0.0

    for i, (_, row) in enumerate(ranked.iterrows(), 1):
        if row["cellosaurus_id"] in known_cvcls:
            return 1.0 / i
    return 0.0


def _evaluate(weight_fn, include_pathway: bool, name_to_cvcl: dict) -> dict:
    """weight_fn(gene) -> weights dict to use for that gene."""
    per_gene: dict[str, float] = {}
    for gene, known in VALIDATION_SET.items():
        per_gene[gene] = _reciprocal_rank(
            gene, known, weight_fn(gene), include_pathway, name_to_cvcl
        )

    overall = sum(per_gene.values()) / len(per_gene)

    by_class: dict[str, float | None] = {}
    for cls in ("tissue_specific", "ubiquitous", "loss_of_function"):
        genes_in_class = [g for g in VALIDATION_SET if classify_gene(g) == cls]
        by_class[cls] = (
            sum(per_gene[g] for g in genes_in_class) / len(genes_in_class)
            if genes_in_class else None
        )

    return {"per_gene": per_gene, "overall": overall, "by_class": by_class}


def _fmt(v: float | None) -> str:
    return f"{v:.4f}" if v is not None else "N/A"


def run_full_evaluation() -> None:
    print("=" * 70)
    print("CELLSELECTOR OMICS — SCORING EVALUATION")
    print("=" * 70)
    print()

    name_to_cvcl = _build_name_to_cvcl()

    t0 = time.time()
    print("Pre-computing validation scores ONCE (shared across all "
          "optimisation runs below)...")
    hpa, ach, gsm = load_mappings()
    precomputed = _precompute_scores(VALIDATION_SET, hpa, ach, gsm)
    print(f"  done in {time.time() - t0:.0f}s")

    print("\nLearning weights for configs 2-4...")
    w_global_4d = _run_optimisation(VALIDATION_SET, precomputed, label="global 4-D", include_pathway=False)
    w_global_5d = _run_optimisation(VALIDATION_SET, precomputed, label="global 5-D", include_pathway=True)

    w_by_class: dict[str, dict | None] = {}
    classified_genes: dict[str, dict] = {}
    for cls in ("tissue_specific", "ubiquitous", "loss_of_function"):
        genes_in_class = {g: v for g, v in VALIDATION_SET.items() if classify_gene(g) == cls}
        classified_genes[cls] = genes_in_class
        if genes_in_class:
            w_by_class[cls] = _run_optimisation(genes_in_class, precomputed, label=cls, include_pathway=True)
        else:
            print(f"\n[{cls}] No dedicated genes in validation set; skipping.")
            w_by_class[cls] = None

    # Config 4's pathway weight per class is grid-search-verified, not
    # SLSQP's — see weights_learned._apply_grid_search_pathway_weights.
    w_by_class = _apply_grid_search_pathway_weights(w_by_class, classified_genes, precomputed)
    # LOF class: replace the pathway vector with the mutation-primary
    # production vector (mutation=0.80) — this is what ranker.rank() uses.
    if classified_genes.get("loss_of_function"):
        w_by_class["loss_of_function"] = _apply_lof_mutation_weight(
            classified_genes["loss_of_function"], precomputed
        )
        print(f"\nLOF class production weights (mutation): {w_by_class['loss_of_function']}")

    configs = [
        ("1. Fixed weights, no pathway",
         (lambda g: {**FIXED_WEIGHTS, "pathway": 0.0}), False),
        ("2. Learned weights (global), no pathway",
         (lambda g: w_global_4d), False),
        ("3. Learned weights (global), WITH pathway",
         (lambda g: w_global_5d), True),
        ("4. Learned per-class: pathway (ts/ub) + mutation@0.80 (LOF)",
         (lambda g: _weights_for_gene(g, w_by_class)), True),
    ]

    print("\nEvaluating MRR for all four configs via rank() "
          "(the production scoring pipeline)...")
    results = {}
    for name, weight_fn, include_pathway in configs:
        t0 = time.time()
        results[name] = _evaluate(weight_fn, include_pathway, name_to_cvcl)
        print(f"  {name}: done in {time.time() - t0:.0f}s")

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    for name, weight_fn, _ in configs:
        r = results[name]
        print(f"Config: {name}")
        print(f"  Overall MRR: {r['overall']:.4f}")
        print(f"  Tissue-specific MRR: {_fmt(r['by_class']['tissue_specific'])}")
        print(f"  Ubiquitous MRR: {_fmt(r['by_class']['ubiquitous'])}")
        print(f"  Loss-of-function MRR: {_fmt(r['by_class']['loss_of_function'])}")
        if name.startswith("4."):
            print("  Weights (per gene class):")
            for cls, w in w_by_class.items():
                print(f"    {cls}: {w}")
        else:
            print(f"  Weights: {weight_fn(next(iter(VALIDATION_SET)))}")
        print()

    # ── Per-gene MRR across all four configs ─────────────────────────────────
    print("Per-Gene MRR (all configs):")
    header = f"  {'Gene':12s} {'Class':18s} " + " ".join(f"{n.split('.')[0]:>8s}" for n, _, _ in configs)
    print(header)
    for gene in VALIDATION_SET:
        cls = classify_gene(gene)
        row = f"  {gene:12s} {cls:18s} "
        row += " ".join(f"{results[n]['per_gene'][gene]:>8.3f}" for n, _, _ in configs)
        print(row)
    print()

    # ── Config 4 vs Config 2 comparison (the specific claim this feature
    # makes: per-class pathway weighting beats the pathway-free baseline) ────
    before = results["2. Learned weights (global), no pathway"]["per_gene"]
    after = results["4. Learned per-class: pathway (ts/ub) + mutation@0.80 (LOF)"]["per_gene"]

    print("Per-Gene Comparison (Config 4 vs Config 2):")
    print(f"  {'Gene':12s} {'Class':18s} {'Before':8s} {'After':8s} {'Change':8s}")
    n_improved = n_degraded = n_unchanged = 0
    for gene in VALIDATION_SET:
        b, a = before[gene], after[gene]
        diff = a - b
        cls = classify_gene(gene)
        if diff > 1e-9:
            n_improved += 1
            marker = "+"
        elif diff < -1e-9:
            n_degraded += 1
            marker = "-"
        else:
            n_unchanged += 1
            marker = "="
        print(f"  {gene:12s} {cls:18s} {b:<8.3f} {a:<8.3f} {marker}{abs(diff):.3f}")

    print()
    print(f"Genes improved: {n_improved}/{len(VALIDATION_SET)}")
    print(f"Genes degraded: {n_degraded}/{len(VALIDATION_SET)}")
    print(f"Genes unchanged: {n_unchanged}/{len(VALIDATION_SET)}")


if __name__ == "__main__":
    run_full_evaluation()

