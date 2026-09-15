from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import OUTPUTS_DIR
from src.models.classical.ranker import FIXED_WEIGHTS, rank
from src.models.classical.scorer import classify_gene
from src.models.classical.weights_learned import (
    VALIDATION_SET,
    _build_name_to_cvcl,
    optimise_weights,
    optimise_weights_by_class,
)


def _gene_mrr(results: pd.DataFrame, known_cvcls: set) -> float:
    """MRR for a single gene's results."""
    rr = []
    for cvcl in known_cvcls:
        hits = results.index[results["cellosaurus_id"] == cvcl].tolist()
        rr.append(1.0 / (hits[0] + 1) if hits else 0.0)
    return float(np.mean(rr)) if rr else 0.0


def _compute_mrr(results_by_gene: dict, name_to_cvcl: dict) -> float:
    all_rr: list[float] = []
    for gene, (results, known_names) in results_by_gene.items():
        if results is None or len(results) == 0:
            continue
        known_cvcls = {name_to_cvcl.get(n.lower()) for n in known_names} - {None}
        for cvcl in known_cvcls:
            hits = results.index[results["cellosaurus_id"] == cvcl].tolist()
            all_rr.append(1.0 / (hits[0] + 1) if hits else 0.0)
    return float(np.mean(all_rr)) if all_rr else 0.0


def _hit_rate(results_by_gene: dict, name_to_cvcl: dict, k: int) -> float:
    if not results_by_gene:
        return 0.0
    hits = 0
    for gene, (results, known_names) in results_by_gene.items():
        if results is None or len(results) == 0:
            continue
        known_cvcls = {name_to_cvcl.get(n.lower()) for n in known_names} - {None}
        top_k = set(results.head(k)["cellosaurus_id"])
        if known_cvcls & top_k:
            hits += 1
    return hits / len(results_by_gene)


def _build_per_gene_detail(
    results_by_gene: dict,
    name_to_cvcl: dict,
    label: str,
) -> dict:
    """Build per-gene rank and MRR detail dict."""
    detail = {}
    for gene, (results, known_names) in results_by_gene.items():
        lines = []
        known_cvcls_found = set()
        for name in known_names:
            cvcl = name_to_cvcl.get(name.lower())
            if cvcl is None:
                lines.append({"name": name, "status": "not_in_spine", "rank": None})
                continue
            known_cvcls_found.add(cvcl)
            if results is None or len(results) == 0:
                lines.append({"name": name, "cvcl": cvcl, "status": "no_expression_data", "rank": None})
                continue
            hits = results.index[results["cellosaurus_id"] == cvcl].tolist()
            rank_pos = hits[0] + 1 if hits else None
            lines.append({
                "name":   name,
                "cvcl":   cvcl,
                "status": "found" if rank_pos else "not_in_results",
                "rank":   rank_pos,
            })
        gene_mrr = (
            _gene_mrr(results, known_cvcls_found)
            if results is not None and len(results) > 0 and known_cvcls_found
            else 0.0
        )
        detail[gene] = {"lines": lines, "gene_mrr": round(gene_mrr, 4)}
    return detail


def _class_mrr(results_by_gene: dict, name_to_cvcl: dict, gene_class: str) -> float:
    """MRR restricted to genes of a given class."""
    subset = {
        g: v for g, v in results_by_gene.items()
        if classify_gene(g) == gene_class
    }
    return _compute_mrr(subset, name_to_cvcl)


def evaluate_both() -> dict:
    """
    Compare FIXED_WEIGHTS vs per-class LEARNED weights on the 25-gene validation set.

    Reports:
      - Overall MRR (fixed vs learned)
      - Per-class MRR (tissue_specific / ubiquitous / loss_of_function)
      - Per-gene MRR table with ranks
    """
    name_to_cvcl = _build_name_to_cvcl()

    # ── Spine presence audit ──────────────────────────────────────────────────
    total_lines = sum(len(v) for v in VALIDATION_SET.values())
    found_count = sum(
        1 for names in VALIDATION_SET.values()
        for n in names if name_to_cvcl.get(n.lower())
    )
    missing = [
        (gene, n)
        for gene, names in VALIDATION_SET.items()
        for n in names if not name_to_cvcl.get(n.lower())
    ]
    print(f"Spine audit: {found_count}/{total_lines} known lines present "
          f"({total_lines - found_count} missing)")
    if missing:
        for gene, n in missing:
            print(f"  MISSING  {gene}: {n}")

    # ── Optimise weights ──────────────────────────────────────────────────────
    print("\n=== Optimising overall weights ===")
    learned_weights = optimise_weights(VALIDATION_SET)

    print("\n=== Optimising per-class weights ===")
    class_weights = optimise_weights_by_class(VALIDATION_SET)

    # ── Rank with each weight set ─────────────────────────────────────────────
    print("\n=== Ranking with FIXED weights ===")
    fixed_results: dict = {}
    for gene, known_names in VALIDATION_SET.items():
        print(f"  {gene}...")
        fixed_results[gene] = (rank(gene, top_n=None, weights=FIXED_WEIGHTS), known_names)

    print("\n=== Ranking with LEARNED (overall) weights ===")
    learned_results: dict = {}
    for gene, known_names in VALIDATION_SET.items():
        print(f"  {gene}...")
        learned_results[gene] = (rank(gene, top_n=None, weights=learned_weights), known_names)

    print("\n=== Ranking with per-class LEARNED weights ===")
    class_results: dict = {}
    for gene, known_names in VALIDATION_SET.items():
        gc = classify_gene(gene)
        w  = class_weights.get(gc) or class_weights.get("tissue_specific") or FIXED_WEIGHTS
        print(f"  {gene} [{gc}]...")
        class_results[gene] = (rank(gene, top_n=None, weights=w), known_names)

    # ── Overall metrics ───────────────────────────────────────────────────────
    fixed_mrr    = _compute_mrr(fixed_results,   name_to_cvcl)
    learned_mrr  = _compute_mrr(learned_results, name_to_cvcl)
    class_mrr    = _compute_mrr(class_results,   name_to_cvcl)

    fixed_hr5    = _hit_rate(fixed_results,   name_to_cvcl, k=5)
    fixed_hr10   = _hit_rate(fixed_results,   name_to_cvcl, k=10)
    learned_hr5  = _hit_rate(learned_results, name_to_cvcl, k=5)
    learned_hr10 = _hit_rate(learned_results, name_to_cvcl, k=10)
    class_hr5    = _hit_rate(class_results,   name_to_cvcl, k=5)
    class_hr10   = _hit_rate(class_results,   name_to_cvcl, k=10)

    # ── Per-class MRR breakdown ───────────────────────────────────────────────
    gene_classes = ["tissue_specific", "ubiquitous", "loss_of_function"]
    fixed_class_mrr   = {gc: _class_mrr(fixed_results,   name_to_cvcl, gc) for gc in gene_classes}
    learned_class_mrr = {gc: _class_mrr(learned_results, name_to_cvcl, gc) for gc in gene_classes}
    class_class_mrr   = {gc: _class_mrr(class_results,   name_to_cvcl, gc) for gc in gene_classes}

    fixed_detail  = _build_per_gene_detail(fixed_results,  name_to_cvcl, "fixed")
    learned_detail = _build_per_gene_detail(learned_results, name_to_cvcl, "learned")
    class_detail  = _build_per_gene_detail(class_results,  name_to_cvcl, "class")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS  (25-gene validation set)")
    print("=" * 70)
    print(f"{'Metric':<30} {'FIXED':>10} {'LEARNED':>10} {'CLASS-W':>10}")
    print("-" * 62)
    print(f"{'Overall MRR':<30} {fixed_mrr:>10.4f} {learned_mrr:>10.4f} {class_mrr:>10.4f}")
    print(f"{'Hit rate @5':<30} {fixed_hr5:>10.4f} {learned_hr5:>10.4f} {class_hr5:>10.4f}")
    print(f"{'Hit rate @10':<30} {fixed_hr10:>10.4f} {learned_hr10:>10.4f} {class_hr10:>10.4f}")

    print(f"\n── MRR by gene class ──")
    for gc in gene_classes:
        n_genes = sum(1 for g in VALIDATION_SET if classify_gene(g) == gc)
        lof_note = "  [ranking inverted for LOF; high-expr = control]" if gc == "loss_of_function" else ""
        print(f"  {gc:<22} ({n_genes} genes)  "
              f"fixed={fixed_class_mrr[gc]:.4f}  "
              f"learned={learned_class_mrr[gc]:.4f}  "
              f"class-w={class_class_mrr[gc]:.4f}{lof_note}")

    # Per-gene table grouped by class
    print(f"\n{'Gene':<8} {'Class':<18} {'Fix':>6} {'Lrn':>6} {'CW':>6}   Known lines & ranks (fix→lrn→cw)")
    print("-" * 110)
    gene_mrrs = []
    for gc in gene_classes:
        genes_in_class = [g for g in VALIDATION_SET if classify_gene(g) == gc]
        if not genes_in_class:
            continue
        print(f"  ── {gc} ──")
        for gene in genes_in_class:
            fmrr = fixed_detail[gene]["gene_mrr"]
            lmrr = learned_detail[gene]["gene_mrr"]
            cmrr = class_detail[gene]["gene_mrr"]
            gene_mrrs.append((gene, gc, fmrr, lmrr, cmrr))
            line_parts = []
            for fl, ll, cl in zip(
                fixed_detail[gene]["lines"],
                learned_detail[gene]["lines"],
                class_detail[gene]["lines"],
            ):
                name_s = fl["name"]
                fr = fl["rank"] or ("?" if fl["status"] == "not_in_spine" else ">all")
                lr = ll["rank"] or ("?" if ll["status"] == "not_in_spine" else ">all")
                cr = cl["rank"] or ("?" if cl["status"] == "not_in_spine" else ">all")
                flag = "[MISS]" if fl["status"] == "not_in_spine" else ""
                line_parts.append(f"{name_s}({fr}→{lr}→{cr}){flag}")
            print(f"  {gene:<6} {gc:<18} {fmrr:>6.3f} {lmrr:>6.3f} {cmrr:>6.3f}   {', '.join(line_parts)}")

    # Best / worst by class-weighted MRR
    sorted_by_mrr = sorted(gene_mrrs, key=lambda x: x[4], reverse=True)
    print("\n── Best genes (class-weighted MRR ≥ 0.10) ──")
    good = [t for t in sorted_by_mrr if t[4] >= 0.10]
    if good:
        for g, gc, f, l, c in good:
            print(f"  {g:<8} [{gc}]  MRR={c:.4f}")
    else:
        print("  (none above 0.10)")

    print("\n── Zero-MRR genes (class-weighted) ──")
    zero = [t for t in sorted_by_mrr if t[4] == 0.0]
    for g, gc, f, l, c in zero:
        names_in_spine = [ln["name"] for ln in class_detail[g]["lines"]
                          if ln["status"] != "not_in_spine"]
        all_missing = all(ln["status"] == "not_in_spine" for ln in class_detail[g]["lines"])
        reason = "all lines missing from spine" if all_missing else "in spine but ranked low"
        print(f"  {g:<8} [{gc}] ({reason}): {names_in_spine}")

    print(f"\n→ Class-weighted weights: {class_weights}")
    best_overall = max([("FIXED", fixed_mrr), ("LEARNED", learned_mrr), ("CLASS-W", class_mrr)],
                       key=lambda x: x[1])
    print(f"→ Best overall: {best_overall[0]}  MRR={best_overall[1]:.4f}")

    # ── Sample explain() for a LOF gene ──────────────────────────────────────
    lof_genes = [g for g in VALIDATION_SET if classify_gene(g) == "loss_of_function"]
    if lof_genes:
        from src.models.classical.ranker import explain
        sample_gene = lof_genes[0]
        sample_results = class_results[sample_gene][0]
        if sample_results is not None and len(sample_results) > 0:
            print(f"\n── Sample explain() for {sample_gene} (LOF gene) ──")
            top_row = sample_results.iloc[0].copy()
            top_row["gene"] = sample_gene
            print(explain(top_row))

    # ── Save ─────────────────────────────────────────────────────────────────
    output = {
        "n_genes":                len(VALIDATION_SET),
        "n_known_lines":          total_lines,
        "n_found_in_spine":       found_count,
        "n_missing_from_spine":   total_lines - found_count,
        "missing_from_spine":     [{"gene": g, "name": n} for g, n in missing],
        "fixed_weights":          FIXED_WEIGHTS,
        "learned_weights":        {k: round(float(v), 6) for k, v in learned_weights.items()},
        "class_weights": {
            gc: ({k: round(float(v), 6) for k, v in w.items()} if w else None)
            for gc, w in class_weights.items()
        },
        "fixed_mrr":              fixed_mrr,
        "learned_mrr":            learned_mrr,
        "class_weighted_mrr":     class_mrr,
        "fixed_class_mrr":        fixed_class_mrr,
        "learned_class_mrr":      learned_class_mrr,
        "class_weighted_class_mrr": class_class_mrr,
        "fixed_hit_rate_at_5":    fixed_hr5,
        "fixed_hit_rate_at_10":   fixed_hr10,
        "learned_hit_rate_at_5":  learned_hr5,
        "learned_hit_rate_at_10": learned_hr10,
        "class_hit_rate_at_5":    class_hr5,
        "class_hit_rate_at_10":   class_hr10,
        "best_overall":           best_overall[0],
        "per_gene_fixed":         fixed_detail,
        "per_gene_learned":       learned_detail,
        "per_gene_class":         class_detail,
    }
    out_path = OUTPUTS_DIR / "model_evaluation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_path}")

    return output


if __name__ == "__main__":
    evaluate_both()

