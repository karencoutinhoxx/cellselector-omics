from pathlib import Path
import hashlib
import json
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import MASTER_MERGED, OUTPUTS_DIR
from src.models.classical.ranker import rank
from src.models.classical.similarity import find_alternatives
from src.models.agentic.retriever import format_context, retrieve_evidence
from src.models.agentic.generator import (
    add_citations_to_justification,
    generate_comparison,
    generate_justification,
)


def _parse_justification(text: str) -> dict:
    """
    Parse the numbered justification into structured fields.

    Sections 1-5 are LLM-generated prose extracted as strings.
    Sections 6 (DATA SOURCES) and 7 (LITERATURE) are appended by
    add_citations_to_justification() and parsed into lists.
    """
    key_map = {
        "1": "recommendation",
        "2": "key_reason",
        "3": "evidence_summary",
        "4": "trade_offs",
        "5": "best_for",
    }
    sections: dict = {v: "" for v in key_map.values()}
    sections["data_citations"]      = []
    sections["literature_citations"] = []

    pattern = re.compile(r"^\s*(\d)\.\s+[A-Z][A-Z\s\-]+:\s*(.*)", re.MULTILINE)
    matches = list(pattern.finditer(text))

    raw: dict[str, str] = {}
    for idx, m in enumerate(matches):
        num        = m.group(1)
        first_line = m.group(2).strip()
        start = m.end()
        end   = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        continuation = text[start:end].strip()
        content = (first_line + (" " + continuation if continuation else "")).strip()
        if num in key_map:
            sections[key_map[num]] = content
        else:
            raw[num] = content

    # Section 6: DATA SOURCES → list of "[D1] ..." strings
    raw6 = raw.get("6", "")
    if raw6:
        markers = re.findall(r'\[D\d+\]', raw6)
        entries = re.split(r'\[D\d+\]', raw6)
        for marker, entry in zip(markers, entries[1:]):
            sections["data_citations"].append(f"{marker} {entry.strip()}")

    # Section 7: LITERATURE → list of "[P1] ..." strings
    raw7 = raw.get("7", "")
    if raw7:
        markers = re.findall(r'\[P\d+\]', raw7)
        entries = re.split(r'\[P\d+\]', raw7)
        for marker, entry in zip(markers, entries[1:]):
            sections["literature_citations"].append(f"{marker} {entry.strip()}")

    return sections


def _verification_notes(evidence: dict) -> list[str]:
    """
    Structured data-gap caveats, from the SAME data_coverage the LLM prompt
    gets (retriever section 6b) so the JSON's notes and the generated
    TRADE-OFFS text can't disagree. Attached to each result in run().
    """
    from src.models.classical.scorer import classify_gene

    notes = []
    gene = evidence.get("gene", "")
    has_mut_calls = bool(evidence.get("metadata", {}).get("has_mutations"))
    if classify_gene(gene) == "loss_of_function" and not evidence.get("mutation"):
        if has_mut_calls:
            notes.append(
                f"{gene} is a loss-of-function target; this line HAS somatic "
                f"variant calls and none damage {gene} — likely wild-type, a "
                f"control rather than a disease model"
            )
        else:
            notes.append(
                f"{gene} is a loss-of-function target, but this line has NO "
                f"somatic variant calls — its {gene} mutation status is UNKNOWN, "
                f"not confirmed wild-type"
            )
    if not has_mut_calls:
        notes.append(
            "No somatic variant calls for this cell line — mutation-status "
            "claims (wild-type or mutant) cannot be made from this data"
        )

    cov = evidence.get("data_coverage", {})
    missing = [src for src, c in cov.items() if not c["present"]]
    if missing:
        notes.append("Missing evidence sources for this gene/cell-line pair: "
                     + ", ".join(missing))
    if not cov.get("HPA RNA expression", {}).get("present") \
            and not cov.get("DepMap RNA expression", {}).get("present"):
        notes.append("No primary RNA expression data (HPA and DepMap both absent)")

    if evidence.get("scores", {}).get("geo_confirmation", 0) < 0:
        notes.append("GEO data contradicts primary RNA sources — treat with caution")
    if not evidence.get("literature"):
        notes.append("No PubMed literature found for this gene/cell-line pair")
    return notes


def run(
    gene: str,
    disease_filter: str | None = None,
    lineage_filter: str | None = None,
    top_n: int = 5,
    target_cellosaurus_id: str | None = None,
    exclude_genes: list[str] | None = None,
) -> dict:
    """
    Full agentic ranking pipeline:
      1. Classical rank → top_n results (or all if target_cellosaurus_id is set)
         exclude_genes penalties are applied inside rank() itself.
      2. Filter to target cell line if target_cellosaurus_id is set.
      3. For each: retrieve evidence, format context, generate LLM justification.
      4. Generate LLM comparative summary (skipped when targeting a single cell line).
      5. Save to outputs/agentic_results_{gene}_{hash}.json.

    Returns the full structured output dict.
    """
    print(f"[pipeline] Ranking {gene}"
          + (f" | disease={disease_filter}" if disease_filter else "")
          + (f" | lineage={lineage_filter}" if lineage_filter else "")
          + (f" | exclude={','.join(exclude_genes)}" if exclude_genes else "")
          + (f" | target={target_cellosaurus_id}" if target_cellosaurus_id else f" | top_n={top_n}"))

    # ── Step 1: Classical ranking ─────────────────────────────────────────────
    # When a specific cell line is requested, rank without a top_n cap so the
    # target is always present in the results before we filter down to it.
    rank_top_n = None if target_cellosaurus_id else top_n
    ranked = rank(gene, disease_filter=disease_filter,
                  lineage_filter=lineage_filter, top_n=rank_top_n,
                  exclude_genes=exclude_genes)

    if ranked is not None and target_cellosaurus_id:
        filtered = ranked[ranked["cellosaurus_id"] == target_cellosaurus_id]
        if len(filtered) == 0:
            print(f"[pipeline] target {target_cellosaurus_id} not found in ranked results")
        else:
            ranked = filtered

    if ranked is None or len(ranked) == 0:
        print(f"[pipeline] No results for gene: {gene}")
        return {"gene": gene, "results": [], "comparative_summary": ""}

    # ── Compute similarity-based alternatives (once, for all ranked lines) ────
    print("[pipeline] Computing similarity alternatives...")
    try:
        alternatives_map = find_alternatives(
            gene, ranked, MASTER_MERGED, top_k=3,
            disease_filter=disease_filter,
            lineage_filter=lineage_filter,
        )
    except Exception as exc:
        print(f"  [warning] similarity failed: {exc}")
        alternatives_map = {}

    # ── Steps 2a–c: Per-result evidence + LLM justification ──────────────────
    results: list[dict] = []
    evidence_list: list[dict] = []

    for rank_pos, row in ranked.iterrows():
        cvcl = row["cellosaurus_id"]
        name = row.get("official_name") or cvcl
        final_score = float(row.get("final_score") or 0)

        print(f"  [{rank_pos + 1}] {name} ({cvcl}) — score={final_score:.3f}")

        # 2a: Retrieve evidence
        evidence = retrieve_evidence(gene, cvcl, row)
        evidence_list.append(evidence)

        # 2b: Format context for LLM; append exclusion warnings when relevant
        context_str = format_context(gene, evidence)

        exclusion_info = {"excluded_genes": exclude_genes or [], "warnings": []}
        if exclude_genes:
            excl_lines = []
            for excl_gene in exclude_genes:
                score = float(row.get(f"excluded_{excl_gene}_score", 0) or 0)
                if score > 0.5:
                    msg = (
                        f"EXCLUSION WARNING: This cell line also expresses "
                        f"{excl_gene} (score={score:.2f}) which was requested "
                        f"to be excluded. This may confound experimental results."
                    )
                    excl_lines.append(msg)
                    exclusion_info["warnings"].append(
                        f"{excl_gene} expressed at score {score:.2f} — may confound results"
                    )
            if excl_lines:
                context_str += "\n\n" + "\n".join(excl_lines)

        # 2c: Generate LLM justification
        try:
            justification = generate_justification(gene, context_str, name)
        except ConnectionError as exc:
            print(f"  [warning] {exc}")
            justification = str(exc)
        except Exception as exc:
            print(f"  [warning] LLM error: {exc}")
            justification = f"LLM unavailable: {exc}"

        justification = add_citations_to_justification(
            justification,
            evidence.get("dataset_citations", []),
            evidence.get("literature", []),
        )

        results.append({
            "rank":                rank_pos + 1,
            "cellosaurus_id":      cvcl,
            "official_name":       name,
            "hpa_evidence":        row.get("hpa_evidence"),
            "depmap_evidence":     row.get("depmap_evidence"),
            "geo_evidence":        row.get("geo_evidence"),
            "protein_evidence":    row.get("protein_evidence"),
            "vs_next_rank":        row.get("vs_next_rank"),
            "quality_explanation": row.get("quality_explanation") or "",
            "context_explanation": row.get("context_explanation") or "",
            "scores": {
                "final_score":      final_score,
                "rna_score":        float(row.get("rna_score") or 0),
                "protein_score":    float(row.get("protein_score") or 0),
                "quality_score":    float(row.get("quality_score") or 0),
                "context_score":    float(row.get("context_score") or 0),
                "geo_confirmation": float(row.get("geo_confirmation") or 0),
                "pathway_activity_score": float(row.get("pathway_activity_score") or 0),
            },
            "evidence":       evidence,
            "justification":  justification,
            "verification_notes": _verification_notes(evidence),
            "data_coverage":  evidence.get("data_coverage", {}),
            "exclusion_info": exclusion_info,
            "alternatives":   alternatives_map.get(cvcl, []),
        })

    # ── Step 3: Comparative summary (skipped for single-target queries) ───────
    if target_cellosaurus_id:
        comparative_summary = ""
    else:
        print("[pipeline] Generating comparative summary...")
        try:
            comparative_summary = generate_comparison(gene, results, evidence_list)
        except ConnectionError as exc:
            comparative_summary = str(exc)
        except Exception as exc:
            comparative_summary = f"LLM unavailable: {exc}"

    # ── Step 4: Save output ───────────────────────────────────────────────────
    output = {
        "gene":               gene,
        "query": {
            "disease_filter":        disease_filter,
            "lineage_filter":        lineage_filter,
            "top_n":                 top_n,
            "target_cellosaurus_id": target_cellosaurus_id,
            "exclude_genes":         exclude_genes or [],
        },
        "results":            results,
        "comparative_summary": comparative_summary,
    }

    excl_key = "-".join(sorted(exclude_genes)) if exclude_genes else ""
    query_key = f"{gene}_{disease_filter}_{lineage_filter}_{target_cellosaurus_id}_{excl_key}"
    query_hash = hashlib.md5(query_key.encode()).hexdigest()[:8]
    out_path = OUTPUTS_DIR / f"agentic_results_{gene}_{query_hash}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"[pipeline] Saved → {out_path}")

    return output

