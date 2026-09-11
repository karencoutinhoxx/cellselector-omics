import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import COPY_NUMBER_FILE, SAMPLE_INFO

# ─────────────────────────────────────────────────────────────────────────────
# Copy-number amplification scoring — the PRIMARY signal for genes whose
# selection criterion is gene amplification (MYCN, ERBB2), not expression
# level or point mutation. Same rationale as mutation_scorer.py for
# loss-of-function genes: ranking an amplification-driven oncogene by RNA
# alone measures a downstream consequence, not the actual mechanism a
# researcher selects a cell line for.
#
# Source: DepMap 23Q4 OmicsCNGene.csv — gene-level log2 relative copy ratio
# (log2(copy_number / ploidy), so ~0 = diploid/no change). Deliberately 23Q4,
# NOT 24Q4 (matching FILES["depmap"] / OMICS_PROFILES, not
# 6_OmicsSomaticMutationsProfile.csv — see config.py comment on
# COPY_NUMBER_FILE for why release consistency matters here).
#
# ID resolution is ONE HOP, not the two-hop PR- -> ACH- -> CVCL_ chain used
# in mutation_scorer.py: OmicsCNGene.csv is keyed directly by ModelID (ACH-),
# not ProfileID (PR-) — confirmed by inspecting the file's own row index.
#     ACH- --(9_DepMap_sample_info.csv: DepMap_ID -> RRID)-->  CVCL_
# ─────────────────────────────────────────────────────────────────────────────

# Genes whose selection criterion is gene amplification (not expression
# level or point mutation) — currently a small, explicit set, not derived
# from GENE_CLASSES. Gated separately from gene_class (both MYCN and ERBB2
# classify as tissue_specific) so this stays opt-in per gene rather than
# applying broadly to every tissue_specific gene, most of which have no
# amplification mechanism at all.
#
# Single source of truth: ranker.py, weights_learned.py (_precompute_scores)
# and scripts/evaluation/cross_validated_evaluation.py all import this (and
# the weight/rescale helper below) from here, rather than each defining
# their own copy — the original bug this fixed was full_evaluation.py's and
# cross_validated_evaluation.py's separate scoring paths silently not
# knowing about copy_number at all.
AMPLIFICATION_DRIVEN_GENES = {"MYCN", "ERBB2"}

# Fixed constant, NOT fit via optimise_weights_by_class()'s SLSQP — same
# treatment as weights_learned._LOF_PRODUCTION_MUTATION_WEIGHT /
# _LOF_PRODUCTION_PATHWAY_WEIGHT: a deliberate domain choice verified by a
# manual 1D grid search (0.0-1.0, MYCN peaks in a plateau at 0.08-0.30 then
# WORSENS past it; ERBB2 climbs to a plateau at 0.92-1.00), not something to
# hand to SLSQP. Bounded/exploratory: verified against a 2-gene category
# (MYCN, ERBB2), not cross-validated at the scale the other class weights
# are. Revisit — and consider per-gene rather than one shared constant,
# given the two genes' grid shapes actively disagree above ~0.30 — if
# AMPLIFICATION_DRIVEN_GENES grows.
AMPLIFICATION_COPY_NUMBER_WEIGHT = 0.30


def apply_amplification_copy_number_weight(weights: dict) -> dict:
    """
    Rescale {rna, protein, quality, context} proportionally so they sum to
    (1 - AMPLIFICATION_COPY_NUMBER_WEIGHT), add "copy_number" at the fixed
    constant. Same rescale-by-(1-w) construction as
    weights_learned._apply_lof_mutation_weight — whatever `weights` came in
    with (learned tissue_specific dict, a LOO-CV fold's refit 4D baseline,
    or FIXED_WEIGHTS) is the base being rescaled, so this reflects "the
    existing tissue_specific weights, minus room for copy_number", not a
    fresh refit. `pathway` is dropped (was 0.0 in the learned tissue_specific
    weights already; not meaningful here).
    """
    cn_w = AMPLIFICATION_COPY_NUMBER_WEIGHT
    scale = 1.0 - cn_w
    out = {k: weights.get(k, 0.0) * scale for k in ("rna", "protein", "quality", "context")}
    out["copy_number"] = cn_w
    return out


_CN_CACHE: dict[str, pd.DataFrame] = {}
_CN_DF: pd.DataFrame | None = None
_ACH_TO_CVCL: dict[str, str] | None = None


def _load_ach_to_cvcl() -> dict[str, str]:
    """ModelID (ACH-) -> cellosaurus_id (CVCL_), via DepMap sample_info.csv."""
    global _ACH_TO_CVCL
    if _ACH_TO_CVCL is not None:
        return _ACH_TO_CVCL

    samp = pd.read_csv(SAMPLE_INFO, usecols=["DepMap_ID", "RRID"], low_memory=False)
    _ACH_TO_CVCL = {
        r.DepMap_ID: r.RRID for r in samp.itertuples() if pd.notna(r.RRID)
    }
    return _ACH_TO_CVCL


def _load_cn_df() -> pd.DataFrame:
    """
    One-time load of the full gene-level CN matrix, indexed by ModelID, with
    cellosaurus_id resolved. Kept wide (not melted) — genome-wide is ~24k
    gene columns, so a single gene's column is sliced out on demand in
    score_copy_number() rather than melting the whole matrix up front.
    """
    global _CN_DF
    if _CN_DF is not None:
        return _CN_DF

    if not COPY_NUMBER_FILE.exists():
        _CN_DF = pd.DataFrame()
        return _CN_DF

    df = pd.read_csv(COPY_NUMBER_FILE, low_memory=False)
    df = df.rename(columns={df.columns[0]: "ModelID"}).copy()
    df["cellosaurus_id"] = df["ModelID"].map(_load_ach_to_cvcl())
    df = df.dropna(subset=["cellosaurus_id"])
    _CN_DF = df
    return _CN_DF


def _find_gene_column(df: pd.DataFrame, gene: str) -> str | None:
    """Columns are 'SYMBOL (EntrezID)' — match on the symbol prefix."""
    prefix = f"{gene} ("
    for col in df.columns:
        if col == gene or col.startswith(prefix):
            return col
    return None


def _cn_score(log2_ratio: float) -> float:
    """
    No flat ceiling — a genuine tie-break signal always exists between a
    borderline 4-copy call and a 100+-copy amplicon (the original 1.0-cap
    at log2>=1.0 collapsed dozens of cell lines per gene to an identical
    score; see the verification that flagged this).

    log2 <  0.3            (near-diploid, ~0-1 extra copy):    0.0
    log2  0.3-1.0          (modest/borderline gain, ~2-4 cp):  linear 0.0-0.5
    log2  1.0-2.0          (clear amplification, ~4-8 cp):     linear 0.5-0.8
    log2 >= 2.0 (amplicon, 8+ copies): 0.8 + 0.2*(1 - exp(-(log2-2.0)/5))

    NOTE: not `0.8 + min(0.2, (log2-2.0)/10)` as originally specified — that
    formula hits its min() cap and goes EXACTLY flat for every log2 >= 4.0,
    reproducing the same tie (verified live: log2=4.88 and log2=7.21 both
    scored 1.0000 under it), which is the one thing this fix was for. The
    exponential asymptote below still approaches 1.0 for very high copy
    number but, being a true asymptote, never actually flattens at any
    finite log2 — e.g. log2=4.88 -> 0.888, log2=7.21 -> 0.929, still
    differentiated from each other, not just from the lower bands.
    """
    if log2_ratio >= 2.0:
        return 0.8 + 0.2 * (1 - np.exp(-(log2_ratio - 2.0) / 5))
    if log2_ratio >= 1.0:
        return 0.5 + (log2_ratio - 1.0) * 0.3
    if log2_ratio >= 0.3:
        return (log2_ratio - 0.3) / 0.7 * 0.5
    return 0.0


def _cn_detail(gene: str, log2_ratio: float, score: float) -> str:
    if score <= 0.0:
        return ""
    copies = round(2 * (2 ** log2_ratio))  # assume diploid (ploidy=2) baseline
    if log2_ratio >= 2.0:
        label = "high-level amplification"
    elif log2_ratio >= 1.0:
        label = "clear amplification"
    else:
        label = "modest/borderline gain"
    return f"{gene} log2CN={log2_ratio:.2f} ({label}, ~{copies} copies)"


def score_copy_number(gene: str) -> pd.DataFrame:
    """
    Compute an amplification score (0-1) per cell line for `gene`, from
    DepMap gene-level copy-number data (log2 relative copy ratio).

    A cell line with no CN call for this gene is simply absent from the
    result (not scored 0.0) — unlike mutation_scorer, where "no variant
    record" is itself informative (wild-type). Here, missing CN coverage
    is a data-gap, not a biological signal, so it's left for the caller
    (ranker.py) to handle via its existing missing-source machinery rather
    than silently asserting "not amplified".

    Returns: DataFrame [cellosaurus_id, copy_number_score, copy_number_detail]
    Cached per gene (same pattern as score_mutation_impact).
    """
    if gene in _CN_CACHE:
        return _CN_CACHE[gene]

    empty = pd.DataFrame({
        "cellosaurus_id":      pd.Series(dtype="object"),
        "copy_number_score":   pd.Series(dtype="float64"),
        "copy_number_detail":  pd.Series(dtype="object"),
    })

    df = _load_cn_df()
    if df.empty:
        _CN_CACHE[gene] = empty
        return empty

    col = _find_gene_column(df, gene)
    if col is None:
        _CN_CACHE[gene] = empty
        return empty

    sub = df[["cellosaurus_id", col]].dropna(subset=[col]).copy()
    sub = sub.rename(columns={col: "_log2"})
    sub["_log2"] = sub["_log2"].astype(float)

    # Multiple ModelIDs can map to the same cellosaurus_id (rare); keep the
    # highest log2 ratio (most amplified), same "most damaging wins"
    # convention as mutation_scorer.py.
    sub = sub.sort_values("_log2", ascending=False).drop_duplicates("cellosaurus_id")

    sub["copy_number_score"] = sub["_log2"].apply(_cn_score)
    sub["copy_number_detail"] = sub.apply(
        lambda r: _cn_detail(gene, r["_log2"], r["copy_number_score"]), axis=1
    )

    result = sub[["cellosaurus_id", "copy_number_score", "copy_number_detail"]].reset_index(drop=True)
    _CN_CACHE[gene] = result
    return result
