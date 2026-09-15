import pandas as pd
import numpy as np


def compute_rrf_score(
    df: pd.DataFrame,
    source_columns: list[str],
    k: int = 60
) -> pd.Series:
    """
    Compute Reciprocal Rank Fusion score across multiple
    evidence source columns already present in df.

    Args:
        df: DataFrame with one row per cell line, columns
            for each evidence source (e.g. rna_score,
            protein_score, quality_score, context_score)
        source_columns: which columns to fuse
        k: RRF constant (default 60, per Cormack et al. 2009)

    Returns:
        pd.Series of RRF scores, same index as df, higher
        = better
    """
    rrf_total = pd.Series(0.0, index=df.index)

    for col in source_columns:
        if col not in df.columns:
            continue
        # Rank descending (higher raw score = better = rank 1)
        # method='min' handles ties consistently
        ranks = df[col].rank(ascending=False, method='min')
        rrf_total += 1.0 / (k + ranks)

    return rrf_total


def rank_by_rrf(
    df: pd.DataFrame,
    source_columns: list[str],
    k: int = 60
) -> pd.DataFrame:
    """
    Return df sorted by RRF score, descending, with an
    'rrf_score' column added.
    """
    result = df.copy()
    result["rrf_score"] = compute_rrf_score(df, source_columns, k)
    return result.sort_values("rrf_score", ascending=False)

