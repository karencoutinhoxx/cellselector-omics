import pandas as pd
import numpy as np
import lightgbm as lgb

FEATURE_COLUMNS = [
    "rna_score", "protein_score", "quality_score",
    "context_score", "pathway_activity_score", "mutation_impact_score",
    "rwr_score", "copy_number_score",
]
# Full feature parity with production as of tonight: copy_number_score
# (MYCN/ERBB2/FGFR1 only — 0.0 for every other gene) was the last missing
# source, flagged earlier as out of scope and now closed.


def build_training_data(
    validation_set: dict[str, list[str]],
    scores_cache: dict[str, pd.DataFrame],
    name_to_cvcl: dict[str, str]
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Build (X, y, group_sizes) for LightGBM's LambdaRank.

    X: feature matrix, all candidate cell lines for all
       genes stacked
    y: relevance labels (1 = known correct cell line for
       that gene, 0 = not)
    group_sizes: number of candidates per gene, in the same
       order as genes were stacked into X/y (LightGBM needs
       this to know query boundaries)
    """
    X_rows = []
    y_rows = []
    group_sizes = []

    for gene, known_names in validation_set.items():
        if gene not in scores_cache or scores_cache[gene] is None:
            continue
        # cellosaurus_id is the DataFrame's index (see
        # weights_learned._precompute_scores' .set_index), not a
        # column — reset_index puts it back as one before we need
        # to compare it against known_cvcls row-by-row.
        df = scores_cache[gene].reset_index().copy()

        # _build_name_to_cvcl()'s keys are lowercased; VALIDATION_SET
        # names are mixed-case ("HCC827") — without .lower() this
        # never matches, known_cvcls stays empty, and every gene gets
        # silently skipped (X_rows ends up empty, np.vstack crashes).
        known_cvcls = set()
        for name in known_names:
            cvcl = name_to_cvcl.get(name.lower())
            if cvcl:
                known_cvcls.add(cvcl)

        if not known_cvcls:
            continue

        df["label"] = df["cellosaurus_id"].apply(
            lambda x: 1 if x in known_cvcls else 0
        )

        # Fill missing features with 0
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0
        df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0.0)

        X_rows.append(df[FEATURE_COLUMNS].values)
        y_rows.append(df["label"].values)
        group_sizes.append(len(df))

    if not X_rows:
        raise ValueError(
            "build_training_data produced no rows — check that "
            "validation_set names resolve via name_to_cvcl and that "
            "scores_cache has entries for these genes."
        )

    X = np.vstack(X_rows)
    y = np.concatenate(y_rows)
    return X, y, group_sizes


def train_lambdamart(
    X: np.ndarray,
    y: np.ndarray,
    group_sizes: list[int]
) -> lgb.Booster:
    """Train a LambdaMART ranking model."""
    train_data = lgb.Dataset(X, label=y, group=group_sizes)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10],
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 5,
        "verbose": -1,
        "force_row_wise": True,
        # Reproducible across runs: no multithread FP-order variance, fixed seed.
        "deterministic": True,
        "num_threads": 1,
        "seed": 42,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=100
    )
    return model


def score_with_lambdamart(
    model: lgb.Booster,
    df: pd.DataFrame
) -> pd.Series:
    """Score candidate cell lines using a trained model."""
    df = df.copy()  # avoid mutating the caller's frame with filled columns
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    X = df[FEATURE_COLUMNS].fillna(0.0).values
    return pd.Series(model.predict(X), index=df.index)

