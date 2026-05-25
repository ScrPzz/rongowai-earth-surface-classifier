"""Flight-level fold assignment with combined geographic + temporal holdout.

After the geographic and temporal holdouts are stripped from the dataset,
remaining flights are partitioned into ``n_splits`` folds via
:class:`sklearn.model_selection.StratifiedGroupKFold` keyed on ``flight_id``
and stratified by the per-flight dominant class label.

P0.1 produces the harness using a dummy per-flight label (random binary)
so the leak checks can run before any NetCDF I/O has happened. P0.3 will
swap the dummy label for the real dominant-surface-type label per flight
after the ETL pass over the labelled dataset.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Canonical labels for the 'split_type' column in fold_assignments.parquet.
SPLIT_TRAIN_CV = "train_cv"
SPLIT_VAL_GEOGRAPHIC = "val_geographic_holdout"
SPLIT_VAL_TEMPORAL = "val_temporal_holdout"
SPLIT_VAL_BOTH = "val_geographic_and_temporal_holdout"


def assign_holdouts(
    df: pd.DataFrame,
    temporal_holdout_months: int = 6,
    reference_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """Mark each flight as train_cv or one of three OOD holdouts.

    Geographic holdout: any flight touching the South Island.
    Temporal holdout: any flight whose date is within
    ``temporal_holdout_months`` of ``reference_date``. The reference date
    defaults to the most recent flight in the dataset, so the holdout is
    always the trailing-N-months window of the available data.

    A flight in BOTH holdouts is labelled :data:`SPLIT_VAL_BOTH`.
    """
    df = df.copy()
    if reference_date is None:
        reference_date = df["date"].max()
    cutoff = reference_date - timedelta(days=30 * temporal_holdout_months)

    is_geographic = df["touches_south_island"]
    is_temporal = df["date"] >= cutoff

    df["is_geographic_holdout"] = is_geographic
    df["is_temporal_holdout"] = is_temporal
    df["temporal_cutoff"] = cutoff

    conditions = [
        is_geographic & is_temporal,
        is_geographic & ~is_temporal,
        ~is_geographic & is_temporal,
    ]
    choices = [SPLIT_VAL_BOTH, SPLIT_VAL_GEOGRAPHIC, SPLIT_VAL_TEMPORAL]
    df["split_type"] = np.select(conditions, choices, default=SPLIT_TRAIN_CV)
    return df


def assign_train_cv_folds(
    df: pd.DataFrame,
    n_splits: int = 10,
    random_state: int = 42,
    flight_label_col: str = "flight_label_dummy",
) -> pd.DataFrame:
    """Within the train_cv pool, assign each flight to one of ``n_splits`` folds.

    Uses :class:`StratifiedGroupKFold` keyed on ``flight_id`` and stratified
    by ``flight_label_col``. Flights outside the train_cv pool keep
    ``fold_id = -1``.
    """
    df = df.copy()
    train_cv_mask = df["split_type"] == SPLIT_TRAIN_CV
    n_train = int(train_cv_mask.sum())
    if n_train == 0:
        raise RuntimeError("No flights remain in the train/cv pool after holdouts")

    if flight_label_col not in df.columns:
        rng = np.random.default_rng(random_state)
        df[flight_label_col] = rng.integers(0, 2, size=len(df))

    skf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    flight_ids = df.loc[train_cv_mask, "flight_id"].values
    labels = df.loc[train_cv_mask, flight_label_col].values
    x_dummy = np.zeros((n_train, 1))

    fold_assignments = np.full(n_train, -1, dtype=int)
    for fold_idx, (_, val_idx) in enumerate(
        skf.split(x_dummy, labels, groups=flight_ids)
    ):
        fold_assignments[val_idx] = fold_idx

    df["fold_id"] = -1
    df.loc[train_cv_mask, "fold_id"] = fold_assignments
    return df
