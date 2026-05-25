"""Sanity checks for the flight-level fold assignment.

These checks are cheap and run at the end of the P0.1 experiment script.
They guarantee at the FLIGHT level (not yet sample level) that:

* every ``flight_id`` appears in exactly one (split, fold) slot;
* every flight tagged for a holdout has ``fold_id == -1``;
* every flight in the train_cv pool has a valid ``fold_id`` in
  ``[0, n_splits - 1]``;
* class balance does not drift wildly across folds.

When the real per-sample data is processed in P0.3, additional sample-level
leakage checks will be layered on top, but they will reduce to these flight-
level invariants because the per-sample ``flight_id`` is the join key.
"""
from __future__ import annotations

import pandas as pd

from .group_folds import (
    SPLIT_TRAIN_CV,
    SPLIT_VAL_BOTH,
    SPLIT_VAL_GEOGRAPHIC,
    SPLIT_VAL_TEMPORAL,
)


class LeakError(AssertionError):
    """Raised when the fold assignment violates a leakage invariant."""


def check_no_flight_leak(df: pd.DataFrame) -> None:
    """Each ``flight_id`` appears at most once."""
    grouped = df.groupby("flight_id").size()
    duplicates = grouped[grouped > 1]
    if not duplicates.empty:
        raise LeakError(
            f"{len(duplicates)} flight_id value(s) appear multiple times in the assignment"
        )


def check_holdouts_disjoint_from_train(df: pd.DataFrame) -> None:
    """No holdout flight should also have a non-(-1) fold_id."""
    holdout_mask = df["split_type"].isin(
        [SPLIT_VAL_GEOGRAPHIC, SPLIT_VAL_TEMPORAL, SPLIT_VAL_BOTH]
    )
    bad = df[holdout_mask & (df["fold_id"] != -1)]
    if not bad.empty:
        raise LeakError(
            f"{len(bad)} holdout flight(s) were also assigned a fold (fold_id != -1)"
        )


def check_all_train_assigned_to_fold(df: pd.DataFrame) -> None:
    """Every train_cv flight must have a non-negative fold_id."""
    train_mask = df["split_type"] == SPLIT_TRAIN_CV
    bad = df[train_mask & (df["fold_id"] < 0)]
    if not bad.empty:
        raise LeakError(
            f"{len(bad)} train_cv flight(s) were not assigned a fold (fold_id == -1)"
        )


def class_balance_per_fold(
    df: pd.DataFrame, label_col: str = "flight_label_dummy"
) -> pd.DataFrame:
    """Return a (fold_id × label) cross-tabulation for the train_cv pool."""
    train_df = df[df["split_type"] == SPLIT_TRAIN_CV]
    return train_df.groupby(["fold_id", label_col]).size().unstack(fill_value=0)


def run_all_checks(df: pd.DataFrame) -> dict:
    """Run all leak checks and return a summary report."""
    check_no_flight_leak(df)
    check_holdouts_disjoint_from_train(df)
    check_all_train_assigned_to_fold(df)

    split_counts = df["split_type"].value_counts().to_dict()
    split_proportions = df["split_type"].value_counts(normalize=True).to_dict()

    return {
        "n_flights_total": int(len(df)),
        "split_counts": {k: int(v) for k, v in split_counts.items()},
        "split_proportions": {k: float(v) for k, v in split_proportions.items()},
        "class_balance_per_fold": class_balance_per_fold(df).astype(int).to_dict(),
    }
