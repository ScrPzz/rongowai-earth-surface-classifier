"""P0.2 — Adversarial validation diagnostic on the flight index.

Runs four adversarial-validation experiments to characterise the train /
holdout shift introduced by the P0.1 group-fold strategy:

1. **Within-pool consistency** — predict "is this flight in fold 0?" using
   only train_cv flights. *Should* yield AUC ≈ 0.5; anything higher signals
   that the StratifiedGroupKFold is accidentally leaking structural patterns
   beyond the dummy stratification key.
2. **train_cv vs val_geographic_holdout** — characterises the geographic
   shift. AUC will be very high because we *designed* the holdout to be
   different (South Island only); the feature importances reveal which
   metadata variables most reliably distinguish a South-Island flight from
   a North-Island one, beyond the obvious airport-code leakage features
   (which are excluded from the feature set for this experiment).
3. **train_cv vs val_temporal_holdout** — analogously for the temporal
   shift. Date-derived features that directly encode the cutoff
   (``date_timestamp``, ``year``) are excluded so the classifier must rely
   on subtler signals (month, day-of-year, hour-of-day, file size).
4. **train_cv vs ALL holdouts combined** — the overall production-distribution
   shift, with both geographic and temporal direct leakage features excluded.

This is a metadata-only diagnostic. Once the ETL produces per-sample DDM
features in P0.3 / P1, the same :func:`run_adversarial_validation` helper
can be re-applied at the sample level to detect feature-space drift.

Usage::

    python rework/experiments/p0_2_adversarial_validation.py

Outputs (under ``rework/results/``):
    p0_2_adversarial_report.json    # full per-experiment report
    p0_2_feature_importance.csv     # flat table of top-K features per experiment
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rework.validation import run_adversarial_validation  # noqa: E402

INPUT_PATH = _PROJECT_ROOT / "rework" / "results" / "p0_1_flight_index.parquet"
OUTPUT_DIR = _PROJECT_ROOT / "rework" / "results"

# Full feature pool. Per-experiment feature subsets are derived from this list
# by excluding direct-leakage variables (see EXCLUDED_BY_EXPERIMENT below).
FEATURE_COLS_ALL = [
    "date_timestamp",
    "year",
    "month",
    "day_of_year",
    "hour_of_day",
    "origin",
    "dest",
    "origin_island",
    "dest_island",
    "touches_south_island",
    "file_size_log",
]

# Direct-leakage features per experiment. These columns trivially encode the
# adversarial target and would make the AUC uninformative if included.
EXCLUDED_BY_EXPERIMENT = {
    "within_pool_fold0_vs_rest": [],  # nothing should leak fold assignment
    "train_cv_vs_val_geographic": [
        "touches_south_island",
        "origin_island",
        "dest_island",
    ],
    "train_cv_vs_val_temporal": [
        "date_timestamp",
        "year",
    ],
    "train_cv_vs_all_holdouts": [
        "touches_south_island",
        "origin_island",
        "dest_island",
        "date_timestamp",
        "year",
    ],
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived columns used by the adversarial classifier."""
    df = df.copy()
    df["date_timestamp"] = df["date"].astype("int64") // 10**9  # unix seconds
    df["day_of_year"] = df["date"].dt.dayofyear
    df["hour_of_day"] = df["date"].dt.hour
    df["file_size_log"] = np.log1p(df["file_size"])
    return df


def _features_for(name: str) -> list[str]:
    excluded = set(EXCLUDED_BY_EXPERIMENT[name])
    return [c for c in FEATURE_COLS_ALL if c not in excluded]


def _print_result(name: str, result: dict) -> None:
    print(
        f"    AUC = {result['mean_auc']:.4f} ± {result['std_auc']:.4f}  "
        f"→ {result['verdict']}"
    )
    print(f"    Top features:")
    for fi in result["top_features"][:5]:
        print(f"      {fi['feature']:22s}  {fi['importance']:.4f}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"P0.1 flight index not found at {INPUT_PATH}; "
            "run rework/experiments/p0_1_build_harness.py first."
        )

    print(f"[P0.2] Loading flight index from {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    df = engineer_features(df)
    print(
        f"[P0.2] Loaded {len(df)} flights; "
        f"{len(FEATURE_COLS_ALL)} adversarial features engineered."
    )

    experiments: list[dict] = []

    # Experiment 1: within-pool consistency
    name = "within_pool_fold0_vs_rest"
    print(f"\n[P0.2] === Exp 1: {name} ===")
    train_cv = df[df["split_type"] == "train_cv"].copy()
    train_cv["adv_target"] = (train_cv["fold_id"] == 0).astype(int)
    print(
        f"    n_pos (fold 0) = {int(train_cv['adv_target'].sum())}, "
        f"n_neg (folds 1-9) = {int((1 - train_cv['adv_target']).sum())}"
    )
    result = run_adversarial_validation(
        train_cv, _features_for(name), target_col="adv_target"
    )
    result["name"] = name
    result["expected"] = "AUC ≈ 0.5 (no drift within train_cv)"
    result["features_used"] = _features_for(name)
    result["features_excluded"] = EXCLUDED_BY_EXPERIMENT[name]
    _print_result(name, result)
    experiments.append(result)

    # Experiment 2: train vs geographic
    name = "train_cv_vs_val_geographic"
    print(f"\n[P0.2] === Exp 2: {name} ===")
    subset = df[
        df["split_type"].isin(["train_cv", "val_geographic_holdout"])
    ].copy()
    subset["adv_target"] = (
        subset["split_type"] == "val_geographic_holdout"
    ).astype(int)
    print(
        f"    n_train = {int((subset['adv_target'] == 0).sum())}, "
        f"n_holdout = {int(subset['adv_target'].sum())}"
    )
    result = run_adversarial_validation(
        subset, _features_for(name), target_col="adv_target"
    )
    result["name"] = name
    result["expected"] = "AUC very high (designed: South Island vs rest)"
    result["features_used"] = _features_for(name)
    result["features_excluded"] = EXCLUDED_BY_EXPERIMENT[name]
    _print_result(name, result)
    experiments.append(result)

    # Experiment 3: train vs temporal
    name = "train_cv_vs_val_temporal"
    print(f"\n[P0.2] === Exp 3: {name} ===")
    subset = df[
        df["split_type"].isin(["train_cv", "val_temporal_holdout"])
    ].copy()
    subset["adv_target"] = (
        subset["split_type"] == "val_temporal_holdout"
    ).astype(int)
    print(
        f"    n_train = {int((subset['adv_target'] == 0).sum())}, "
        f"n_holdout = {int(subset['adv_target'].sum())}"
    )
    result = run_adversarial_validation(
        subset, _features_for(name), target_col="adv_target"
    )
    result["name"] = name
    result["expected"] = (
        "AUC high if month / day-of-year / hour features capture seasonal "
        "or diurnal drift the model would have to handle in production."
    )
    result["features_used"] = _features_for(name)
    result["features_excluded"] = EXCLUDED_BY_EXPERIMENT[name]
    _print_result(name, result)
    experiments.append(result)

    # Experiment 4: train vs all holdouts
    name = "train_cv_vs_all_holdouts"
    print(f"\n[P0.2] === Exp 4: {name} ===")
    subset = df.copy()
    subset["adv_target"] = (subset["split_type"] != "train_cv").astype(int)
    print(
        f"    n_train = {int((subset['adv_target'] == 0).sum())}, "
        f"n_holdout = {int(subset['adv_target'].sum())}"
    )
    result = run_adversarial_validation(
        subset, _features_for(name), target_col="adv_target"
    )
    result["name"] = name
    result["expected"] = (
        "Aggregate production-distribution shift signature, with direct "
        "geographic and temporal leakage variables excluded."
    )
    result["features_used"] = _features_for(name)
    result["features_excluded"] = EXCLUDED_BY_EXPERIMENT[name]
    _print_result(name, result)
    experiments.append(result)

    # Persist
    report_path = OUTPUT_DIR / "p0_2_adversarial_report.json"
    fi_path = OUTPUT_DIR / "p0_2_feature_importance.csv"

    with report_path.open("w") as f:
        json.dump({"experiments": experiments}, f, indent=2)

    fi_rows = []
    for exp in experiments:
        for rank, fi in enumerate(exp["top_features"], 1):
            fi_rows.append(
                {
                    "experiment": exp["name"],
                    "rank": rank,
                    "feature": fi["feature"],
                    "importance": fi["importance"],
                }
            )
    pd.DataFrame(fi_rows).to_csv(fi_path, index=False)

    print(f"\n[P0.2] Wrote:\n  {report_path}\n  {fi_path}")

    print("\n[P0.2] === SUMMARY ===")
    for exp in experiments:
        print(
            f"  {exp['name']:38s}: AUC={exp['mean_auc']:.4f} ± "
            f"{exp['std_auc']:.4f}  ({exp['verdict']})"
        )


if __name__ == "__main__":
    main()
