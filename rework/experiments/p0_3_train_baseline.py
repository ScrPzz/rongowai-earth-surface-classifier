"""P0.3 baseline training: XGBoost on the 78-D features under honest CV.

Reads the per-split Parquet files produced by ``p0_3_etl.py``, then:

1. Trains XGBoost on the train_cv pool using ``StratifiedGroupKFold`` keyed
   on ``flight_id`` (the P0.1 fold harness).
2. Runs hyperparameter search (RandomizedSearchCV) with the production
   search space, or skips search if ``--smoke`` is passed.
3. Fits the final model on the full train_cv pool with early stopping.
4. Evaluates on each holdout split (val_geographic, val_temporal, val_both).
5. Saves model, scaler, hyperparams, and a JSON metrics report.

Usage::

    # quick smoke test: skip HP search, default params
    python rework/experiments/p0_3_train_baseline.py --smoke

    # full training (slow!): full HP search, 250 iterations
    python rework/experiments/p0_3_train_baseline.py --n-iter 250
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rework.data import STAT_FEATURE_NAMES  # noqa: E402
from rework.models import XGB_DEFAULT_PARAMS, XGBoostBaseline  # noqa: E402

DEFAULT_FEATURE_DIR = _PROJECT_ROOT / "rework" / "results" / "p0_3_features"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "rework" / "results"

ENC_FEATURE_NAMES = tuple(f"enc_{i:02d}" for i in range(20))
FEATURE_COLS = list(ENC_FEATURE_NAMES) + list(STAT_FEATURE_NAMES)

HOLDOUT_SPLITS = (
    "val_geographic_holdout",
    "val_temporal_holdout",
    "val_geographic_and_temporal_holdout",
)


def _load_split(feature_dir: Path, split: str) -> pd.DataFrame:
    path = feature_dir / f"{split}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=DEFAULT_FEATURE_DIR,
        help="Directory containing the per-split Parquet files from p0_3_etl.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the model, scaler, params, and metrics output.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Skip RandomizedSearchCV; train a single XGBoost with default "
            "hyperparameters on whatever data is present. Use for end-to-end "
            "smoke tests after a small ETL run."
        ),
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=250,
        help="RandomizedSearchCV iterations (ignored under --smoke).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of CV splits for HP search (ignored under --smoke).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="If set, subsample the train_cv pool to this many rows (debugging).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="XGBoost device ('cuda' or 'cpu'); default auto-detect.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[P0.3 train] Loading train_cv from {args.feature_dir}")
    train_df = _load_split(args.feature_dir, "train_cv")
    if train_df.empty:
        raise RuntimeError(
            f"No train_cv data found at {args.feature_dir / 'train_cv.parquet'}. "
            "Run p0_3_etl.py first."
        )
    print(f"[P0.3 train] train_cv: {len(train_df)} samples across "
          f"{train_df['flight_id'].nunique()} flights")

    if args.max_samples is not None and len(train_df) > args.max_samples:
        train_df = train_df.sample(
            n=args.max_samples, random_state=42
        ).reset_index(drop=True)
        print(f"[P0.3 train] Subsampled train_cv to {len(train_df)} rows.")

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["label"].values
    groups_train = train_df["flight_id"].values

    print(f"[P0.3 train] Class balance train_cv: "
          f"{int((y_train == 0).sum())} water / {int((y_train == 1).sum())} land")

    model = XGBoostBaseline(device=args.device, random_state=42)

    t0 = time.time()
    if args.smoke:
        print("[P0.3 train] SMOKE: skipping HP search, using default XGBoost params.")
        best_params = dict(XGB_DEFAULT_PARAMS)
        model.best_params = best_params
    else:
        print(
            f"[P0.3 train] RandomizedSearchCV (n_iter={args.n_iter}, "
            f"n_splits={args.n_splits}) — this is the slow step…"
        )
        best_params = model.hp_search(
            X_train,
            y_train,
            groups_train,
            n_iter=args.n_iter,
            n_splits=args.n_splits,
        )
        print(f"[P0.3 train] Best params: {json.dumps(best_params, indent=2, default=str)}")
    print(f"[P0.3 train] HP step elapsed: {time.time() - t0:.1f}s")

    print("[P0.3 train] Fitting final model with early stopping…")
    t0 = time.time()
    model.fit_final(X_train, y_train, params=best_params)
    print(f"[P0.3 train] Final fit elapsed: {time.time() - t0:.1f}s")

    # Evaluate
    all_metrics = []
    train_metrics = model.evaluate(X_train, y_train, split_name="train_cv (self)")
    all_metrics.append(train_metrics)
    print(f"\n[P0.3 train] === EVALUATION ===")
    print(f"  train_cv  (self)        : AUC={train_metrics.roc_auc:.4f}  F1={train_metrics.f1:.4f}")

    for split in HOLDOUT_SPLITS:
        df = _load_split(args.feature_dir, split)
        if df.empty:
            print(f"  {split:38s}: <no data>")
            continue
        X = df[FEATURE_COLS].values
        y = df["label"].values
        metrics = model.evaluate(X, y, split_name=split)
        all_metrics.append(metrics)
        print(
            f"  {split:38s}: AUC={metrics.roc_auc:.4f}  F1={metrics.f1:.4f}  "
            f"Brier={metrics.brier:.4f}  N={metrics.n_samples}"
        )

    # Persist
    model_path = args.output_dir / "p0_3_baseline_model.joblib"
    scaler_path = args.output_dir / "p0_3_baseline_scaler.joblib"
    params_path = args.output_dir / "p0_3_baseline_params.json"
    metrics_path = args.output_dir / "p0_3_baseline_metrics.json"

    model.save(model_path, scaler_path, params_path)
    with metrics_path.open("w") as f:
        json.dump([asdict(m) for m in all_metrics], f, indent=2)

    print(f"\n[P0.3 train] Wrote:")
    print(f"  {model_path}")
    print(f"  {scaler_path}")
    print(f"  {params_path}")
    print(f"  {metrics_path}")


if __name__ == "__main__":
    main()
