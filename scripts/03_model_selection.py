"""Step 3: screen model families under the same flight-grouped cross-validation.

Every family sees the same rows, the same engineered feature set and the same
folds. Slow families make the subsample necessary; the winner is tuned on more
data in the next steps. The rule based on the L1 coherence flag is reported on
the same rows as a floor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from _common import base_parser, load_config, save_json

from rongowai_ddm.data.dataset import feature_matrix, load_samples, subsample
from rongowai_ddm.data.splits import SPLIT_TRAIN
from rongowai_ddm.evaluation.cv import run_grouped_cv, summarize_folds
from rongowai_ddm.evaluation.metrics import binary_metrics
from rongowai_ddm.features.registry import DDM_GROUPS
from rongowai_ddm.models.rule_baseline import coherence_state_score
from rongowai_ddm.models.zoo import MODEL_NAMES, NEEDS_VALIDATION_SET, make_model

DEFAULT_GROUPS = DDM_GROUPS + ("polarimetric",)


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--n-rows", type=int, default=300_000, help="train rows used for screening")
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = load_config(args)

    fl = pd.read_parquet(cfg.flight_table_path)
    df = load_samples(cfg.samples_dir, groups=DEFAULT_GROUPS, splits=[SPLIT_TRAIN], flight_table=fl)
    df = subsample(df, args.n_rows, cfg.split.seed)
    X, names = feature_matrix(df, DEFAULT_GROUPS)
    y = df["label"].to_numpy(dtype=int)
    folds = df["fold_id"].to_numpy()
    print(
        f"screening on {len(df):,} rows, {len(names)} features, folds {np.unique(folds).tolist()}"
    )

    per_fold = []
    rule = coherence_state_score(df["coherence_state"].to_numpy())
    for k in np.unique(folds):
        m = binary_metrics(y[folds == k], rule[folds == k], 0.5)
        m.update(
            {"model": "rule_coherence_state", "fold": int(k), "n_train": 0, "fit_seconds": 0.0}
        )
        per_fold.append(m)
    for name in args.models:
        _, folds_df = run_grouped_cv(
            lambda name=name: make_model(name, device=args.device, seed=cfg.split.seed),
            X,
            y,
            folds,
            needs_validation=name in NEEDS_VALIDATION_SET,
            label=name,
        )
        per_fold.append(folds_df)
    per_fold_df = pd.concat(
        [pd.DataFrame([r]) if isinstance(r, dict) else r for r in per_fold], ignore_index=True
    )
    per_fold_df.to_csv(cfg.results_dir / "model_selection_folds.csv", index=False)
    summary = summarize_folds(per_fold_df).sort_values("roc_auc_mean", ascending=False)
    summary.to_csv(cfg.results_dir / "model_selection.csv", index=False)
    print(
        summary[
            [
                "model",
                "roc_auc_mean",
                "roc_auc_std",
                "avg_precision_mean",
                "f1_mean",
                "fit_seconds_mean",
            ]
        ].to_string(index=False)
    )
    save_json(
        {
            "n_rows": int(len(df)),
            "n_features": len(names),
            "groups": list(DEFAULT_GROUPS),
            "models": args.models,
        },
        cfg.results_dir / "model_selection_meta.json",
    )


if __name__ == "__main__":
    main()
