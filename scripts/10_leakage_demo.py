"""Step 10: how much a sample-level split inflates the score.

The same rows, the same model, two ways of splitting: at random over DDMs
(what the first version of this work did) and by flight (what this
repository does). The gap is the memorisation that a random split rewards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from _common import base_parser, load_config, save_json
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from rongowai_ddm.data.dataset import feature_matrix, load_samples, subsample
from rongowai_ddm.data.splits import SPLIT_TRAIN
from rongowai_ddm.features.registry import DDM_GROUPS
from rongowai_ddm.models.xgb import fit_with_fallback, make_xgb

DEFAULT_GROUPS = DDM_GROUPS + ("polarimetric",)


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--n-rows", type=int, default=500_000)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = load_config(args)
    fl = pd.read_parquet(cfg.flight_table_path)
    df = load_samples(
        cfg.samples_dir,
        groups=DEFAULT_GROUPS,
        splits=[SPLIT_TRAIN],
        flight_table=fl,
        include_analysis=False,
    )
    df = subsample(df, args.n_rows, cfg.split.seed)
    X, _ = feature_matrix(df, DEFAULT_GROUPS)
    y = df["label"].to_numpy(dtype=int)
    folds = df["fold_id"].to_numpy()
    rows = []
    # (a) flight-grouped folds, as everywhere else in the repository
    for k in np.unique(folds):
        te = folds == k
        m = make_xgb(device=args.device, n_estimators=args.n_estimators, seed=cfg.split.seed)
        fit_with_fallback(m, X[~te], y[~te])
        p = m.predict_proba(X[te])[:, 1]
        rows.append(
            {
                "split": "by flight",
                "fold": int(k),
                "roc_auc": roc_auc_score(y[te], p),
                "avg_precision": average_precision_score(y[te], p),
            }
        )
    # (b) random row-level folds over the same rows
    skf = StratifiedKFold(n_splits=len(np.unique(folds)), shuffle=True, random_state=cfg.split.seed)
    for k, (tr, te) in enumerate(skf.split(X, y)):
        m = make_xgb(device=args.device, n_estimators=args.n_estimators, seed=cfg.split.seed)
        fit_with_fallback(m, X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        rows.append(
            {
                "split": "random rows",
                "fold": k,
                "roc_auc": roc_auc_score(y[te], p),
                "avg_precision": average_precision_score(y[te], p),
            }
        )
    res = pd.DataFrame(rows)
    res.to_csv(cfg.results_dir / "leakage_demo_folds.csv", index=False)
    summary = res.groupby("split")[["roc_auc", "avg_precision"]].agg(["mean", "std"])
    print(summary.round(4).to_string())
    save_json(
        {
            "n_rows": int(len(df)),
            "summary": {
                s: {
                    "roc_auc": float(g["roc_auc"].mean()),
                    "avg_precision": float(g["avg_precision"].mean()),
                }
                for s, g in res.groupby("split")
            },
        },
        cfg.results_dir / "leakage_demo.json",
    )


if __name__ == "__main__":
    main()
