"""Step 5: tune XGBoost with Optuna under flight-grouped folds.

Each trial trains on four folds and scores the fifth for three of the five
folds (median pruning stops bad trials after the first fold). The number of
trees is a hyper-parameter, so no early stopping is needed inside a trial.
"""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
from _common import base_parser, load_config, save_json
from sklearn.metrics import roc_auc_score

from rongowai_ddm.data.dataset import feature_matrix, load_samples, subsample
from rongowai_ddm.data.splits import SPLIT_TRAIN
from rongowai_ddm.features.registry import DDM_GROUPS
from rongowai_ddm.models.xgb import fit_with_fallback, make_xgb, suggest_xgb_params

DEFAULT_GROUPS = DDM_GROUPS + ("polarimetric",)


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--n-trials", type=int, default=60)
    parser.add_argument("--n-rows", type=int, default=400_000)
    parser.add_argument(
        "--eval-folds", type=int, default=3, help="how many of the folds are scored per trial"
    )
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
    X, names = feature_matrix(df, DEFAULT_GROUPS)
    y = df["label"].to_numpy(dtype=int)
    folds = df["fold_id"].to_numpy()
    fold_ids = np.sort(np.unique(folds))[: args.eval_folds]
    print(f"tuning on {len(df):,} rows, {len(names)} features, scoring folds {fold_ids.tolist()}")

    def objective(trial: optuna.Trial) -> float:
        params = suggest_xgb_params(trial)
        n_estimators = trial.suggest_int("n_estimators", 200, 1500, log=True)
        aucs = []
        for step, k in enumerate(fold_ids):
            te = folds == k
            model = make_xgb(
                params, device=args.device, n_estimators=n_estimators, seed=cfg.split.seed
            )
            fit_with_fallback(model, X[~te], y[~te])
            aucs.append(roc_auc_score(y[te], model.predict_proba(X[te])[:, 1]))
            trial.report(float(np.mean(aucs)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(aucs))

    storage = f"sqlite:///{cfg.results_dir / 'optuna_xgb.db'}"
    study = optuna.create_study(
        study_name="xgb",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=cfg.split.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0),
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
    best = dict(study.best_params)
    print(f"best CV AUC {study.best_value:.4f} with {best}")
    trials = study.trials_dataframe()
    trials.to_csv(cfg.results_dir / "xgb_tuning_trials.csv", index=False)
    save_json(
        {
            "best_params": best,
            "best_cv_auc": study.best_value,
            "n_trials": len(trials),
            "n_rows": int(len(df)),
            "groups": list(DEFAULT_GROUPS),
        },
        cfg.results_dir / "xgb_best_params.json",
    )


if __name__ == "__main__":
    main()
