"""Step 6: tune TabNet with Optuna on one flight-grouped split.

TabNet is far slower than gradient boosting, so each trial trains once on
folds 1-3, early-stops on fold 4 and is scored on fold 0.
"""

from __future__ import annotations

import optuna
import pandas as pd
from _common import base_parser, load_config, save_json
from sklearn.metrics import roc_auc_score

from rongowai_ddm.data.dataset import feature_matrix, load_samples, subsample
from rongowai_ddm.data.splits import SPLIT_TRAIN
from rongowai_ddm.features.registry import DDM_GROUPS
from rongowai_ddm.models.tabnet_model import TabNetModel, suggest_tabnet_params

DEFAULT_GROUPS = DDM_GROUPS + ("polarimetric",)


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--n-rows", type=int, default=300_000)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
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
    test, val = folds == 0, folds == 4
    train = ~test & ~val
    print(
        f"tuning on {int(train.sum()):,} train / {int(val.sum()):,} val / {int(test.sum()):,} test rows, {len(names)} features"
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_tabnet_params(trial)
        model = TabNetModel(
            params,
            device=args.device,
            max_epochs=args.max_epochs,
            patience=args.patience,
            seed=cfg.split.seed,
        )
        model.fit(X[train], y[train], X[val], y[val])
        auc = roc_auc_score(y[test], model.predict_proba(X[test])[:, 1])
        trial.set_user_attr("epochs", len(model.history.get("loss", [])))
        return float(auc)

    storage = f"sqlite:///{cfg.results_dir / 'optuna_tabnet.db'}"
    study = optuna.create_study(
        study_name="tabnet",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=cfg.split.seed),
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
    best = dict(study.best_params)
    best["n_a"] = best["n_d"]
    print(f"best held-out AUC {study.best_value:.4f} with {best}")
    study.trials_dataframe().to_csv(cfg.results_dir / "tabnet_tuning_trials.csv", index=False)
    save_json(
        {
            "best_params": best,
            "best_auc": study.best_value,
            "n_trials": len(study.trials),
            "n_rows": int(len(df)),
            "groups": list(DEFAULT_GROUPS),
        },
        cfg.results_dir / "tabnet_best_params.json",
    )


if __name__ == "__main__":
    main()
