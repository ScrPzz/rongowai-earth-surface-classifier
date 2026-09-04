"""Step 7: train the final models and evaluate them on every holdout.

Fold 0 of the train pool is the validation fold: it stops the training,
chooses the decision threshold and fits the calibrators. Everything else in
the train pool trains the models. The geographic, temporal and combined
holdouts are only ever scored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from _common import Timer, base_parser, load_config, load_json, save_json
from sklearn.metrics import roc_auc_score

from rongowai_ddm.data.dataset import feature_matrix, load_samples
from rongowai_ddm.data.splits import HOLDOUT_SPLITS, SPLIT_TRAIN
from rongowai_ddm.evaluation.metrics import (
    best_threshold,
    binary_metrics,
    coast_bins,
    metrics_by_group,
    snr_bins,
)
from rongowai_ddm.features.registry import DDM_GROUPS, FEATURE_GROUPS
from rongowai_ddm.models.calibration import CALIBRATORS
from rongowai_ddm.models.rule_baseline import coherence_state_score
from rongowai_ddm.models.tabnet_model import TabNetModel
from rongowai_ddm.models.xgb import fit_xgb

DEFAULT_GROUPS = DDM_GROUPS + ("polarimetric",)
VAL_FOLD = 0


def group_permutation_importance(model, X, y, names, groups, rng, n_rows=150_000) -> pd.DataFrame:
    idx = rng.choice(len(y), min(n_rows, len(y)), replace=False)
    Xs, ys = X[idx].copy(), y[idx]
    base = roc_auc_score(ys, model.predict_proba(Xs)[:, 1])
    rows = [{"group": "none (reference)", "auc": base, "auc_drop": 0.0}]
    for g in groups:
        cols = [names.index(n) for n in FEATURE_GROUPS[g]]
        Xp = Xs.copy()
        Xp[:, cols] = Xp[rng.permutation(len(Xp))][:, cols]
        auc = roc_auc_score(ys, model.predict_proba(Xp)[:, 1])
        rows.append({"group": g, "auc": auc, "auc_drop": base - auc})
    return pd.DataFrame(rows)


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-tabnet", action="store_true")
    parser.add_argument("--tabnet-epochs", type=int, default=60)
    args = parser.parse_args()
    cfg = load_config(args)
    rng = np.random.default_rng(cfg.split.seed)
    out = cfg.results_dir / "final"
    out.mkdir(parents=True, exist_ok=True)

    fl = pd.read_parquet(cfg.flight_table_path)
    df = load_samples(cfg.samples_dir, groups=DEFAULT_GROUPS, flight_table=fl)
    X, names = feature_matrix(df, DEFAULT_GROUPS)
    y = df["label"].to_numpy(dtype=int)
    split = df["split"].to_numpy()
    fold = df["fold_id"].to_numpy()
    tr = (split == SPLIT_TRAIN) & (fold != VAL_FOLD)
    va = (split == SPLIT_TRAIN) & (fold == VAL_FOLD)
    print(
        f"train {tr.sum():,}  val {va.sum():,}  "
        + "  ".join(f"{s} {int((split == s).sum()):,}" for s in HOLDOUT_SPLITS)
    )

    preds = df[
        [
            "flight_id",
            "flight_idx",
            "split",
            "fold_id",
            "epoch",
            "channel",
            "is_lhcp",
            "label",
            "surface_type",
            "ddm_snr",
            "sp_lat",
            "sp_lon",
            "dist_to_coast_km",
            "coherence_state",
        ]
    ].copy()
    preds["p_rule"] = coherence_state_score(df["coherence_state"].to_numpy())
    models: dict[str, object] = {}

    xgb_params = (
        load_json(cfg.results_dir / "xgb_best_params.json")["best_params"]
        if (cfg.results_dir / "xgb_best_params.json").exists()
        else {}
    )
    n_est = int(xgb_params.pop("n_estimators", 800))
    with Timer("xgboost"):
        xgb_model = fit_xgb(
            X[tr],
            y[tr],
            X[va],
            y[va],
            xgb_params,
            device=args.device,
            n_estimators=max(3000, 2 * n_est),
            early_stopping_rounds=100,
            seed=cfg.split.seed,
        )
    print(f"xgboost stopped at {xgb_model.best_iteration} trees")
    xgb_model.save_model(out / "xgboost.json")
    preds["p_xgb"] = xgb_model.predict_proba(X)[:, 1]
    models["xgb"] = xgb_model

    if not args.no_tabnet:
        tab_params = (
            load_json(cfg.results_dir / "tabnet_best_params.json")["best_params"]
            if (cfg.results_dir / "tabnet_best_params.json").exists()
            else None
        )
        with Timer("tabnet"):
            tab = TabNetModel(
                tab_params,
                device=args.device,
                max_epochs=args.tabnet_epochs,
                patience=10,
                seed=cfg.split.seed,
                verbose=0,
            )
            tab.fit(X[tr], y[tr], X[va], y[va])
        tab.save(out / "tabnet")
        preds["p_tabnet"] = tab.predict_proba(X)[:, 1]
        models["tabnet"] = tab

    report: dict = {
        "n_train": int(tr.sum()),
        "n_val": int(va.sum()),
        "features": names,
        "groups": list(DEFAULT_GROUPS),
        "xgb_params": xgb_params,
        "xgb_trees": int(xgb_model.best_iteration),
        "models": {},
    }
    eval_sets = {"val_fold": va, **{s: split == s for s in HOLDOUT_SPLITS}}
    for m in ["rule"] + list(models):
        p = preds[f"p_{m}"].to_numpy()
        thr = 0.5 if m == "rule" else best_threshold(y[va], p[va], "f1")
        entry = {"threshold": thr, "splits": {}}
        for s, mask in eval_sets.items():
            entry["splits"][s] = binary_metrics(y[mask], p[mask], thr)
        report["models"][m] = entry
        print(
            f"{m:8s} thr={thr:.3f} "
            + "  ".join(
                f"{s}: auc {entry['splits'][s]['roc_auc']:.4f} f1 {entry['splits'][s]['f1']:.3f}"
                for s in eval_sets
            )
        )

    # calibration fitted on the validation fold, evaluated on the holdouts
    cal_rows = []
    for m in models:
        p = preds[f"p_{m}"].to_numpy()
        for method, cls in {"none": None, **CALIBRATORS}.items():
            calibrator = cls().fit(p[va], y[va]) if cls else None
            for s in HOLDOUT_SPLITS:
                mask = split == s
                q = calibrator.transform(p[mask]) if calibrator else p[mask]
                met = binary_metrics(y[mask], q, report["models"][m]["threshold"])
                cal_rows.append(
                    {
                        "model": m,
                        "calibration": method,
                        "split": s,
                        "brier": met["brier"],
                        "log_loss": met["log_loss"],
                        "ece": met["ece"],
                        "roc_auc": met["roc_auc"],
                    }
                )
            if method == "beta":
                preds[f"p_{m}_cal"] = calibrator.transform(p)
    pd.DataFrame(cal_rows).to_csv(out / "calibration.csv", index=False)

    # breakdowns for the XGBoost model
    p = preds["p_xgb"].to_numpy()
    thr = report["models"]["xgb"]["threshold"]
    breakdown = []
    df["snr_bin"] = snr_bins(df["ddm_snr"].to_numpy())
    df["coast_bin"] = coast_bins(df["dist_to_coast_km"].to_numpy())
    df["polarization"] = np.where(df["is_lhcp"] == 1, "LHCP", "RHCP")
    for s in ("val_fold",) + HOLDOUT_SPLITS:
        mask = eval_sets[s]
        for col in ("polarization", "surface_type", "snr_bin", "coast_bin"):
            t = metrics_by_group(df[mask].reset_index(drop=True), p[mask], col, thr)
            t.insert(0, "split", s)
            t.insert(1, "by", col)
            t = t.rename(columns={col: "value"})
            breakdown.append(t)
    pd.concat(breakdown, ignore_index=True).to_csv(out / "breakdown_xgb.csv", index=False)

    # per-flight AUC on the geographic holdout (how uniform is the skill across flights?)
    geo = split == "geo_holdout"
    per_flight = (
        preds[geo]
        .groupby("flight_id")
        .apply(
            lambda g: pd.Series(
                {
                    "n": len(g),
                    "auc": roc_auc_score(g["label"], g["p_xgb"])
                    if g["label"].nunique() == 2
                    else np.nan,
                    "land_fraction": g["label"].mean(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    per_flight.to_csv(out / "per_flight_geo_xgb.csv", index=False)

    # importances
    gain = pd.DataFrame({"feature": names, "gain": xgb_model.feature_importances_}).sort_values(
        "gain", ascending=False
    )
    gain.to_csv(out / "importance_gain_xgb.csv", index=False)
    perm = group_permutation_importance(xgb_model, X[geo], y[geo], names, DEFAULT_GROUPS, rng)
    perm.to_csv(out / "importance_group_permutation_xgb.csv", index=False)
    print(perm.to_string(index=False))

    preds.to_parquet(out / "predictions.parquet", index=False)
    save_json(report, out / "metrics.json")


if __name__ == "__main__":
    main()
