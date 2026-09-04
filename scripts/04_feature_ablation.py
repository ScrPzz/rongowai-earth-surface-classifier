"""Step 4: which feature groups matter, measured with one fixed model.

XGBoost with fixed hyper-parameters is trained on every feature set under the
flight-grouped folds (in-distribution score) and once on the whole train pool
to score the geographic and temporal holdouts (out-of-distribution scores).
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from _common import base_parser, load_config, save_json

from rongowai_ddm.data.dataset import feature_matrix, load_samples, subsample
from rongowai_ddm.data.splits import SPLIT_GEO, SPLIT_TIME, SPLIT_TRAIN
from rongowai_ddm.evaluation.cv import run_grouped_cv
from rongowai_ddm.evaluation.metrics import binary_metrics
from rongowai_ddm.features.registry import DDM_GROUPS
from rongowai_ddm.models.xgb import make_xgb

DEFAULT = list(DDM_GROUPS) + ["polarimetric"]
FEATURE_SETS: dict[str, list[str]] = {
    "raw pixels": ["raw"],
    "global": ["global"],
    "peak": ["peak"],
    "coherence": ["coherence"],
    "power": ["power"],
    "polarimetric": ["polarimetric"],
    "quadrant": ["quadrant"],
    "meta": ["meta"],
    "global+peak": ["global", "peak"],
    "global+peak+coherence": ["global", "peak", "coherence"],
    "DDM set (global+peak+coherence+power)": list(DDM_GROUPS),
    "default (DDM set + polarimetric)": DEFAULT,
    "default + quadrant": DEFAULT + ["quadrant"],
    "default + meta": DEFAULT + ["meta"],
    "default + raw pixels": DEFAULT + ["raw"],
    "default - global": [g for g in DEFAULT if g != "global"],
    "default - peak": [g for g in DEFAULT if g != "peak"],
    "default - coherence": [g for g in DEFAULT if g != "coherence"],
    "default - power": [g for g in DEFAULT if g != "power"],
    "default - polarimetric": [g for g in DEFAULT if g != "polarimetric"],
}


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--n-rows", type=int, default=500_000, help="train rows for the CV part")
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sets", nargs="+", default=None, help="subset of feature-set names")
    parser.add_argument(
        "--latent",
        type=Path,
        default=None,
        help="parquet with latent columns aligned to the samples (adds a 'latent' set)",
    )
    args = parser.parse_args()
    cfg = load_config(args)
    sets = dict(FEATURE_SETS)
    if args.sets:
        sets = {k: v for k, v in sets.items() if k in args.sets}

    fl = pd.read_parquet(cfg.flight_table_path)
    all_groups = sorted({g for v in sets.values() for g in v})
    df = load_samples(cfg.samples_dir, groups=all_groups, flight_table=fl)
    train = df[df["split"] == SPLIT_TRAIN].reset_index(drop=True)
    geo = df[df["split"] == SPLIT_GEO].reset_index(drop=True)
    tim = df[df["split"] == SPLIT_TIME].reset_index(drop=True)
    train_cv = subsample(train, args.n_rows, cfg.split.seed)
    print(
        f"train {len(train):,} rows (CV on {len(train_cv):,}), geo holdout {len(geo):,}, time holdout {len(tim):,}"
    )

    rows = []
    for name, groups in sets.items():
        t0 = time.time()
        Xcv, feats = feature_matrix(train_cv, groups)
        ycv = train_cv["label"].to_numpy(dtype=int)
        _, folds_df = run_grouped_cv(
            lambda: make_xgb(
                device=args.device, n_estimators=args.n_estimators, seed=cfg.split.seed
            ),
            Xcv,
            ycv,
            train_cv["fold_id"].to_numpy(),
            label=name,
        )
        Xtr, _ = feature_matrix(train, groups)
        model = make_xgb(device=args.device, n_estimators=args.n_estimators, seed=cfg.split.seed)
        model.fit(Xtr, train["label"].to_numpy(dtype=int))
        out = {
            "feature_set": name,
            "groups": "+".join(groups),
            "n_features": len(feats),
            "cv_auc": folds_df["roc_auc"].mean(),
            "cv_auc_std": folds_df["roc_auc"].std(),
            "cv_ap": folds_df["avg_precision"].mean(),
            "cv_f1": folds_df["f1"].mean(),
        }
        for split_name, part in (("geo", geo), ("time", tim)):
            Xh, _ = feature_matrix(part, groups)
            m = binary_metrics(
                part["label"].to_numpy(dtype=int), model.predict_proba(Xh)[:, 1], 0.5
            )
            out[f"{split_name}_auc"] = m["roc_auc"]
            out[f"{split_name}_ap"] = m["avg_precision"]
            out[f"{split_name}_f1"] = m["f1"]
        out["seconds"] = time.time() - t0
        rows.append(out)
        print(
            f"{name:40s} n={len(feats):3d}  cv {out['cv_auc']:.4f}±{out['cv_auc_std']:.4f}  geo {out['geo_auc']:.4f}  time {out['time_auc']:.4f}",
            flush=True,
        )
        pd.DataFrame(rows).to_csv(cfg.results_dir / "feature_ablation.csv", index=False)
    save_json(
        {"n_rows_cv": int(len(train_cv)), "n_estimators": args.n_estimators, "sets": sets},
        cfg.results_dir / "feature_ablation_meta.json",
    )


if __name__ == "__main__":
    main()
