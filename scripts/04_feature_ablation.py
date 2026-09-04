"""Step 4: which feature groups matter, measured with one fixed model.

XGBoost with fixed hyper-parameters is trained on every feature set under the
flight-grouped folds (in-distribution score) and once on the whole train pool
to score the geographic and temporal holdouts (out-of-distribution scores).
A feature set is a list of groups plus, optionally, single features to exclude.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from _common import base_parser, load_config, save_json

from rongowai_ddm.data.dataset import feature_matrix, load_samples, subsample
from rongowai_ddm.data.splits import SPLIT_GEO, SPLIT_TIME, SPLIT_TRAIN
from rongowai_ddm.evaluation.cv import run_grouped_cv
from rongowai_ddm.evaluation.metrics import binary_metrics
from rongowai_ddm.features.registry import DDM_GROUPS, FEATURE_GROUPS
from rongowai_ddm.models.xgb import make_xgb

DEFAULT = list(DDM_GROUPS) + ["polarimetric"]
# Features that encode where the peak sits along delay (a geometry effect flagged by the adversarial validation).
DELAY_POSITION_FEATURES = ["peak_delay", "com_delay", "peak_centroid_delay", "peak_rel_pos_delay", "peak_dist_to_center"] + [
    f"band{i}_{s}" for i in range(5) for s in ("sum", "max")
]
LATENT_KEYS = ["flight_idx", "epoch", "channel"]

FEATURE_SETS: dict[str, dict] = {
    "raw pixels": {"groups": ["raw"]},
    "global": {"groups": ["global"]},
    "peak": {"groups": ["peak"]},
    "coherence": {"groups": ["coherence"]},
    "power": {"groups": ["power"]},
    "polarimetric": {"groups": ["polarimetric"]},
    "quadrant": {"groups": ["quadrant"]},
    "meta": {"groups": ["meta"]},
    "global+peak": {"groups": ["global", "peak"]},
    "global+peak+coherence": {"groups": ["global", "peak", "coherence"]},
    "DDM set (global+peak+coherence+power)": {"groups": list(DDM_GROUPS)},
    "default (DDM set + polarimetric)": {"groups": DEFAULT},
    "default + quadrant": {"groups": DEFAULT + ["quadrant"]},
    "default + meta": {"groups": DEFAULT + ["meta"]},
    "default + raw pixels": {"groups": DEFAULT + ["raw"]},
    "default - global": {"groups": [g for g in DEFAULT if g != "global"]},
    "default - peak": {"groups": [g for g in DEFAULT if g != "peak"]},
    "default - coherence": {"groups": [g for g in DEFAULT if g != "coherence"]},
    "default - power": {"groups": [g for g in DEFAULT if g != "power"]},
    "default - polarimetric": {"groups": [g for g in DEFAULT if g != "polarimetric"]},
    "default - delay position": {"groups": DEFAULT, "exclude": DELAY_POSITION_FEATURES},
    "latent": {"groups": ["latent"]},
    "default + latent": {"groups": DEFAULT + ["latent"]},
}


def build_matrix(df: pd.DataFrame, spec: dict) -> tuple[np.ndarray, list[str]]:
    X, names = feature_matrix(df, spec["groups"])
    exclude = set(spec.get("exclude", []))
    if exclude:
        keep = [i for i, n in enumerate(names) if n not in exclude]
        X, names = X[:, keep], [names[i] for i in keep]
    return X, names


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--n-rows", type=int, default=500_000, help="train rows for the CV part")
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sets", nargs="+", default=None, help="subset of feature-set names")
    parser.add_argument("--latent", type=Path, default=None, help="latent parquet from 09_train_autoencoder.py")
    parser.add_argument("--append", action="store_true", help="append to results/feature_ablation.csv instead of overwriting")
    args = parser.parse_args()
    cfg = load_config(args)
    sets = {k: v for k, v in FEATURE_SETS.items() if (args.sets is None or k in args.sets)}
    if args.latent is None:
        sets = {k: v for k, v in sets.items() if "latent" not in v["groups"]}

    fl = pd.read_parquet(cfg.flight_table_path)
    all_groups = sorted({g for v in sets.values() for g in v["groups"]} - {"latent"})
    df = load_samples(cfg.samples_dir, groups=all_groups, flight_table=fl)
    if args.latent is not None:
        lat = pd.read_parquet(args.latent)
        latent_cols = [c for c in lat.columns if c.startswith("latent_")]
        FEATURE_GROUPS["latent"] = tuple(latent_cols)
        df = df.merge(lat, on=LATENT_KEYS, how="left")
        assert df[latent_cols].notna().all().all(), "latent rows do not match the samples"
    train = df[df["split"] == SPLIT_TRAIN].reset_index(drop=True)
    geo = df[df["split"] == SPLIT_GEO].reset_index(drop=True)
    tim = df[df["split"] == SPLIT_TIME].reset_index(drop=True)
    train_cv = subsample(train, args.n_rows, cfg.split.seed)
    print(f"train {len(train):,} rows (CV on {len(train_cv):,}), geo holdout {len(geo):,}, time holdout {len(tim):,}")

    out_path = cfg.results_dir / "feature_ablation.csv"
    rows = pd.read_csv(out_path).to_dict("records") if (args.append and out_path.exists()) else []
    for name, spec in sets.items():
        t0 = time.time()
        Xcv, feats = build_matrix(train_cv, spec)
        ycv = train_cv["label"].to_numpy(dtype=int)
        _, folds_df = run_grouped_cv(
            lambda: make_xgb(device=args.device, n_estimators=args.n_estimators, seed=cfg.split.seed),
            Xcv,
            ycv,
            train_cv["fold_id"].to_numpy(),
            label=name,
        )
        Xtr, _ = build_matrix(train, spec)
        model = make_xgb(device=args.device, n_estimators=args.n_estimators, seed=cfg.split.seed)
        model.fit(Xtr, train["label"].to_numpy(dtype=int))
        out = {
            "feature_set": name,
            "groups": "+".join(spec["groups"]) + (" - " + ",".join(spec["exclude"]) if spec.get("exclude") else ""),
            "n_features": len(feats),
            "cv_auc": folds_df["roc_auc"].mean(),
            "cv_auc_std": folds_df["roc_auc"].std(),
            "cv_ap": folds_df["avg_precision"].mean(),
            "cv_f1": folds_df["f1"].mean(),
        }
        for split_name, part in (("geo", geo), ("time", tim)):
            Xh, _ = build_matrix(part, spec)
            m = binary_metrics(part["label"].to_numpy(dtype=int), model.predict_proba(Xh)[:, 1], 0.5)
            out[f"{split_name}_auc"] = m["roc_auc"]
            out[f"{split_name}_ap"] = m["avg_precision"]
            out[f"{split_name}_f1"] = m["f1"]
        out["seconds"] = time.time() - t0
        rows = [r for r in rows if r["feature_set"] != name] + [out]
        print(
            f"{name:40s} n={len(feats):3d}  cv {out['cv_auc']:.4f}±{out['cv_auc_std']:.4f}  geo {out['geo_auc']:.4f}  time {out['time_auc']:.4f}",
            flush=True,
        )
        pd.DataFrame(rows).to_csv(out_path, index=False)
    save_json({"n_rows_cv": int(len(train_cv)), "n_estimators": args.n_estimators, "sets": sets}, cfg.results_dir / "feature_ablation_meta.json")


if __name__ == "__main__":
    main()
