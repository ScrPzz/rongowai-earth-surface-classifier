"""Step 2: adversarial validation of the splits.

Two levels. At the flight level, metadata alone (dates, airports, file size) is
used to separate the train pool from each holdout: this documents how different
the holdouts are by construction and which metadata would leak if used as a
feature. At the sample level, the DDM features themselves are used, which tells
whether the feature distributions drift across geography and time, and whether
the CV folds are exchangeable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from _common import base_parser, load_config, save_json

from rongowai_ddm.data.dataset import feature_matrix, load_samples
from rongowai_ddm.data.splits import SPLIT_GEO, SPLIT_TIME, SPLIT_TRAIN
from rongowai_ddm.evaluation.adversarial import adversarial_validation
from rongowai_ddm.features.registry import DDM_GROUPS

META_FEATURES = ["month", "day_of_year", "hour", "origin_code", "dest_code", "file_size_log"]


def flight_features(fl: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=fl.index)
    out["month"] = fl["date"].dt.month
    out["day_of_year"] = fl["date"].dt.dayofyear
    out["hour"] = fl["date"].dt.hour
    codes = {c: i for i, c in enumerate(sorted(set(fl["origin"]) | set(fl["dest"])))}
    out["origin_code"] = fl["origin"].map(codes)
    out["dest_code"] = fl["dest"].map(codes)
    out["file_size_log"] = np.log10(fl["file_size"])
    return out


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument(
        "--max-rows", type=int, default=300_000, help="rows per pool at the sample level"
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = load_config(args)
    fl = pd.read_parquet(cfg.flight_table_path)
    report = {"flight_level": {}, "sample_level": {}}

    ff = flight_features(fl)
    train = fl["split"] == SPLIT_TRAIN
    for split in (SPLIT_GEO, SPLIT_TIME):
        other = fl["split"] == split
        feats = [
            f for f in META_FEATURES if not (split == SPLIT_TIME and f in ("month", "day_of_year"))
        ]
        res = adversarial_validation(
            ff.loc[train, feats].to_numpy(float),
            ff.loc[other, feats].to_numpy(float),
            feats,
            device=args.device,
        )
        res["features"] = feats
        report["flight_level"][f"train_vs_{split}"] = res
        print(
            f"flight level  train vs {split:13s}: AUC {res['mean_auc']:.3f} ({res['verdict']}); top: {res['top_features'][0]['feature']}"
        )

    groups = DDM_GROUPS + ("polarimetric",)
    df = load_samples(cfg.samples_dir, groups=groups, flight_table=fl, include_analysis=False)
    X, names = feature_matrix(df, groups)
    is_train = (df["split"] == SPLIT_TRAIN).to_numpy()
    rng = np.random.default_rng(cfg.split.seed)

    def pool(mask):
        idx = np.flatnonzero(mask)
        if len(idx) > args.max_rows:
            idx = rng.choice(idx, args.max_rows, replace=False)
        return X[idx]

    fold0 = is_train & (df["fold_id"].to_numpy() == 0)
    rest = is_train & (df["fold_id"].to_numpy() > 0)
    res = adversarial_validation(pool(rest), pool(fold0), names, device=args.device)
    report["sample_level"]["fold0_vs_other_folds"] = res
    print(f"sample level  fold 0 vs rest         : AUC {res['mean_auc']:.3f} ({res['verdict']})")
    for split in (SPLIT_GEO, SPLIT_TIME):
        other = (df["split"] == split).to_numpy()
        res = adversarial_validation(pool(is_train), pool(other), names, device=args.device)
        report["sample_level"][f"train_vs_{split}"] = res
        print(
            f"sample level  train vs {split:13s}: AUC {res['mean_auc']:.3f} ({res['verdict']}); top: {res['top_features'][0]['feature']}"
        )

    rows = []
    for level, d in report.items():
        for name, res in d.items():
            for rank, f in enumerate(res["top_features"], 1):
                rows.append({"level": level, "experiment": name, "rank": rank, **f})
    pd.DataFrame(rows).to_csv(cfg.results_dir / "adversarial_top_features.csv", index=False)
    save_json(report, cfg.results_dir / "adversarial_report.json")


if __name__ == "__main__":
    main()
