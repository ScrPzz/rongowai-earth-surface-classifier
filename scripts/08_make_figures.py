"""Step 8: regenerate every figure and table of the README from saved artefacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as pads
from _common import base_parser, load_config, load_json

from rongowai_ddm.data.labels import SURFACE_TYPE_NAMES
from rongowai_ddm.data.splits import HOLDOUT_SPLITS
from rongowai_ddm.evaluation import plots
from rongowai_ddm.features.layout import normalize_sum
from rongowai_ddm.features.peak_region import grow_peak_region
from rongowai_ddm.features.registry import RAW_FEATURE_NAMES

KEY_FEATURES = [
    "phpr_db",
    "count_snr_db",
    "pol_peak_ratio_db",
    "delay_width_hm",
    "vn_entropy",
    "peak_energy_fraction",
    "wf_rmsd_ambiguity",
    "pol_shape_corr",
]


def load_rows_with_pixels(samples_dir: Path, flight_idx: int, epochs: list[int]) -> pd.DataFrame:
    ds = pads.dataset(str(samples_dir), format="parquet")
    flt = (pads.field("flight_idx") == flight_idx) & pads.field("epoch").isin(epochs)
    return ds.to_table(filter=flt).to_pandas()


def fig_gallery(cfg, preds: pd.DataFrame, fig_dir: Path) -> None:
    """Eight DDMs of one geographic-holdout flight: water/land, both polarizations, high/low SNR."""
    geo = preds[preds["split"] == "geo_holdout"]
    counts = geo.groupby("flight_id")["label"].agg(["mean", "size"])
    fid = counts[(counts["mean"] > 0.3) & (counts["mean"] < 0.7)].sort_values("size").index[-1]
    rows = geo[geo["flight_id"] == fid]
    picks = []
    for label in (0, 1):
        for pol in (1, 0):
            sub = (
                rows[(rows["label"] == label) & (rows["is_lhcp"] == pol)]
                .dropna(subset=["ddm_snr"])
                .sort_values("ddm_snr")
            )
            if len(sub) >= 2:
                picks += [sub.iloc[-1], sub.iloc[len(sub) // 4]]
    epochs = sorted({int(r["epoch"]) for r in picks})
    px = load_rows_with_pixels(cfg.samples_dir, int(rows["flight_idx"].iloc[0]), epochs)
    ddms, titles = [], []
    for r in picks:
        m = px[(px["epoch"] == r["epoch"]) & (px["channel"] == r["channel"])]
        if m.empty:
            continue
        ddms.append(m[list(RAW_FEATURE_NAMES)].to_numpy(dtype=float)[0].reshape(40, 5))
        titles.append(
            f"{'water' if r['label'] == 0 else 'land'} {'LHCP' if r['is_lhcp'] else 'RHCP'}\nSNR {r['ddm_snr']:.1f} dB  p={r['p_xgb']:.2f}"
        )
    plots.plot_ddm_gallery(np.stack(ddms), titles, fig_dir / "ddm_gallery.png", ncols=4)
    x = normalize_sum(np.stack(ddms[:2]))
    mask = grow_peak_region(x, cfg.features.peak_region_pixels)
    for k, name in enumerate(("water", "land")):
        j = 0 if k == 0 else min(2, len(ddms) - 1)
        plots.plot_peak_region(
            ddms[j],
            grow_peak_region(normalize_sum(ddms[j][None]), cfg.features.peak_region_pixels)[0],
            fig_dir / f"peak_region_{name}.png",
            title=titles[j].split("\n")[0],
        )
    del mask


def fig_feature_distributions(cfg, fig_dir: Path) -> None:
    ds = pads.dataset(str(cfg.samples_dir), format="parquet")
    cols = ["label", "is_lhcp", "split"] + KEY_FEATURES
    t = ds.to_table(columns=cols, filter=pads.field("split") == "train").to_pandas()
    t = t.sample(min(300_000, len(t)), random_state=0)
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.2))
    for ax, f in zip(axes.ravel(), KEY_FEATURES, strict=False):
        for label, name in ((0, "water"), (1, "land")):
            v = t.loc[(t["label"] == label) & (t["is_lhcp"] == 1), f].to_numpy()
            lo, hi = np.nanpercentile(v, [0.5, 99.5])
            ax.hist(
                np.clip(v, lo, hi),
                bins=60,
                range=(lo, hi),
                density=True,
                alpha=0.55,
                color=plots.COLORS[name],
                label=name,
            )
        ax.set_title(f)
        ax.set_yticks([])
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Feature distributions on the train pool (LHCP DDMs)")
    fig.tight_layout()
    plots._save(fig, fig_dir / "feature_distributions.png")


def fig_model_selection(cfg, fig_dir: Path) -> None:
    p = cfg.results_dir / "model_selection.csv"
    if not p.exists():
        return
    df = pd.read_csv(p).sort_values("roc_auc_mean", ascending=False)
    plots.plot_bars(
        df,
        "model",
        ["roc_auc_mean", "avg_precision_mean"],
        fig_dir / "model_selection.png",
        ylabel="score (5-fold, grouped by flight)",
        title="Model screening",
        ylim=(0.5, 1.0),
        err=["roc_auc_std", "avg_precision_std"],
    )


def fig_ablation(cfg, fig_dir: Path) -> None:
    p = cfg.results_dir / "feature_ablation.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    plots.plot_bars(
        df,
        "feature_set",
        ["cv_auc", "geo_auc", "time_auc"],
        fig_dir / "feature_ablation.png",
        ylabel="ROC AUC",
        title="Feature-set ablation (XGBoost, fixed hyper-parameters)",
        ylim=(0.5, 1.0),
    )


def fig_final(cfg, preds: pd.DataFrame, fig_dir: Path) -> None:
    final = cfg.results_dir / "final"
    y = preds["label"].to_numpy()
    for s in HOLDOUT_SPLITS:
        m = (preds["split"] == s).to_numpy()
        curves = {"XGBoost": (y[m], preds.loc[m, "p_xgb"].to_numpy())}
        if "p_tabnet" in preds:
            curves["TabNet"] = (y[m], preds.loc[m, "p_tabnet"].to_numpy())
        curves["L1 coherence flag"] = (y[m], preds.loc[m, "p_rule"].to_numpy())
        plots.plot_roc_pr(curves, fig_dir / f"roc_pr_{s}.png", title=s.replace("_", " "))
    m = (preds["split"] == "geo_holdout").to_numpy()
    series = {"XGBoost raw": (y[m], preds.loc[m, "p_xgb"].to_numpy())}
    if "p_xgb_cal" in preds:
        series["XGBoost beta-calibrated"] = (y[m], preds.loc[m, "p_xgb_cal"].to_numpy())
    plots.plot_reliability(series, fig_dir / "reliability_geo_holdout.png")
    gain = pd.read_csv(final / "importance_gain_xgb.csv")
    plots.plot_feature_importance(
        gain["feature"].tolist(),
        gain["gain"].to_numpy(),
        fig_dir / "importance_gain.png",
        top=25,
        title="XGBoost gain importance",
    )
    perm = pd.read_csv(final / "importance_group_permutation_xgb.csv")
    perm = perm[perm["group"] != "none (reference)"]
    plots.plot_bars(
        perm,
        "group",
        ["auc_drop"],
        fig_dir / "importance_group_permutation.png",
        ylabel="AUC drop when the group is shuffled",
        title="Group permutation importance (geographic holdout)",
    )
    br = pd.read_csv(final / "breakdown_xgb.csv")
    for by, fname in (
        ("snr_bin", "by_snr"),
        ("polarization", "by_polarization"),
        ("surface_type", "by_surface_type"),
        ("coast_bin", "by_coast_distance"),
    ):
        t = br[(br["by"] == by) & (br["split"].isin(["geo_holdout", "time_holdout"]))].copy()
        if t.empty:
            continue
        if by == "surface_type":
            t["value"] = t["value"].map(lambda v: SURFACE_TYPE_NAMES.get(int(v), str(v)))
        t["label"] = t["value"].astype(str) + "\n" + t["split"].str.replace("_holdout", "")
        plots.plot_bars(
            t,
            "label",
            ["roc_auc", "balanced_accuracy"],
            fig_dir / f"metrics_{fname}.png",
            ylabel="score",
            title=f"XGBoost by {by.replace('_', ' ')}",
            ylim=(0.4, 1.0),
        )
    geo = preds[m].sample(min(400_000, int(m.sum())), random_state=0)
    plots.plot_map(
        geo["sp_lat"],
        geo["sp_lon"],
        geo["p_xgb"],
        fig_dir / "map_geo_holdout_probability.png",
        title="Geographic holdout: predicted probability of land",
        label="P(land)",
    )
    err = (geo["p_xgb"] >= 0.5).astype(int) != geo["label"]
    plots.plot_map(
        geo.loc[err, "sp_lat"],
        geo.loc[err, "sp_lon"],
        geo.loc[err, "label"],
        fig_dir / "map_geo_holdout_errors.png",
        title="Geographic holdout: misclassified DDMs (colour = true class)",
        cmap="coolwarm",
        label="true label (0 water, 1 land)",
    )
    pf = pd.read_csv(final / "per_flight_geo_xgb.csv").dropna()
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.hist(pf["auc"], bins=40, color=plots.COLORS["accent"])
    ax.set(
        xlabel="per-flight ROC AUC (geographic holdout)",
        ylabel="flights",
        title=f"median {pf['auc'].median():.3f}, 10th percentile {pf['auc'].quantile(0.1):.3f}",
    )
    plots._save(fig, fig_dir / "per_flight_auc_geo.png")


def fig_tuning(cfg, fig_dir: Path) -> None:
    for name in ("xgb", "tabnet"):
        p = cfg.results_dir / f"{name}_tuning_trials.csv"
        if not p.exists():
            continue
        t = pd.read_csv(p)
        t = t[t["state"] == "COMPLETE"].reset_index(drop=True)
        t["best_so_far"] = t["value"].cummax()
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        ax.scatter(t["number"], t["value"], s=10, color=plots.COLORS["grey"], label="trial")
        ax.plot(t["number"], t["best_so_far"], color=plots.COLORS["accent"], label="best so far")
        ax.set(xlabel="trial", ylabel="held-out AUC", title=f"Optuna search: {name}")
        ax.legend(fontsize=7)
        plots._save(fig, fig_dir / f"tuning_{name}.png")


def write_tables(cfg) -> None:
    out = []
    final = cfg.results_dir / "final"
    if (final / "metrics.json").exists():
        rep = load_json(final / "metrics.json")
        out.append("### Final models\n")
        out.append("| model | split | n | ROC AUC | AP | F1 | balanced acc. | Brier |")
        out.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for m, entry in rep["models"].items():
            for s, met in entry["splits"].items():
                out.append(
                    f"| {m} | {s} | {met['n']:,} | {met['roc_auc']:.4f} | {met['avg_precision']:.4f} | {met['f1']:.3f} | {met['balanced_accuracy']:.3f} | {met['brier']:.4f} |"
                )
        out.append("")
    p = cfg.results_dir / "model_selection.csv"
    if p.exists():
        df = pd.read_csv(p)
        out.append("### Model screening\n")
        out.append("| model | ROC AUC | AP | F1 | fit time [s] |")
        out.append("|---|---:|---:|---:|---:|")
        for _, r in df.iterrows():
            out.append(
                f"| {r['model']} | {r['roc_auc_mean']:.4f} ± {r['roc_auc_std']:.4f} | {r['avg_precision_mean']:.4f} | {r['f1_mean']:.3f} | {r['fit_seconds_mean']:.0f} |"
            )
        out.append("")
    p = cfg.results_dir / "feature_ablation.csv"
    if p.exists():
        df = pd.read_csv(p)
        out.append("### Feature-set ablation\n")
        out.append("| feature set | features | CV AUC | geo AUC | time AUC |")
        out.append("|---|---:|---:|---:|---:|")
        for _, r in df.iterrows():
            out.append(
                f"| {r['feature_set']} | {r['n_features']} | {r['cv_auc']:.4f} ± {r['cv_auc_std']:.4f} | {r['geo_auc']:.4f} | {r['time_auc']:.4f} |"
            )
        out.append("")
    if (final / "calibration.csv").exists():
        df = pd.read_csv(final / "calibration.csv")
        out.append("### Calibration (fitted on the validation fold)\n")
        out.append("| model | calibration | split | Brier | log loss | ECE |")
        out.append("|---|---|---|---:|---:|---:|")
        for _, r in df.iterrows():
            out.append(
                f"| {r['model']} | {r['calibration']} | {r['split']} | {r['brier']:.4f} | {r['log_loss']:.4f} | {r['ece']:.4f} |"
            )
        out.append("")
    (cfg.results_dir / "tables.md").write_text("\n".join(out))
    print(f"wrote {cfg.results_dir / 'tables.md'}")


def main() -> None:
    parser = base_parser(__doc__)
    args = parser.parse_args()
    cfg = load_config(args)
    plots.setup_style()
    fig_dir = cfg.figures_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_feature_distributions(cfg, fig_dir)
    fig_model_selection(cfg, fig_dir)
    fig_ablation(cfg, fig_dir)
    fig_tuning(cfg, fig_dir)
    pred_path = cfg.results_dir / "final" / "predictions.parquet"
    if pred_path.exists():
        preds = pd.read_parquet(pred_path)
        fig_gallery(cfg, preds, fig_dir)
        fig_final(cfg, preds, fig_dir)
    write_tables(cfg)
    print(f"figures in {fig_dir}")


if __name__ == "__main__":
    main()
