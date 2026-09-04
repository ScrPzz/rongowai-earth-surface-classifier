"""Figures with one shared style. Every function saves to ``path`` and returns nothing."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from ..models.calibration import reliability_table

COLORS = {
    "water": "#2a6f97",
    "land": "#b5651d",
    "accent": "#4a4e69",
    "grey": "#9a9a9a",
    "series": [
        "#2a6f97",
        "#b5651d",
        "#4a4e69",
        "#7a9e7e",
        "#c1666b",
        "#e0a458",
        "#5b8e7d",
        "#8c7aa9",
    ],
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
        }
    )


def _save(fig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_ddm_gallery(
    ddms: np.ndarray, titles: list[str], path: str | Path, ncols: int = 4, log: bool = True
) -> None:
    """Grid of DDM images (delay on the vertical axis, Doppler on the horizontal one)."""
    n = len(ddms)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.9 * ncols, 3.6 * nrows), squeeze=False)
    for k, ax in enumerate(axes.ravel()):
        ax.grid(False)
        if k >= n:
            ax.axis("off")
            continue
        img = ddms[k].astype(float)
        if log:
            img = np.log10(np.maximum(img, 1.0))
        ax.imshow(img, aspect="auto", cmap="viridis", origin="upper", interpolation="nearest")
        ax.set_title(titles[k], fontsize=8)
        ax.set_xticks(range(5))
        ax.set_xticklabels([f"{(i - 2) * 250:+d}" for i in range(5)], fontsize=6)
        ax.set_yticks([0, 10, 20, 30, 39])
        ax.set_yticklabels([f"{i * 0.125:.1f}" for i in (0, 10, 20, 30, 39)], fontsize=6)
        if k % ncols == 0:
            ax.set_ylabel("delay [chip]")
        if k >= n - ncols:
            ax.set_xlabel("Doppler [Hz]")
    _save(fig, path)


def plot_peak_region(ddm: np.ndarray, mask: np.ndarray, path: str | Path, title: str = "") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(4.2, 4.2))
    for ax in axes:
        ax.grid(False)
        ax.set_xticks(range(5))
    img = np.log10(np.maximum(ddm.astype(float), 1.0))
    axes[0].imshow(img, aspect="auto", cmap="viridis", interpolation="nearest")
    axes[0].set_title("DDM (log10 counts)")
    axes[1].imshow(img, aspect="auto", cmap="viridis", interpolation="nearest", alpha=0.55)
    rr, cc = np.where(mask)
    axes[1].scatter(cc, rr, s=14, c="#ff3b3b", marker="s", linewidths=0)
    axes[1].set_title(f"peak region ({mask.sum()} px)")
    if title:
        fig.suptitle(title, fontsize=9)
    _save(fig, path)


def plot_roc_pr(
    curves: dict[str, tuple[np.ndarray, np.ndarray]], path: str | Path, title: str = ""
) -> None:
    """``curves`` maps a label to ``(y_true, score)``."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for (name, (y, p)), color in zip(curves.items(), COLORS["series"] * 3, strict=False):
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        axes[0].plot(fpr, tpr, label=name, color=color, lw=1.4)
        axes[1].plot(rec, prec, label=name, color=color, lw=1.4)
    axes[0].plot([0, 1], [0, 1], ls="--", color=COLORS["grey"], lw=0.8)
    axes[0].set(xlabel="false positive rate", ylabel="true positive rate", title="ROC")
    axes[1].set(xlabel="recall (land)", ylabel="precision (land)", title="precision-recall")
    axes[0].legend(fontsize=7, loc="lower right")
    if title:
        fig.suptitle(title)
    _save(fig, path)


def plot_reliability(
    series: dict[str, tuple[np.ndarray, np.ndarray]], path: str | Path, n_bins: int = 15
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for (name, (y, p)), color in zip(series.items(), COLORS["series"] * 3, strict=False):
        t = reliability_table(y, p, n_bins).dropna()
        axes[0].plot(t["mean_p"], t["frac_pos"], marker="o", ms=3, lw=1.2, label=name, color=color)
        axes[1].hist(p, bins=n_bins, range=(0, 1), histtype="step", label=name, color=color, lw=1.2)
    axes[0].plot([0, 1], [0, 1], ls="--", color=COLORS["grey"], lw=0.8)
    axes[0].set(
        xlabel="predicted probability of land",
        ylabel="observed fraction of land",
        title="reliability",
    )
    axes[1].set(
        xlabel="predicted probability of land",
        ylabel="count",
        title="score distribution",
        yscale="log",
    )
    axes[0].legend(fontsize=7)
    _save(fig, path)


def plot_feature_importance(
    names: list[str], values: np.ndarray, path: str | Path, top: int = 25, title: str = ""
) -> None:
    order = np.argsort(values)[::-1][:top]
    fig, ax = plt.subplots(figsize=(6, 0.25 * top + 0.8))
    ax.barh([names[i] for i in order][::-1], values[order][::-1], color=COLORS["accent"])
    ax.set_xlabel("importance")
    if title:
        ax.set_title(title)
    _save(fig, path)


def plot_bars(
    df: pd.DataFrame,
    x: str,
    ys: list[str],
    path: str | Path,
    ylabel: str = "",
    title: str = "",
    ylim=None,
    err: list[str] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(max(5, 0.55 * len(df) + 1.5), 3.4))
    width = 0.8 / len(ys)
    pos = np.arange(len(df))
    for k, y in enumerate(ys):
        e = df[err[k]] if err else None
        ax.bar(
            pos + k * width,
            df[y],
            width=width,
            label=y,
            color=COLORS["series"][k % 8],
            yerr=e,
            capsize=2,
        )
    ax.set_xticks(pos + width * (len(ys) - 1) / 2)
    ax.set_xticklabels(df[x], rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    if title:
        ax.set_title(title)
    if len(ys) > 1:
        ax.legend(fontsize=7)
    _save(fig, path)


def plot_map(
    lat: np.ndarray,
    lon: np.ndarray,
    values: np.ndarray,
    path: str | Path,
    title: str = "",
    cmap: str = "coolwarm",
    vmin=0,
    vmax=1,
    label: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 6))
    sc = ax.scatter(
        lon, lat, c=values, s=1.2, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0, rasterized=True
    )
    ax.set(xlabel="longitude", ylabel="latitude", title=title)
    ax.set_aspect(1 / np.cos(np.deg2rad(-41)))
    cb = fig.colorbar(sc, ax=ax, shrink=0.7)
    cb.set_label(label)
    _save(fig, path)


def plot_lines(
    df: pd.DataFrame,
    x: str,
    ys: list[str],
    path: str | Path,
    ylabel: str = "",
    title: str = "",
    logx: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for k, y in enumerate(ys):
        ax.plot(df[x], df[y], marker="o", ms=3, lw=1.3, label=y, color=COLORS["series"][k % 8])
    ax.set(xlabel=x, ylabel=ylabel, title=title)
    if logx:
        ax.set_xscale("log")
    if len(ys) > 1:
        ax.legend(fontsize=7)
    _save(fig, path)
