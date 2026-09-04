"""Binary classification metrics and breakdowns."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from ..models.calibration import expected_calibration_error

SNR_BIN_EDGES = (-np.inf, -5.0, 0.0, 5.0, 10.0, np.inf)
SNR_BIN_LABELS = ("< -5 dB", "-5 to 0 dB", "0 to 5 dB", "5 to 10 dB", "> 10 dB")


def binary_metrics(y, p, threshold: float = 0.5) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    two_classes = len(np.unique(y)) == 2
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)) if two_classes else float("nan"),
        "avg_precision": float(average_precision_score(y, p)) if two_classes else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "ece": expected_calibration_error(y, p),
        "threshold": float(threshold),
        "accuracy": (tp + tn) / max(len(y), 1),
        "balanced_accuracy": 0.5 * (rec + spec),
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1": 2 * prec * rec / max(prec + rec, 1e-12),
    }


def best_threshold(y, p, objective: str = "f1") -> float:
    """Threshold maximising F1, Youden's J or balanced accuracy on ``(y, p)``."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if objective == "f1":
        prec, rec, thr = precision_recall_curve(y, p)
        f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
        return float(thr[int(np.argmax(f1))])
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    j[~np.isfinite(thr)] = -np.inf
    return float(np.clip(thr[int(np.argmax(j))], 0.0, 1.0))


def metrics_by_group(
    df: pd.DataFrame, p: np.ndarray, group_col: str, threshold: float, min_n: int = 200
) -> pd.DataFrame:
    rows = []
    for key, idx in df.groupby(group_col, observed=True).indices.items():
        if len(idx) < min_n:
            continue
        m = binary_metrics(df["label"].to_numpy()[idx], np.asarray(p)[idx], threshold)
        m[group_col] = key
        rows.append(m)
    return pd.DataFrame(rows)


def snr_bins(snr: np.ndarray) -> pd.Categorical:
    return pd.cut(np.asarray(snr, dtype=float), bins=SNR_BIN_EDGES, labels=SNR_BIN_LABELS)


def coast_bins(dist_km: np.ndarray) -> pd.Categorical:
    """Absolute distance to the coast in bins; the label says which side."""
    d = np.abs(np.asarray(dist_km, dtype=float))
    return pd.cut(
        d,
        bins=(-0.001, 1, 3, 10, 30, np.inf),
        labels=("< 1 km", "1-3 km", "3-10 km", "10-30 km", "> 30 km"),
    )
