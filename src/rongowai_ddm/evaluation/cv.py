"""Grouped cross-validation loop shared by the screening and ablation scripts."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pandas as pd

from ..models.xgb import fit_with_fallback
from .metrics import binary_metrics


def run_grouped_cv(
    make_model: Callable[[], object],
    X: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    needs_validation: bool = False,
    threshold: float = 0.5,
    label: str = "",
) -> tuple[np.ndarray, pd.DataFrame]:
    """Out-of-fold predictions and per-fold metrics for a model built by ``make_model``.

    ``folds`` holds the flight-level fold id of every row. When
    ``needs_validation`` is set, the model's ``fit`` receives a validation set
    made of one of the *training* folds (never the held-out one).
    """
    fold_ids = np.sort(np.unique(folds))
    oof = np.full(len(y), np.nan, dtype=np.float64)
    rows = []
    for k in fold_ids:
        te = folds == k
        tr = ~te
        t0 = time.time()
        model = make_model()
        if needs_validation:
            va_fold = fold_ids[(int(np.flatnonzero(fold_ids == k)[0]) + 1) % len(fold_ids)]
            va = folds == va_fold
            tr = tr & ~va
            fit_with_fallback(model, X[tr], y[tr], X[va], y[va])
        else:
            fit_with_fallback(model, X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        oof[te] = p
        m = binary_metrics(y[te], p, threshold)
        m.update(
            {
                "model": label,
                "fold": int(k),
                "n_train": int(tr.sum()),
                "fit_seconds": time.time() - t0,
            }
        )
        rows.append(m)
        print(
            f"  [{label}] fold {k}: auc={m['roc_auc']:.4f} ap={m['avg_precision']:.4f} f1={m['f1']:.4f} ({m['fit_seconds']:.0f}s)",
            flush=True,
        )
    return oof, pd.DataFrame(rows)


def summarize_folds(per_fold: pd.DataFrame, by: str = "model") -> pd.DataFrame:
    cols = ["roc_auc", "avg_precision", "f1", "accuracy", "brier", "fit_seconds"]
    agg = per_fold.groupby(by)[cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()
