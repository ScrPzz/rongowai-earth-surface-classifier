"""Adversarial validation: detect distribution shift between two sample pools.

An adversarial classifier tries to predict, given a sample, *which pool it
came from*. If the classifier's AUC is significantly above 0.5, the two
pools are distinguishable — i.e. they are drawn from different distributions
and the model trained on pool A may not generalize to pool B.

The pattern was popularised by Kaggle competitors and has standard uses:

* **Sanity-checking CV folds**: predict "is this sample in fold 0?". AUC
  should be ≈ 0.5. If it isn't, the fold assignment is leaking structure.
* **Train-vs-test drift detection**: predict "is this sample from the
  production / hold-out distribution?". Feature importances reveal *which*
  variables drive the shift, informing importance reweighting or feature
  engineering corrections.

This module provides one reusable function that runs a stratified-K-fold
adversarial XGBoost and reports mean AUC, per-fold AUCs, top feature
importances, and a textual verdict.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


def _label_encode_categoricals(X: pd.DataFrame) -> pd.DataFrame:
    """Label-encode object / category columns; trees do not need one-hot."""
    X = X.copy()
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = pd.Categorical(X[col]).codes
    return X


def _interpret(auc: float) -> str:
    if auc < 0.55:
        return "low drift (≈ i.i.d.)"
    if auc < 0.65:
        return "moderate drift"
    if auc < 0.80:
        return "high drift"
    return "severe drift (highly separable distributions)"


def run_adversarial_validation(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    target_col: str,
    n_splits: int = 5,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    top_k_features: int = 10,
) -> dict:
    """Train an XGBoost adversarial classifier and report shift diagnostics.

    Parameters
    ----------
    df
        DataFrame containing ``feature_cols`` and ``target_col``.
    feature_cols
        Column names used as adversarial features. Object / category columns
        are label-encoded automatically.
    target_col
        Binary target column. Convention: 0 = pool A (e.g. train), 1 = pool B
        (e.g. holdout).
    n_splits
        Number of stratified K-fold splits used to estimate the adversarial
        AUC. Defaults to 5.

    Returns
    -------
    dict
        ``mean_auc``, ``std_auc``, ``aucs_per_fold``, ``n_pos``, ``n_neg``,
        ``top_features`` (list of ``{feature, importance}`` dicts), and a
        textual ``verdict``.
    """
    feature_cols = list(feature_cols)
    if target_col not in df.columns:
        raise KeyError(f"target_col '{target_col}' not found in DataFrame")
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"feature_cols not found in DataFrame: {missing}")

    X = _label_encode_categoricals(df[feature_cols])
    y = df[target_col].astype(int).values

    if len(np.unique(y)) != 2:
        raise ValueError(
            f"target_col '{target_col}' must have exactly 2 distinct values; "
            f"got {np.unique(y)}"
        )

    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    aucs: list[float] = []
    importances = np.zeros(len(feature_cols))

    for train_idx, val_idx in skf.split(X, y):
        clf = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            eval_metric="auc",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
            tree_method="hist",
        )
        clf.fit(X.iloc[train_idx], y[train_idx])
        proba = clf.predict_proba(X.iloc[val_idx])[:, 1]
        aucs.append(float(roc_auc_score(y[val_idx], proba)))
        importances += clf.feature_importances_

    importances /= n_splits
    feature_importance_pairs = sorted(
        zip(feature_cols, importances), key=lambda pair: -pair[1]
    )

    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))

    return {
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "aucs_per_fold": aucs,
        "n_pos": int(y.sum()),
        "n_neg": int(len(y) - y.sum()),
        "top_features": [
            {"feature": name, "importance": float(imp)}
            for name, imp in feature_importance_pairs[:top_k_features]
        ],
        "verdict": _interpret(mean_auc),
    }
