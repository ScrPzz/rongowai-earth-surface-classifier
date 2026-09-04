"""Adversarial validation: can a classifier tell two pools apart, and with which features?"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ..models.xgb import make_xgb


def adversarial_validation(
    X_a: np.ndarray,
    X_b: np.ndarray,
    feature_names: list[str],
    n_splits: int = 5,
    seed: int = 42,
    device: str = "cuda",
    top_k: int = 15,
    max_per_pool: int | None = 200_000,
) -> dict:
    """Train a classifier to separate pool A (0) from pool B (1); report AUC and top features."""
    rng = np.random.default_rng(seed)
    if max_per_pool is not None:
        if len(X_a) > max_per_pool:
            X_a = X_a[rng.choice(len(X_a), max_per_pool, replace=False)]
        if len(X_b) > max_per_pool:
            X_b = X_b[rng.choice(len(X_b), max_per_pool, replace=False)]
    X = np.concatenate([X_a, X_b]).astype(np.float32)
    y = np.concatenate([np.zeros(len(X_a)), np.ones(len(X_b))]).astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs, imp = [], np.zeros(X.shape[1])
    for tr, va in skf.split(X, y):
        model = make_xgb(
            {"max_depth": 5, "learning_rate": 0.1}, device=device, n_estimators=200, seed=seed
        )
        model.fit(X[tr], y[tr])
        aucs.append(float(roc_auc_score(y[va], model.predict_proba(X[va])[:, 1])))
        imp += model.feature_importances_
    imp /= n_splits
    order = np.argsort(-imp)[:top_k]
    auc = float(np.mean(aucs))
    verdict = (
        "indistinguishable"
        if auc < 0.55
        else "mild shift"
        if auc < 0.65
        else "strong shift"
        if auc < 0.8
        else "severe shift"
    )
    return {
        "mean_auc": auc,
        "std_auc": float(np.std(aucs)),
        "n_a": int(len(X_a)),
        "n_b": int(len(X_b)),
        "verdict": verdict,
        "top_features": [{"feature": feature_names[i], "importance": float(imp[i])} for i in order],
    }
