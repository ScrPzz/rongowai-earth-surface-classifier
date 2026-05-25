"""XGBoost baseline wrapper that respects the P0.1 flight-level fold harness.

Mirrors the production training configuration extracted from
``deliver/[model_training]geok_xgboost.ipynb``: hist tree method, GPU when
available, RandomizedSearchCV over the same hyperparameter distributions,
StandardScaler on features, early stopping on a held-out validation slice.

The crucial change is the **CV strategy**: instead of a random 5-fold over
samples, we feed an explicit list of (train_idx, val_idx) pairs derived from
``StratifiedGroupKFold(groups=flight_id)`` on the train_cv pool from P0.1.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

# Hyperparameter distributions identical to production search space.
XGB_PARAM_DISTRIBUTIONS = {
    "n_estimators": [500],
    "max_depth": randint(3, 12),
    "min_child_weight": randint(1, 20),
    "learning_rate": loguniform(0.001, 0.3),
    "subsample": uniform(0.5, 0.45),
    "colsample_bytree": uniform(0.5, 0.45),
    "colsample_bylevel": uniform(0.5, 0.45),
    "colsample_bynode": uniform(0.5, 0.45),
    "gamma": loguniform(1e-8, 1.0),
    "reg_alpha": loguniform(1e-8, 100),
    "reg_lambda": loguniform(0.1, 100),
}

# Sensible defaults used during smoke tests when full HP search is impractical.
XGB_DEFAULT_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "min_child_weight": 1,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}


def _xgb_device() -> str:
    """Choose CUDA if available, else CPU."""
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@dataclass
class HoldoutMetrics:
    """Comprehensive per-holdout metrics."""

    split_name: str
    n_samples: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    brier: float
    log_loss: float


class XGBoostBaseline:
    """End-to-end trainer for the 78-D XGBoost baseline under P0.1 folds."""

    def __init__(
        self,
        device: Optional[str] = None,
        random_state: int = 42,
    ) -> None:
        self.device = device or _xgb_device()
        self.random_state = random_state
        self.scaler: Optional[StandardScaler] = None
        self.best_params: Optional[dict] = None
        self.model: Optional[xgb.XGBClassifier] = None

    # ------------------------------------------------------------------
    # Hyperparameter search
    # ------------------------------------------------------------------

    def hp_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        groups_train: np.ndarray,
        n_iter: int = 250,
        n_splits: int = 5,
    ) -> dict:
        """RandomizedSearchCV with StratifiedGroupKFold (groups = flight_id)."""
        self.scaler = StandardScaler().fit(X_train)
        X_scaled = self.scaler.transform(X_train)

        cv = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )

        base = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device=self.device,
            random_state=self.random_state,
            verbosity=0,
        )
        search = RandomizedSearchCV(
            estimator=base,
            param_distributions=XGB_PARAM_DISTRIBUTIONS,
            n_iter=n_iter,
            cv=cv.split(X_scaled, y_train, groups=groups_train),
            scoring="roc_auc",
            verbose=2,
            random_state=self.random_state,
            return_train_score=True,
            n_jobs=1,  # XGBoost itself parallelises internally
        )
        search.fit(X_scaled, y_train)
        self.best_params = dict(search.best_params_)
        return self.best_params

    # ------------------------------------------------------------------
    # Final fit
    # ------------------------------------------------------------------

    def fit_final(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        params: Optional[dict] = None,
        val_fraction: float = 0.2,
        early_stopping_rounds: int = 10,
        n_estimators_max: int = 10000,
    ) -> "XGBoostBaseline":
        """Train the final model with early stopping on a stratified holdout slice."""
        if self.scaler is None:
            self.scaler = StandardScaler().fit(X_train)
        X_scaled = self.scaler.transform(X_train)

        params = dict(params or self.best_params or XGB_DEFAULT_PARAMS)
        params["n_estimators"] = n_estimators_max

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_scaled,
            y_train,
            test_size=val_fraction,
            random_state=self.random_state,
            stratify=y_train,
        )

        self.model = xgb.XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device=self.device,
            random_state=self.random_state,
            early_stopping_rounds=early_stopping_rounds,
            verbosity=0,
        )
        self.model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return self

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str,
    ) -> HoldoutMetrics:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not fitted; call hp_search()/fit_final() first")
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)[:, 1]
        pred = (proba >= 0.5).astype(int)
        return HoldoutMetrics(
            split_name=split_name,
            n_samples=int(len(y)),
            accuracy=float((pred == y).mean()),
            precision=float(((pred == 1) & (y == 1)).sum() / max(int((pred == 1).sum()), 1)),
            recall=float(((pred == 1) & (y == 1)).sum() / max(int((y == 1).sum()), 1)),
            f1=float(2 * ((pred == 1) & (y == 1)).sum() / max(int((pred == 1).sum() + (y == 1).sum()), 1)),
            roc_auc=float(roc_auc_score(y, proba)),
            pr_auc=float(average_precision_score(y, proba)),
            brier=float(brier_score_loss(y, proba)),
            log_loss=float(log_loss(y, proba, labels=[0, 1])),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, model_path: Path, scaler_path: Path, params_path: Path) -> None:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not fitted; cannot save")
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        with Path(params_path).open("w") as f:
            json.dump(self.best_params or XGB_DEFAULT_PARAMS, f, indent=2)


def metrics_to_dataframe(metrics: list[HoldoutMetrics]) -> pd.DataFrame:
    """Convenience: flatten a list of HoldoutMetrics to a DataFrame."""
    return pd.DataFrame([asdict(m) for m in metrics])
