"""XGBoost wrapper: defaults, Optuna search space, fitting with early stopping."""

from __future__ import annotations

import numpy as np
import xgboost as xgb

XGB_DEFAULT_PARAMS: dict = {
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "colsample_bynode": 1.0,
    "min_child_weight": 1.0,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}


def make_xgb(
    params: dict | None = None,
    device: str = "cuda",
    n_estimators: int = 800,
    seed: int = 42,
    early_stopping_rounds: int | None = None,
) -> xgb.XGBClassifier:
    p = dict(XGB_DEFAULT_PARAMS)
    p.update(params or {})
    return xgb.XGBClassifier(
        **p,
        n_estimators=n_estimators,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device=device,
        random_state=seed,
        early_stopping_rounds=early_stopping_rounds,
        verbosity=0,
    )


def fit_xgb(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict | None = None,
    device: str = "cuda",
    n_estimators: int = 3000,
    early_stopping_rounds: int = 50,
    seed: int = 42,
) -> xgb.XGBClassifier:
    """Fit with early stopping on ``(X_val, y_val)``; ``model.best_iteration`` holds the stopping round."""
    model = make_xgb(params, device, n_estimators, seed, early_stopping_rounds)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def suggest_xgb_params(trial) -> dict:
    """Optuna search space (log scales for the regularisation terms)."""
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-3, 5.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 50.0, log=True),
    }
