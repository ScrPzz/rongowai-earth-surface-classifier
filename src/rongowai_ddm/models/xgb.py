"""XGBoost wrapper: defaults, Optuna search space, fitting with early stopping."""

from __future__ import annotations

import subprocess

import numpy as np
import xgboost as xgb

MIN_FREE_GPU_GB = 1.5


def gpu_free_gb() -> float | None:
    """Free GPU memory in GB from nvidia-smi (no CUDA context is created); ``None`` without a GPU."""
    try:
        out = (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            .stdout.strip()
            .splitlines()[0]
        )
        return float(out) / 1024.0
    except Exception:  # noqa: BLE001
        return None


def resolve_device(requested: str = "cuda", min_free_gb: float = MIN_FREE_GPU_GB) -> str:
    """Use the GPU only when enough memory is free (the GPU may be shared with other jobs)."""
    if requested != "cuda":
        return requested
    free = gpu_free_gb()
    return "cuda" if free is not None and free >= min_free_gb else "cpu"


def is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower() or "bad_alloc" in str(exc).lower()


def fit_with_fallback(model, *args, **kwargs):
    """Fit; on a GPU out-of-memory error, refit the same model on the CPU."""
    try:
        return model.fit(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        if not is_oom(exc) or not hasattr(model, "set_params"):
            raise
        print("[xgb] GPU out of memory; refitting on CPU", flush=True)
        model.set_params(device="cpu")
        return model.fit(*args, **kwargs)


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
        device=resolve_device(device),
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
    fit_with_fallback(model, X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
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
