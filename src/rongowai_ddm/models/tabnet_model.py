"""TabNet wrapper with the same fit/predict interface as the other families."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .xgb import is_oom, resolve_device

TABNET_DEFAULT_PARAMS: dict = {
    "n_d": 64,
    "n_a": 64,
    "n_steps": 4,
    "gamma": 1.5,
    "lambda_sparse": 1e-4,
    "mask_type": "entmax",
    "lr": 2e-2,
    "batch_size": 2048,
}


class TabNetModel:
    """Median imputation + standardisation + ``TabNetClassifier``."""

    def __init__(
        self,
        params: dict | None = None,
        device: str = "cuda",
        max_epochs: int = 100,
        patience: int = 15,
        seed: int = 42,
        verbose: int = 0,
    ) -> None:
        self.params = dict(TABNET_DEFAULT_PARAMS)
        self.params.update(params or {})
        self.device = resolve_device(device) if torch.cuda.is_available() else "cpu"
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self.verbose = verbose
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.model = None
        self.history: dict = {}

    def _prep(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if fit:
            X = self.scaler.fit_transform(self.imputer.fit_transform(X))
        else:
            X = self.scaler.transform(self.imputer.transform(X))
        return X.astype(np.float32)

    def fit(self, X, y, X_val=None, y_val=None) -> TabNetModel:
        try:
            return self._fit(X, y, X_val, y_val)
        except Exception as exc:  # noqa: BLE001
            if not is_oom(exc) or self.device == "cpu":
                raise
            print("[tabnet] GPU out of memory; training on CPU", flush=True)
            torch.cuda.empty_cache()
            self.device = "cpu"
            return self._fit(X, y, X_val, y_val)

    def _fit(self, X, y, X_val=None, y_val=None) -> TabNetModel:
        from pytorch_tabnet.tab_model import TabNetClassifier

        p = dict(self.params)
        lr = p.pop("lr")
        batch_size = int(p.pop("batch_size"))
        virtual_batch_size = min(256, batch_size)
        self.model = TabNetClassifier(
            **p,
            optimizer_fn=torch.optim.AdamW,
            optimizer_params={"lr": lr},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params={"step_size": 10, "gamma": 0.9},
            seed=self.seed,
            verbose=self.verbose,
            device_name=self.device,
        )
        Xt = self._prep(X, fit=True)
        yt = np.asarray(y, dtype=np.int64)
        eval_set, eval_name = [], []
        if X_val is not None:
            eval_set = [(self._prep(X_val), np.asarray(y_val, dtype=np.int64))]
            eval_name = ["val"]
        self.model.fit(
            Xt,
            yt,
            eval_set=eval_set,
            eval_name=eval_name,
            eval_metric=["auc"],
            max_epochs=self.max_epochs,
            patience=self.patience if eval_set else 0,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            drop_last=False,
        )
        self.history = dict(self.model.history.history) if hasattr(self.model, "history") else {}
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(self._prep(X))

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_

    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        self.model.save_model(str(path.with_suffix("")))  # writes <path>.zip
        joblib.dump(
            {"imputer": self.imputer, "scaler": self.scaler, "params": self.params},
            path.with_suffix(".prep.joblib"),
        )


def suggest_tabnet_params(trial) -> dict:
    n_d = trial.suggest_categorical("n_d", [32, 64, 128])
    return {
        "n_d": n_d,
        "n_a": n_d,
        "n_steps": trial.suggest_int("n_steps", 3, 6),
        "gamma": trial.suggest_float("gamma", 1.0, 2.0, step=0.1),
        "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True),
        "mask_type": trial.suggest_categorical("mask_type", ["entmax", "sparsemax"]),
        "lr": trial.suggest_float("lr", 5e-4, 5e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [1024, 2048, 4096]),
    }
