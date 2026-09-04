"""Model families used in the screening step, built with one function."""

from __future__ import annotations

from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .tabnet_model import TabNetModel
from .xgb import make_xgb

MODEL_NAMES = ("logreg", "rf", "extratrees", "hgb", "lgbm", "xgb", "mlp", "tabnet")
NEEDS_VALIDATION_SET = ("tabnet",)


def make_model(name: str, device: str = "cuda", seed: int = 42, n_jobs: int = -1):
    if name == "logreg":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0),
        )
    if name == "rf":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=300, min_samples_leaf=5, n_jobs=n_jobs, random_state=seed
            ),
        )
    if name == "extratrees":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=300, min_samples_leaf=5, n_jobs=n_jobs, random_state=seed
            ),
        )
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=500, learning_rate=0.1, random_state=seed)
    if name == "lgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=n_jobs,
            verbose=-1,
        )
    if name == "xgb":
        return make_xgb(device=device, n_estimators=800, seed=seed)
    if name == "mlp":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(256, 128), early_stopping=True, max_iter=100, random_state=seed
            ),
        )
    if name == "tabnet":
        return TabNetModel(device=device, max_epochs=40, patience=8, seed=seed)
    raise KeyError(f"unknown model {name!r}; known: {MODEL_NAMES}")
