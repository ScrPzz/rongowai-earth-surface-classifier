"""Model wrappers for the rework pipeline (XGBoost baseline + future siblings)."""
from .xgboost_baseline import (
    XGBoostBaseline,
    XGB_PARAM_DISTRIBUTIONS,
    XGB_DEFAULT_PARAMS,
)

__all__ = ["XGBoostBaseline", "XGB_PARAM_DISTRIBUTIONS", "XGB_DEFAULT_PARAMS"]
