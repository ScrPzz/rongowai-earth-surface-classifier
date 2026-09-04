"""Post-hoc probability calibration and calibration metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-7


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)


class PlattCalibrator:
    """Logistic regression on the logit of the score (two parameters)."""

    def fit(self, p, y):
        z = np.log(_clip(p) / (1 - _clip(p)))[:, None]
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(z, np.asarray(y, dtype=int))
        return self

    def transform(self, p):
        z = np.log(_clip(p) / (1 - _clip(p)))[:, None]
        return self.lr_.predict_proba(z)[:, 1]


class BetaCalibrator:
    """Beta calibration (Kull et al. 2017): logistic regression on ``(log p, -log(1-p))``."""

    def _features(self, p):
        p = _clip(p)
        return np.column_stack([np.log(p), -np.log(1 - p)])

    def fit(self, p, y):
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(
            self._features(p), np.asarray(y, dtype=int)
        )
        return self

    def transform(self, p):
        return self.lr_.predict_proba(self._features(p))[:, 1]


class IsotonicCalibrator:
    def fit(self, p, y):
        self.iso_ = IsotonicRegression(out_of_bounds="clip").fit(
            np.asarray(p, dtype=float), np.asarray(y, dtype=float)
        )
        return self

    def transform(self, p):
        return self.iso_.predict(np.asarray(p, dtype=float))


CALIBRATORS = {"platt": PlattCalibrator, "beta": BetaCalibrator, "isotonic": IsotonicCalibrator}


def make_calibrator(method: str):
    return CALIBRATORS[method]()


def reliability_table(y, p, n_bins: int = 15) -> pd.DataFrame:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append(
            {
                "bin_lo": edges[b],
                "bin_hi": edges[b + 1],
                "n": int(m.sum()),
                "mean_p": float(p[m].mean()) if m.any() else np.nan,
                "frac_pos": float(y[m].mean()) if m.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(y, p, n_bins: int = 15) -> float:
    t = reliability_table(y, p, n_bins).dropna()
    if t.empty:
        return float("nan")
    return float((t["n"] / t["n"].sum() * (t["mean_p"] - t["frac_pos"]).abs()).sum())
