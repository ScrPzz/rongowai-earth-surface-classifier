"""Beta calibration for binary classifier outputs (Kull, Silva Filho & Flach 2017).

Beta calibration extends Platt scaling with a three-parameter family that is
well-suited to GBDT outputs (whose logits are non-Gaussian). The calibrator
fits

.. math::

    \\sigma^{-1}(\\hat{p}_{\\text{cal}}) = a \\log p - b \\log(1 - p) + c

where ``p`` is the raw classifier probability and ``(a, b, c)`` are estimated
by maximum likelihood. The fit reduces to a logistic regression on the
two-dimensional feature vector ``(log p, -log(1 - p))``.

Compared to alternatives:

- **Platt (sigmoid)**: 2 parameters, fits a logistic distortion of the logit.
- **Isotonic**: non-parametric, monotone; more flexible but needs more data
  and is prone to overfitting on small held-outs.
- **Temperature scaling**: a single parameter; best-suited to NN logits.

Beta calibration's three-parameter form gives more flexibility than Platt
without the data-efficiency cost of Isotonic — empirically the right tool
for XGBoost / LightGBM / CatBoost outputs on tabular binary tasks.

Reference
---------
Kull, M., Silva Filho, T. M. & Flach, P. (2017). "Beta calibration: a
well-founded and easily implemented improvement on logistic calibration
for binary classifiers." AISTATS, PMLR 54:623–631.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression


class BetaCalibrator(BaseEstimator):
    """Sklearn-compatible Beta calibration.

    Fit on (uncalibrated_prob, true_label); then call :meth:`transform` or
    :meth:`predict_proba` to obtain calibrated probabilities.
    """

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps

    def _features(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64).clip(self.eps, 1.0 - self.eps)
        return np.column_stack([np.log(p), -np.log(1.0 - p)])

    def fit(self, p: np.ndarray, y: np.ndarray) -> "BetaCalibrator":
        """Estimate ``(a, b, c)`` by maximum likelihood on held-out scores."""
        X = self._features(p)
        y = np.asarray(y, dtype=np.int32)
        # Use a high C to approximate unregularised MLE; lbfgs handles 2-D well.
        self._lr = LogisticRegression(solver="lbfgs", C=1e6, max_iter=1000)
        self._lr.fit(X, y)
        self.a_ = float(self._lr.coef_[0, 0])
        self.b_ = float(self._lr.coef_[0, 1])
        self.c_ = float(self._lr.intercept_[0])
        return self

    def predict_proba(self, p: np.ndarray) -> np.ndarray:
        """Return shape ``(N, 2)`` array of [P(class=0), P(class=1)]."""
        X = self._features(p)
        proba_pos = self._lr.predict_proba(X)[:, 1]
        return np.column_stack([1.0 - proba_pos, proba_pos])

    def predict(self, p: np.ndarray) -> np.ndarray:
        return self.predict_proba(p).argmax(axis=1)

    def transform(self, p: np.ndarray) -> np.ndarray:
        """Convenience: return only ``P(class=1)`` for the input probabilities."""
        return self.predict_proba(p)[:, 1]


def _self_test() -> None:
    """Synthetic check: an overconfident classifier should be corrected by Beta."""
    from sklearn.metrics import brier_score_loss, log_loss

    rng = np.random.default_rng(42)
    n = 5000
    # True latent probabilities cluster around 0.3 and 0.7
    true_p = rng.choice([0.3, 0.7], size=n)
    y = (rng.uniform(size=n) < true_p).astype(int)

    # Simulate an overconfident classifier that pushes scores toward {0, 1}
    base = np.where(true_p > 0.5, 0.95, 0.05)
    raw_scores = (base + rng.normal(0, 0.05, n)).clip(1e-3, 1 - 1e-3)

    cal = BetaCalibrator()
    cal.fit(raw_scores, y)
    calibrated = cal.transform(raw_scores)

    brier_raw = brier_score_loss(y, raw_scores)
    brier_cal = brier_score_loss(y, calibrated)
    log_raw = log_loss(y, raw_scores)
    log_cal = log_loss(y, calibrated)

    print("=== beta_calibration._self_test ===")
    print(f"  fitted params : a={cal.a_:.4f}, b={cal.b_:.4f}, c={cal.c_:.4f}")
    print(f"  Brier raw     : {brier_raw:.4f}")
    print(f"  Brier calibrated: {brier_cal:.4f}  (Δ {brier_cal - brier_raw:+.4f})")
    print(f"  LogLoss raw     : {log_raw:.4f}")
    print(f"  LogLoss calibrated: {log_cal:.4f}  (Δ {log_cal - log_raw:+.4f})")
    assert brier_cal < brier_raw, "calibration must lower Brier"
    assert log_cal < log_raw, "calibration must lower log loss"
    assert (calibrated >= 0).all() and (calibrated <= 1).all(), "probabilities out of range"
    print("  ✓ all assertions passed")


if __name__ == "__main__":
    _self_test()
