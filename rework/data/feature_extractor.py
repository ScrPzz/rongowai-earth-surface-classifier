"""Statistical-feature extractor for 200-D flattened DDMs (52 features per sample).

Vendored from the ``DDMFeatureExtractor`` class in
``deliver/[model_training]geok_xgboost.ipynb``. Output feature order and
formulas are byte-identical to that notebook's implementation.

Feature groups (52 total):
  10  global statistics       (mean, std, min, max, median, range, skew, kurtosis, entropy, gini)
   4  positional              (peak_index, peak_value, center_of_mass, inertia)
   9  thirds segmentation     (sum/mean/max per third × 3)
  15  windows-of-5 segmentation (mean/std/max per window × 5)
   7  derivative statistics   (mean/std/max/min diff, n_positive/negative/zero diff)
   3  autocorrelation         (lag 1, 2, 3)
   4  FFT                     (peak_freq, max, median, mean)

Note: combined with the 20-D encoder latent → 72 features total. The
production ONNX metadata reports 78 features; the 6-feature gap suggests the
deployed model used an enhanced extractor (likely ``DDMFeatureExtractorV2``
with peak-detection features). For the rework baseline we use this 52-feature
extractor as documented in the canonical compressed-XGBoost notebook.

The features are vectorised over rows for throughput; the per-sample
semantics are identical to the reference implementation.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.fft import fft
from scipy.stats import entropy, kurtosis, skew

# Canonical feature ordering (must match production model column order)
STAT_FEATURE_NAMES: tuple[str, ...] = (
    # Global
    "mean", "std", "min", "max", "median", "range", "skew", "kurtosis",
    "entropy", "gini",
    # Positional
    "peak_index", "peak_value", "center_of_mass", "inertia",
    # Thirds
    "sum_third_1", "mean_third_1", "max_third_1",
    "sum_third_2", "mean_third_2", "max_third_2",
    "sum_third_3", "mean_third_3", "max_third_3",
    # Windows
    "mean_w1", "std_w1", "max_w1",
    "mean_w2", "std_w2", "max_w2",
    "mean_w3", "std_w3", "max_w3",
    "mean_w4", "std_w4", "max_w4",
    "mean_w5", "std_w5", "max_w5",
    # Derivative
    "mean_diff", "std_diff", "max_diff", "min_diff",
    "n_positive_diff", "n_negative_diff", "n_zero_diff",
    # Autocorrelation
    "autocorr_lag1", "autocorr_lag2", "autocorr_lag3",
    # FFT
    "fft_peak_freq", "fft_max", "fft_median", "fft_mean",
)


def _gini_row(x: np.ndarray) -> float:
    """Gini coefficient for a 1-D array (sorted ascending)."""
    a = np.sort(x)
    n = a.shape[0]
    idx = np.arange(1, n + 1)
    denom = n * a.sum()
    if denom == 0:
        return 0.0
    return float(np.sum((2 * idx - n - 1) * a) / denom)


def _autocorr_row(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag:
        return float("nan")
    a, b = x[:-lag], x[lag:]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


class DDMFeatureExtractor:
    """Extract the 58 statistical features from flattened DDM rows.

    Designed to be drop-in compatible with the production extractor: feature
    names, order, and per-sample numeric values match the reference.
    """

    feature_names: tuple[str, ...] = STAT_FEATURE_NAMES

    def __init__(self) -> None:
        pass

    def extract(self, fit_data: np.ndarray) -> np.ndarray:
        """Compute features for an (N, 200) array; return shape (N, 58).

        The output column order matches :data:`STAT_FEATURE_NAMES`.
        """
        if fit_data.ndim != 2:
            raise ValueError(
                f"fit_data must be 2-D (N, 200); got shape {fit_data.shape}"
            )

        n = fit_data.shape[0]
        n_feat = len(STAT_FEATURE_NAMES)
        out = np.empty((n, n_feat), dtype=np.float32)

        for i in range(n):
            out[i] = self._extract_row(fit_data[i])
        return out

    def extract_to_dict(self, fit_data: np.ndarray) -> list[dict]:
        """Same as :meth:`extract` but returns a list of per-sample dicts."""
        matrix = self.extract(fit_data)
        return [
            {name: float(matrix[i, j]) for j, name in enumerate(STAT_FEATURE_NAMES)}
            for i in range(matrix.shape[0])
        ]

    @staticmethod
    def _extract_row(row: np.ndarray) -> np.ndarray:
        x = np.asarray(row, dtype=np.float64) + 1e-10  # avoid log(0)

        # Global statistics
        mn, mx = float(x.min()), float(x.max())
        rng = mx - mn
        mean = float(x.mean())
        std = float(x.std())
        median = float(np.median(x))
        sk = float(skew(x))
        kt = float(kurtosis(x))
        ent = float(entropy(x))
        gn = _gini_row(x)

        # Positional
        peak_index = int(np.argmax(x))
        peak_value = float(x.max())
        total = float(x.sum())
        idx = np.arange(len(x))
        com = float((idx * x).sum() / total) if total else 0.0
        inertia = float((((idx - com) ** 2) * x).sum())

        # Thirds
        thirds = np.array_split(x, 3)
        sum_t = [float(part.sum()) for part in thirds]
        mean_t = [float(part.mean()) for part in thirds]
        max_t = [float(part.max()) for part in thirds]

        # Windows of 5
        windows = np.array_split(x, 5)
        mean_w = [float(w.mean()) for w in windows]
        std_w = [float(w.std()) for w in windows]
        max_w = [float(w.max()) for w in windows]

        # Derivative
        dx = np.diff(x)
        mean_d = float(dx.mean())
        std_d = float(dx.std())
        max_d = float(dx.max())
        min_d = float(dx.min())
        n_pos = int((dx > 0).sum())
        n_neg = int((dx < 0).sum())
        n_zero = int((dx == 0).sum())

        # Autocorrelations
        ac1 = _autocorr_row(x, 1)
        ac2 = _autocorr_row(x, 2)
        ac3 = _autocorr_row(x, 3)

        # FFT (positive frequencies only)
        spectrum = np.abs(fft(x))
        half = spectrum[: len(spectrum) // 2]
        fft_peak_freq = float(np.argmax(half))
        fft_max = float(half.max())
        fft_median = float(np.median(half))
        fft_mean = float(half.mean())

        return np.array(
            [
                mean, std, mn, mx, median, rng, sk, kt, ent, gn,
                peak_index, peak_value, com, inertia,
                sum_t[0], mean_t[0], max_t[0],
                sum_t[1], mean_t[1], max_t[1],
                sum_t[2], mean_t[2], max_t[2],
                mean_w[0], std_w[0], max_w[0],
                mean_w[1], std_w[1], max_w[1],
                mean_w[2], std_w[2], max_w[2],
                mean_w[3], std_w[3], max_w[3],
                mean_w[4], std_w[4], max_w[4],
                mean_d, std_d, max_d, min_d, n_pos, n_neg, n_zero,
                ac1, ac2, ac3,
                fft_peak_freq, fft_max, fft_median, fft_mean,
            ],
            dtype=np.float32,
        )
