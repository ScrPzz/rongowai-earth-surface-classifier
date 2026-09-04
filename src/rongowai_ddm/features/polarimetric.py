"""Polarimetric features of a reflection (7 features).

The reflected GPS signal is mostly left-hand circularly polarized; the
right-hand component grows with surface roughness and dielectric changes. Every
reflection is observed in both polarizations by twin channels, so ratios
between the two DDMs are available without any extra data. The features are
defined at the reflection level (LHCP relative to RHCP) and are identical for
the two DDMs of a pair; ``is_lhcp`` tells the model which one it is looking at.
"""

from __future__ import annotations

import numpy as np

POLARIMETRIC_FEATURE_NAMES: tuple[str, ...] = (
    "is_lhcp",
    "pol_peak_ratio_db",
    "pol_excess_ratio_db",
    "pol_snr_diff_db",
    "pol_shape_corr",
    "pol_phpr_diff_db",
    "pol_entropy_diff",
)
PAIR_FEATURE_NAMES = POLARIMETRIC_FEATURE_NAMES[1:]
DB_CLIP = 40.0


def _ratio_db(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 10.0 * np.log10(np.maximum(num, 1.0) / np.maximum(den, 1.0))
    return np.clip(out, -DB_CLIP, DB_CLIP)


def pair_features(
    raw_l: np.ndarray,
    raw_r: np.ndarray,
    nf_l: np.ndarray,
    nf_r: np.ndarray,
    snr_l: np.ndarray,
    snr_r: np.ndarray,
    phpr_db_l: np.ndarray,
    phpr_db_r: np.ndarray,
    entropy_l: np.ndarray,
    entropy_r: np.ndarray,
) -> np.ndarray:
    """``(N, 6)`` reflection-level features from the LHCP and RHCP raw DDMs ``(N, 40, 5)``."""
    raw_l = np.asarray(raw_l, dtype=np.float64)
    raw_r = np.asarray(raw_r, dtype=np.float64)
    nf_l = np.nan_to_num(np.asarray(nf_l, dtype=np.float64), nan=0.0)
    nf_r = np.nan_to_num(np.asarray(nf_r, dtype=np.float64), nan=0.0)
    peak_ratio = _ratio_db(raw_l.max((1, 2)) - nf_l, raw_r.max((1, 2)) - nf_r)
    exc_l = np.maximum(raw_l - nf_l[:, None, None], 0).sum((1, 2))
    exc_r = np.maximum(raw_r - nf_r[:, None, None], 0).sum((1, 2))
    excess_ratio = _ratio_db(exc_l, exc_r)
    snr_diff = np.asarray(snr_l, dtype=np.float64) - np.asarray(snr_r, dtype=np.float64)
    n = raw_l.shape[0]
    a = raw_l.reshape(n, -1)
    b = raw_r.reshape(n, -1)
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    corr = np.where(den > 0, (a * b).sum(1) / np.where(den > 0, den, 1.0), 0.0)
    out = np.column_stack(
        [
            peak_ratio,
            excess_ratio,
            snr_diff,
            corr,
            np.asarray(phpr_db_l) - np.asarray(phpr_db_r),
            np.asarray(entropy_r) - np.asarray(entropy_l),
        ]
    ).astype(np.float32)
    return out
