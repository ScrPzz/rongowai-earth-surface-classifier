"""Absolute power level of the DDM (4 features).

Every other DDM group works on the sum-normalized image and is blind to the
absolute signal level. These features restore it: the total and peak raw counts
and the excess of the peak over the per-DDM noise floor reported in the L1 file.
"""

from __future__ import annotations

import numpy as np

POWER_FEATURE_NAMES: tuple[str, ...] = (
    "raw_sum_log10",
    "raw_peak_log10",
    "peak_excess_db",
    "count_snr_db",
)


def power_features(raw: np.ndarray, noise_floor: np.ndarray) -> np.ndarray:
    """``raw`` is ``(N, 40, 5)`` in counts, ``noise_floor`` is ``(N,)`` in counts."""
    raw = np.asarray(raw, dtype=np.float64)
    nf = np.asarray(noise_floor, dtype=np.float64)
    total = raw.sum((1, 2))
    peak = raw.max((1, 2))
    excess = np.maximum(peak - nf, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.column_stack(
            [
                np.log10(total + 1.0),
                np.log10(peak + 1.0),
                10.0 * np.log10(excess),
                10.0 * np.log10(excess / np.where(nf > 0, nf, np.nan)),
            ]
        )
    return out.astype(np.float32)
