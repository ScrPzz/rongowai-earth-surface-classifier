"""Global statistics of a normalized DDM (45 features).

The group keeps the families of the original extractor: distribution
statistics of the pixel values, positional moments, delay-band energies, and
derivative / autocorrelation / spectral statistics. The last three families are
computed on the delay waveform (the DDM integrated over Doppler) rather than on
the flattened pixel vector, so that neighbouring samples are neighbouring
delays.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import entropy, kurtosis, skew

from .layout import N_BANDS, N_DELAY, N_DOPPLER

BAND_ROWS = N_DELAY // N_BANDS

GLOBAL_FEATURE_NAMES: tuple[str, ...] = tuple(
    [
        "std",
        "min",
        "max",
        "median",
        "range",
        "skew",
        "kurtosis",
        "entropy",
        "gini",
        "p10",
        "p25",
        "p75",
        "p90",
        "energy",
        "moment3",
        "moment4",
        "peak_delay",
        "peak_doppler",
        "com_delay",
        "com_doppler",
        "inertia_delay",
        "inertia_doppler",
    ]
    + [f"band{i}_{s}" for i in range(N_BANDS) for s in ("sum", "max")]
    + [
        "wf_diff_mean",
        "wf_diff_std",
        "wf_diff_max",
        "wf_diff_min",
        "wf_n_pos_diff",
        "wf_n_neg_diff",
        "wf_autocorr_lag1",
        "wf_autocorr_lag2",
        "wf_autocorr_lag3",
        "wf_fft_peak_freq",
        "wf_fft_max",
        "wf_fft_median",
        "wf_fft_mean",
    ]
)


def gini(flat: np.ndarray) -> np.ndarray:
    """Gini coefficient of every row of a non-negative matrix."""
    a = np.sort(flat, axis=1)
    n = a.shape[1]
    idx = np.arange(1, n + 1)
    return ((2 * idx - n - 1) * a).sum(1) / (n * a.sum(1))


def autocorr(w: np.ndarray, lag: int) -> np.ndarray:
    """Pearson autocorrelation of every row at a given lag (0 where undefined)."""
    a = w[:, :-lag]
    b = w[:, lag:]
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    num = (a * b).sum(1)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


def global_features(x: np.ndarray) -> np.ndarray:
    """``(N, 45)`` float32 features of normalized DDMs ``x`` of shape ``(N, 40, 5)``."""
    n = x.shape[0]
    flat = x.reshape(n, -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        std = flat.std(1)
        mn, mx = flat.min(1), flat.max(1)
        med = np.median(flat, axis=1)
        sk, ku = skew(flat, axis=1), kurtosis(flat, axis=1)
        ent = entropy(flat, axis=1)
        gn = gini(flat)
        p10, p25, p75, p90 = np.percentile(flat, [10, 25, 75, 90], axis=1)
        sq = flat * flat
        energy = sq.sum(1)
        m3 = (sq * flat).mean(1)
        m4 = (sq * sq).mean(1)

        am = flat.argmax(1)
        peak_delay = am // N_DOPPLER
        peak_doppler = am % N_DOPPLER
        w = x.sum(2)  # delay waveform, sums to one
        d = x.sum(1)  # Doppler profile
        rows = np.arange(N_DELAY, dtype=np.float64)
        cols = np.arange(N_DOPPLER, dtype=np.float64)
        com_delay = (w * rows).sum(1)
        com_doppler = (d * cols).sum(1)
        inertia_delay = (w * (rows[None, :] - com_delay[:, None]) ** 2).sum(1)
        inertia_doppler = (d * (cols[None, :] - com_doppler[:, None]) ** 2).sum(1)

        bands = x.reshape(n, N_BANDS, BAND_ROWS, N_DOPPLER)
        band_sum = bands.sum((2, 3))
        band_max = bands.max((2, 3))

        dw = np.diff(w, axis=1)
        spec = np.abs(np.fft.rfft(w, axis=1))[:, 1:]  # drop the DC bin

    cols_out = [
        std,
        mn,
        mx,
        med,
        mx - mn,
        sk,
        ku,
        ent,
        gn,
        p10,
        p25,
        p75,
        p90,
        energy,
        m3,
        m4,
        peak_delay,
        peak_doppler,
        com_delay,
        com_doppler,
        inertia_delay,
        inertia_doppler,
    ]
    for i in range(N_BANDS):
        cols_out += [band_sum[:, i], band_max[:, i]]
    cols_out += [
        dw.mean(1),
        dw.std(1),
        dw.max(1),
        dw.min(1),
        (dw > 0).sum(1),
        (dw < 0).sum(1),
        autocorr(w, 1),
        autocorr(w, 2),
        autocorr(w, 3),
        spec.argmax(1) + 1,
        spec.max(1),
        np.median(spec, axis=1),
        spec.mean(1),
    ]
    out = np.column_stack(cols_out).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
