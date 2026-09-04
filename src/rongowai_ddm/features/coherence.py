"""Coherence observables (16 features).

A calm water surface reflects the GPS signal coherently: the DDM has a sharp
peak, a narrow delay waveform whose shape follows the code auto-correlation
function, and little energy off the specular point. Rough land scatters
incoherently and spreads the energy along delay and Doppler. The features here
measure that difference directly, following the coherence detectors of the
GNSS-R literature (peak-to-horseshoe power ratio, leading/trailing edge
slopes, Shannon and von Neumann entropies, waveform widths and the deviation
from the ideal ambiguity function).
"""

from __future__ import annotations

import numpy as np

from .layout import DELAY_RES_CHIPS, EPS, N_DELAY, N_DOPPLER

COHERENCE_FEATURE_NAMES: tuple[str, ...] = (
    "phpr",
    "phpr_db",
    "les",
    "tes",
    "edge_asymmetry",
    "shannon_entropy",
    "vn_entropy",
    "delay_width_hm",
    "doppler_width_hm",
    "peak_to_mean",
    "peak_z",
    "wf_asymmetry",
    "delay_spread",
    "doppler_spread",
    "wf_rmsd_ambiguity",
    "wf_trailing_ratio",
)

PEAK_WINDOW_DELAY = 3  # +-3 delay bins
PEAK_WINDOW_DOPPLER = 1  # +-1 Doppler bin
EDGE_BINS = 5


def edge_slope(wn: np.ndarray, peak: np.ndarray, side: int, k: int = EDGE_BINS) -> np.ndarray:
    """Least-squares slope of ``wn`` over ``k`` bins before (``side=-1``) or after (``+1``) the peak."""
    n, length = wn.shape
    offsets = np.arange(k + 1)
    idx = peak[:, None] + (offsets[None, :] - k if side < 0 else offsets[None, :])
    valid = (idx >= 0) & (idx < length)
    y = np.take_along_axis(wn, np.clip(idx, 0, length - 1), axis=1)
    cnt = valid.sum(1)
    xf = idx.astype(np.float64)
    xm = np.where(valid, xf, 0).sum(1) / np.maximum(cnt, 1)
    ym = np.where(valid, y, 0).sum(1) / np.maximum(cnt, 1)
    dx = np.where(valid, xf - xm[:, None], 0)
    dy = np.where(valid, y - ym[:, None], 0)
    den = (dx * dx).sum(1)
    slope = np.where(den > 0, (dx * dy).sum(1) / np.where(den > 0, den, 1), 0.0)
    return np.where(cnt >= 2, slope, 0.0)


def coherence_features(x: np.ndarray) -> np.ndarray:
    """``(N, 16)`` float32 features of normalized DDMs ``x`` of shape ``(N, 40, 5)``."""
    n = x.shape[0]
    ar = np.arange(n)
    flat = x.reshape(n, -1)
    w = x.sum(2)
    d = x.sum(1)
    pk = flat.argmax(1)
    pr, pc = pk // N_DOPPLER, pk % N_DOPPLER
    p = w.argmax(1)

    rows = np.arange(N_DELAY)[None, :, None]
    cols = np.arange(N_DOPPLER)[None, None, :]
    win = (np.abs(rows - pr[:, None, None]) <= PEAK_WINDOW_DELAY) & (
        np.abs(cols - pc[:, None, None]) <= PEAK_WINDOW_DOPPLER
    )
    peak_power = (x * win).sum((1, 2))
    phpr = peak_power / np.maximum(1.0 - peak_power, EPS)
    phpr_db = 10.0 * np.log10(phpr + EPS)

    wn = w / w.max(1, keepdims=True)
    les = edge_slope(wn, p, -1)
    tes = edge_slope(wn, p, +1)
    edge_asym = (np.abs(les) - np.abs(tes)) / (np.abs(les) + np.abs(tes) + EPS)

    shannon = -(flat * np.log(flat)).sum(1)
    rho = np.einsum("nrc,nrd->ncd", x, x)
    rho /= np.trace(rho, axis1=1, axis2=2)[:, None, None]
    ev = np.clip(np.linalg.eigvalsh(rho), EPS, None)
    vn = -(ev * np.log(ev)).sum(1)

    delay_width = (wn >= 0.5).sum(1)
    dn = d / d.max(1, keepdims=True)
    doppler_width = (dn >= 0.5).sum(1)
    peak_to_mean = flat.max(1) / flat.mean(1)
    std = flat.std(1)
    peak_z = (flat.max(1) - np.median(flat, axis=1)) / np.where(std > 0, std, 1.0)

    cs = np.cumsum(w, axis=1)
    before = cs[ar, p] - w[ar, p]
    after = 1.0 - cs[ar, p]
    asym = after / (before + EPS)

    r = np.arange(N_DELAY, dtype=np.float64)
    c = np.arange(N_DOPPLER, dtype=np.float64)
    com_r = (w * r).sum(1)
    com_c = (d * c).sum(1)
    delay_spread = np.sqrt((w * (r[None, :] - com_r[:, None]) ** 2).sum(1))
    doppler_spread = np.sqrt((d * (c[None, :] - com_c[:, None]) ** 2).sum(1))

    tri = np.clip(1.0 - np.abs(r[None, :] - p[:, None]) * DELAY_RES_CHIPS, 0.0, None) ** 2
    rmsd = np.sqrt(((wn - tri) ** 2).mean(1))
    trailing = wn[ar, np.clip(p + 4, 0, N_DELAY - 1)]

    out = np.column_stack(
        [
            phpr,
            phpr_db,
            les,
            tes,
            edge_asym,
            shannon,
            vn,
            delay_width,
            doppler_width,
            peak_to_mean,
            peak_z,
            asym,
            delay_spread,
            doppler_spread,
            rmsd,
            trailing,
        ]
    ).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
