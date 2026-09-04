"""DDM geometry shared by every feature module.

A Rongowai DDM has 40 delay bins (0.125 chip each) and 5 Doppler bins (250 Hz
each). Flattened DDMs are stored in C order, so ``flat.reshape(40, 5)`` gives
back the ``(delay, doppler)`` image.
"""

from __future__ import annotations

import numpy as np

N_DELAY = 40
N_DOPPLER = 5
N_PIXELS = N_DELAY * N_DOPPLER
N_BANDS = 5  # delay bands of 8 bins used by the global features
DELAY_RES_CHIPS = 0.125
DOPPLER_RES_HZ = 250.0
EPS = 1e-10


def as_ddm(x: np.ndarray) -> np.ndarray:
    """Return ``(N, 40, 5)`` float64 from a flat, a batch of flat, or an image batch."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return x.reshape(1, N_DELAY, N_DOPPLER)
    if x.ndim == 2 and x.shape[1] == N_PIXELS:
        return x.reshape(-1, N_DELAY, N_DOPPLER)
    if x.ndim == 3 and x.shape[1:] == (N_DELAY, N_DOPPLER):
        return x
    raise ValueError(f"cannot interpret array of shape {x.shape} as DDMs")


def normalize_sum(ddm: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Scale every DDM so that it sums to one (a small offset keeps logs finite)."""
    x = as_ddm(ddm) + eps
    return x / x.sum(axis=(1, 2), keepdims=True)
