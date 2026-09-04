import numpy as np
import pytest

from rongowai_ddm.features.layout import N_DELAY, N_DOPPLER


def synthetic_ddm(
    kind: str, rng: np.random.Generator, peak_row: int = 20, peak_col: int = 2
) -> np.ndarray:
    """A raw-count-like DDM: Gaussian-ish peak on a noisy floor; 'water' is sharp, 'land' is spread."""
    r = np.arange(N_DELAY)[:, None]
    c = np.arange(N_DOPPLER)[None, :]
    floor = 1.0e7 + rng.normal(0, 2.0e5, size=(N_DELAY, N_DOPPLER))
    if kind == "water":
        peak = 4.0e8 * np.exp(
            -((r - peak_row) ** 2) / (2 * 1.2**2) - ((c - peak_col) ** 2) / (2 * 0.6**2)
        )
    else:
        tail = np.clip(r - peak_row, 0, None)
        peak = 6.0e7 * np.exp(
            -((r - peak_row) ** 2) / (2 * 5.0**2)
            - tail / 12.0
            - ((c - peak_col) ** 2) / (2 * 1.5**2)
        )
    return np.maximum(floor + peak, 1.0)


@pytest.fixture
def rng():
    return np.random.default_rng(123)


@pytest.fixture
def batch(rng):
    kinds = ["water", "land"] * 8
    ddms = np.stack(
        [
            synthetic_ddm(
                k, rng, peak_row=int(rng.integers(8, 32)), peak_col=int(rng.integers(0, 5))
            )
            for k in kinds
        ]
    )
    labels = np.array([0 if k == "water" else 1 for k in kinds])
    return ddms, labels
