"""Quadrant statistics (16 features): the DDM cut at delay bin 20 and Doppler bin 2.

Kept as a separate group because removing it improved the original model; the
ablation in this repository measures that effect again.
"""

from __future__ import annotations

import numpy as np

from .layout import N_DELAY, N_DOPPLER

QUADRANT_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"q{i}_{s}" for i in range(1, 5) for s in ("mean", "std", "max", "energy")
)


def quadrant_features(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    mr, mc = N_DELAY // 2, N_DOPPLER // 2
    quads = (x[:, :mr, :mc], x[:, :mr, mc:], x[:, mr:, :mc], x[:, mr:, mc:])
    cols = []
    for q in quads:
        flat = q.reshape(n, -1)
        cols += [flat.mean(1), flat.std(1), flat.max(1), (flat * flat).sum(1)]
    return np.column_stack(cols).astype(np.float32)
