"""Feature groups and the function that computes the DDM-level ones.

Groups computed from a single DDM: ``global``, ``quadrant``, ``peak``,
``coherence`` (all on the sum-normalized image) and ``power`` (raw counts and
noise floor). ``polarimetric`` needs the twin DDM, ``meta`` is instrument and
geometry metadata read from the L1 file, ``raw`` is the normalized pixel vector.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .coherence import COHERENCE_FEATURE_NAMES, coherence_features
from .global_stats import GLOBAL_FEATURE_NAMES, global_features
from .layout import N_PIXELS, normalize_sum
from .peak_region import PEAK_FEATURE_NAMES, peak_region_features
from .polarimetric import POLARIMETRIC_FEATURE_NAMES
from .power import POWER_FEATURE_NAMES, power_features
from .quadrant import QUADRANT_FEATURE_NAMES, quadrant_features

META_FEATURE_NAMES: tuple[str, ...] = (
    "ddm_snr",
    "sp_inc_angle",
    "gain_copol",
    "gain_xpol",
    "slant_range",
    "ac_alt",
)
RAW_FEATURE_NAMES: tuple[str, ...] = tuple(f"px_{i:03d}" for i in range(N_PIXELS))

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "global": GLOBAL_FEATURE_NAMES,
    "quadrant": QUADRANT_FEATURE_NAMES,
    "peak": PEAK_FEATURE_NAMES,
    "coherence": COHERENCE_FEATURE_NAMES,
    "power": POWER_FEATURE_NAMES,
    "polarimetric": POLARIMETRIC_FEATURE_NAMES,
    "meta": META_FEATURE_NAMES,
    "raw": RAW_FEATURE_NAMES,
}

#: Engineered groups that depend on one DDM only (the default model input).
DDM_GROUPS: tuple[str, ...] = ("global", "peak", "coherence", "power")
#: Every engineered group (no raw pixels, no metadata).
ALL_ENGINEERED_GROUPS: tuple[str, ...] = DDM_GROUPS + ("quadrant", "polarimetric")


def feature_names(groups: Iterable[str]) -> list[str]:
    names: list[str] = []
    for g in groups:
        if g not in FEATURE_GROUPS:
            raise KeyError(f"unknown feature group {g!r}; known: {sorted(FEATURE_GROUPS)}")
        names.extend(FEATURE_GROUPS[g])
    return names


def compute_ddm_features(
    raw: np.ndarray, noise_floor: np.ndarray, peak_pixels: int = 40
) -> dict[str, np.ndarray]:
    """Compute the single-DDM groups for raw DDMs ``(N, 40, 5)``; returns ``{group: (N, k) float32}``."""
    x = normalize_sum(raw)
    return {
        "global": global_features(x),
        "quadrant": quadrant_features(x),
        "peak": peak_region_features(x, peak_pixels),
        "coherence": coherence_features(x),
        "power": power_features(raw, noise_floor),
    }
