"""Polarimetric features from LHCP / RHCP DDM channels (Phase 1.1).

The Rongowai receiver is dual-polarised: each acquisition produces both a
**LHCP** (left-hand circular polarisation, dominant for coherent water
reflections) and an **RHCP** (right-hand circular polarisation, retained for
incoherent land scattering) channel. Production pipeline currently uses only
one channel implicitly; adding polarimetric features that contrast LHCP and
RHCP is **the single highest-evidence physics-informed improvement** for the
land/water discrimination task.

Reference: Bai, D., Ruf, C. S. & Moller, D. (2025). "Calibration of the
Polarimetric GNSS-R Sensor in the Rongowai Mission." IEEE TGRS 63, 1–18.
DOI:10.1109/TGRS.2025.3558131. ([arXiv:2501.10334](https://arxiv.org/abs/2501.10334))

---

**STATUS: SKELETON.** The NetCDF channel layout for Rongowai is not yet
verified at the byte level in this codebase. The ``raw_counts`` variable is
4-D ``(n_time, n_samples, 5, 40)``; the existing pipeline collapses
``n_samples`` (which contains both polarisations across multiple PRNs)
implicitly. Before this module can run for real, the following must be
checked against an actual NetCDF file:

1. Is ``n_samples`` ordered as ``[LHCP_PRN0, …, LHCP_PRN9, RHCP_PRN0, …,
   RHCP_PRN9]`` (block layout) or ``[LHCP_PRN0, RHCP_PRN0, LHCP_PRN1, …]``
   (interleaved)?
2. Is there an auxiliary variable indicating polarisation per sample (e.g.
   ``ddm_ant`` or ``polarisation_type``)?
3. Does the existing ``ddm_snr`` / ``sp_rx_gain_*`` filtering apply to one
   polarisation or both?

A short inspection script lives in ``rework/experiments/inspect_polarimetric_layout.py``
(scheduled for the next session — read 1 NetCDF, dump variable metadata,
identify the LHCP/RHCP split, return the answer).

Once verified, the functions below will be wired to extract per-epoch:

* ``phpr_lhcp``, ``phpr_rhcp`` — peak/horseshoe ratio in each polarisation
* ``polarisation_ratio_db`` = 10·log10(P_LHCP / P_RHCP) on the peak region
* ``polarisation_purity`` = (P_LHCP − P_RHCP) / (P_LHCP + P_RHCP)
* ``polarisation_ratio_map`` — full 5×40 ratio image (downstream feature
  extractors can run on this exactly as on a single-channel DDM)

The latter feeds back into :func:`rework.data.coherence_features.extract_coherence_features`
and :class:`rework.data.feature_extractor.DDMFeatureExtractor`, so the
polarimetric story compounds with everything else in Phase 1.
"""
from __future__ import annotations

import numpy as np

POLARIMETRIC_SCALAR_FEATURE_NAMES: tuple[str, ...] = (
    "polarisation_ratio_db",
    "polarisation_purity",
    "phpr_lhcp",
    "phpr_rhcp",
)


def polarisation_ratio_db(
    p_lhcp: np.ndarray,
    p_rhcp: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """10·log10(P_LHCP / P_RHCP), elementwise. Both inputs should be ≥ 0."""
    return 10.0 * np.log10((p_lhcp + eps) / (p_rhcp + eps))


def polarisation_purity(
    p_lhcp: np.ndarray,
    p_rhcp: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """(P_LHCP − P_RHCP) / (P_LHCP + P_RHCP). Range [−1, +1]."""
    return (p_lhcp - p_rhcp) / (p_lhcp + p_rhcp + eps)


def polarisation_ratio_map(
    ddm_lhcp: np.ndarray,
    ddm_rhcp: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Per-bin LHCP/RHCP ratio map, same shape as inputs."""
    if ddm_lhcp.shape != ddm_rhcp.shape:
        raise ValueError(
            f"DDM shape mismatch: LHCP {ddm_lhcp.shape} vs RHCP {ddm_rhcp.shape}"
        )
    return (np.asarray(ddm_lhcp, dtype=np.float64) + eps) / (
        np.asarray(ddm_rhcp, dtype=np.float64) + eps
    )


def extract_polarimetric_scalars(
    ddm_lhcp: np.ndarray,
    ddm_rhcp: np.ndarray,
    peak_window: tuple[int, int] = (1, 3),
) -> np.ndarray:
    """Extract the four scalar polarimetric features for one (LHCP, RHCP) pair.

    Both inputs are 2-D (5, 40) DDMs. The returned vector has length
    ``len(POLARIMETRIC_SCALAR_FEATURE_NAMES) == 4`` and is ordered to match
    that constant.
    """
    from .coherence_features import peak_horseshoe_ratio  # local import

    if ddm_lhcp.shape != (5, 40) or ddm_rhcp.shape != (5, 40):
        raise ValueError(
            f"Both DDMs must be (5, 40); got LHCP {ddm_lhcp.shape}, RHCP {ddm_rhcp.shape}"
        )

    # Use the LHCP peak location as the canonical peak (it's what water privileges)
    peak_dop, peak_del = np.unravel_index(np.argmax(ddm_lhcp), ddm_lhcp.shape)
    d0 = max(0, peak_dop - peak_window[0])
    d1 = min(5, peak_dop + peak_window[0] + 1)
    e0 = max(0, peak_del - peak_window[1])
    e1 = min(40, peak_del + peak_window[1] + 1)

    p_lhcp_peak = float(ddm_lhcp[d0:d1, e0:e1].sum())
    p_rhcp_peak = float(ddm_rhcp[d0:d1, e0:e1].sum())

    return np.array(
        [
            float(polarisation_ratio_db(p_lhcp_peak, p_rhcp_peak)),
            float(polarisation_purity(p_lhcp_peak, p_rhcp_peak)),
            peak_horseshoe_ratio(ddm_lhcp, peak_window=peak_window),
            peak_horseshoe_ratio(ddm_rhcp, peak_window=peak_window),
        ],
        dtype=np.float32,
    )


def _self_test() -> None:
    """Synthetic check: water should show high pol-ratio + purity vs land."""
    rng = np.random.default_rng(42)

    # Water: LHCP has a strong peak, RHCP is near-noise (Fresnel sign flip)
    water_lhcp = np.zeros((5, 40))
    water_lhcp[2, 20] = 1000.0
    water_lhcp += rng.exponential(5.0, water_lhcp.shape)
    water_rhcp = rng.exponential(20.0, water_lhcp.shape)  # background only

    # Land: both LHCP and RHCP have similar diffuse intensity (depolarising scatter)
    land_lhcp = np.zeros((5, 40))
    land_rhcp = np.zeros_like(land_lhcp)
    for i in range(5):
        for j in range(15, 30):
            land_lhcp[i, j] = 200.0 + rng.exponential(50.0)
            land_rhcp[i, j] = 180.0 + rng.exponential(50.0)
    land_lhcp += rng.exponential(30.0, land_lhcp.shape)
    land_rhcp += rng.exponential(30.0, land_rhcp.shape)

    water_feats = extract_polarimetric_scalars(water_lhcp, water_rhcp)
    land_feats = extract_polarimetric_scalars(land_lhcp, land_rhcp)

    print("=== polarimetric._self_test ===")
    print(f"  feature names: {POLARIMETRIC_SCALAR_FEATURE_NAMES}")
    print(f"  water feats  : {water_feats}")
    print(f"  land  feats  : {land_feats}")

    # Water should have much higher pol_ratio_db (LHCP dominates over RHCP)
    assert water_feats[0] > land_feats[0] + 5.0, (
        f"water pol_ratio_db {water_feats[0]:.2f} should be ≫ land {land_feats[0]:.2f}"
    )
    # Water should have higher purity (closer to +1)
    assert water_feats[1] > land_feats[1] + 0.3, (
        f"water purity {water_feats[1]:.2f} should be ≫ land {land_feats[1]:.2f}"
    )
    print("  ✓ water vs land polarimetric separation OK on synthetic data")
    print(
        "  ⚠ this module is a SKELETON — real LHCP/RHCP extraction from NetCDF "
        "still needs the channel-layout inspection (see module docstring)."
    )


if __name__ == "__main__":
    _self_test()
