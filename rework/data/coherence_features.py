"""Coherence observables for GNSS-R DDM classification (Phase 1.3).

Implements four physics-derived scalar features that distinguish coherent
specular reflection (water) from incoherent diffuse scattering (land):

1. **PHPR** — peak-to-horseshoe power ratio. Coherent water reflections
   concentrate energy in a small region near the specular point; incoherent
   land scattering spreads energy outward in the characteristic horseshoe.
   Reference: Wang, J., Hu, Y. & Li, Z. (2022). "A New Coherence Detection
   Method for Mapping Inland Water Bodies Using CYGNSS Data." Remote
   Sensing, 14(13), 3195. DOI:10.3390/rs14133195.
2. **LES** — leading-edge slope of the integrated delay waveform. Steep for
   coherent reflectors (sharp rise to specular peak), shallow for diffuse.
3. **TES** — trailing-edge slope. Mirrors LES on the post-peak side.
4. **Von Neumann entropy** — quantum-like entropy of the DDM treated as a
   density-matrix-style operator. Russo, I. M. et al. (2022). "Entropy-
   Based Coherence Metric for Land Applications of GNSS-R." IEEE TGRS 60.

Plus auxiliary features: PHPR in dB, edge-slope ratio, Shannon entropy of
the normalized DDM.

DDM layout convention: 5 Doppler bins × 40 delay bins, flattened to (200,)
in row-major C order (Doppler axis 0, delay axis 1).
"""
from __future__ import annotations

import numpy as np

DDM_DOPPLER_BINS = 5
DDM_DELAY_BINS = 40

COHERENCE_FEATURE_NAMES: tuple[str, ...] = (
    "phpr",
    "phpr_db",
    "les",
    "tes",
    "edge_ratio_les_abs_tes",
    "shannon_entropy",
    "vn_entropy",
)


def reshape_ddm(flat_ddm: np.ndarray) -> np.ndarray:
    """Reshape (200,) → (5, 40) or (N, 200) → (N, 5, 40)."""
    if flat_ddm.ndim == 1:
        return flat_ddm.reshape(DDM_DOPPLER_BINS, DDM_DELAY_BINS)
    if flat_ddm.ndim == 2:
        return flat_ddm.reshape(-1, DDM_DOPPLER_BINS, DDM_DELAY_BINS)
    raise ValueError(f"flat_ddm must be 1-D or 2-D; got shape {flat_ddm.shape}")


def peak_horseshoe_ratio(
    ddm2d: np.ndarray,
    peak_window: tuple[int, int] = (1, 3),
    eps: float = 1e-12,
) -> float:
    """Ratio of peak-region power to off-specular ('horseshoe') power.

    The peak region is a window of ``(2*peak_window[0]+1)`` Doppler ×
    ``(2*peak_window[1]+1)`` delay bins centred at the DDM argmax. The
    horseshoe is everything else. Defaults yield a 3×7 = 21-pixel peak
    region, which empirically captures the bulk of the coherent return on
    Rongowai DDMs while leaving enough off-specular pixels for a meaningful
    denominator.
    """
    ddm2d = np.asarray(ddm2d, dtype=np.float64)
    if ddm2d.ndim != 2:
        raise ValueError(f"ddm2d must be 2-D; got shape {ddm2d.shape}")

    peak_dop, peak_del = np.unravel_index(np.argmax(ddm2d), ddm2d.shape)

    mask = np.zeros_like(ddm2d, dtype=bool)
    d0 = max(0, peak_dop - peak_window[0])
    d1 = min(ddm2d.shape[0], peak_dop + peak_window[0] + 1)
    e0 = max(0, peak_del - peak_window[1])
    e1 = min(ddm2d.shape[1], peak_del + peak_window[1] + 1)
    mask[d0:d1, e0:e1] = True

    peak_power = float(ddm2d[mask].sum())
    horseshoe_power = float(ddm2d[~mask].sum())
    return peak_power / (horseshoe_power + eps)


def leading_trailing_edge_slopes(
    ddm2d: np.ndarray,
    edge_samples: int = 5,
) -> tuple[float, float]:
    """Linear-fit slopes (LES, TES) around the peak of the delay waveform.

    The DDM is integrated over the Doppler axis to obtain a 40-sample delay
    waveform, then linear regressions are fit on the ``edge_samples`` bins
    leading up to and trailing the peak.

    Returns
    -------
    (les, tes)
        Both floats. For a typical specular return, LES > 0 (sharp rise)
        and TES < 0 (gradual decay). Water tends to have steeper magnitudes
        than land.
    """
    ddm2d = np.asarray(ddm2d, dtype=np.float64)
    waveform = ddm2d.sum(axis=0)
    peak_idx = int(np.argmax(waveform))
    n = len(waveform)

    lead_start = max(0, peak_idx - edge_samples)
    if peak_idx - lead_start >= 2:
        x = np.arange(lead_start, peak_idx + 1, dtype=np.float64)
        y = waveform[lead_start:peak_idx + 1]
        les = float(np.polyfit(x, y, 1)[0])
    else:
        les = 0.0

    trail_end = min(n, peak_idx + edge_samples + 1)
    if trail_end - peak_idx >= 2:
        x = np.arange(peak_idx, trail_end, dtype=np.float64)
        y = waveform[peak_idx:trail_end]
        tes = float(np.polyfit(x, y, 1)[0])
    else:
        tes = 0.0

    return les, tes


def shannon_entropy_ddm(ddm2d: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy of the DDM treated as a 200-cell probability distribution."""
    ddm2d = np.asarray(ddm2d, dtype=np.float64)
    total = ddm2d.sum()
    if total <= 0:
        return 0.0
    p = ddm2d.flatten() / total
    p = p[p > eps]
    return float(-(p * np.log(p)).sum())


def von_neumann_entropy_ddm(ddm2d: np.ndarray, eps: float = 1e-12) -> float:
    """Von Neumann entropy of the 5×5 density operator ρ ∝ D D^T.

    Coherent (water) DDMs have rank-1-like structure → one dominant
    eigenvalue → low entropy. Incoherent (land) DDMs spread mass across
    eigenvalues → high entropy. Per Russo et al. (2022).
    """
    ddm2d = np.asarray(ddm2d, dtype=np.float64)
    rho = ddm2d @ ddm2d.T  # (5, 5)
    trace = np.trace(rho)
    if trace <= 0:
        return 0.0
    rho = rho / trace
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > eps]
    if len(eigvals) == 0:
        return 0.0
    return float(-(eigvals * np.log(eigvals)).sum())


def extract_coherence_features(
    flat_ddm: np.ndarray,
    peak_window: tuple[int, int] = (1, 3),
    edge_samples: int = 5,
) -> np.ndarray:
    """Extract all coherence features from a (200,) sample or (N, 200) batch.

    Returns shape ``(len(COHERENCE_FEATURE_NAMES),)`` or ``(N, 7)``.
    """
    if flat_ddm.ndim == 1:
        return _extract_single(reshape_ddm(flat_ddm), peak_window, edge_samples)
    if flat_ddm.ndim == 2:
        n = flat_ddm.shape[0]
        out = np.empty((n, len(COHERENCE_FEATURE_NAMES)), dtype=np.float32)
        for i in range(n):
            out[i] = _extract_single(reshape_ddm(flat_ddm[i]), peak_window, edge_samples)
        return out
    raise ValueError(f"flat_ddm must be 1-D or 2-D; got shape {flat_ddm.shape}")


def _extract_single(
    ddm2d: np.ndarray,
    peak_window: tuple[int, int],
    edge_samples: int,
) -> np.ndarray:
    phpr = peak_horseshoe_ratio(ddm2d, peak_window=peak_window)
    phpr_db = 10.0 * np.log10(phpr + 1e-12)
    les, tes = leading_trailing_edge_slopes(ddm2d, edge_samples=edge_samples)
    edge_ratio = les / (abs(tes) + 1e-12)
    shan = shannon_entropy_ddm(ddm2d)
    vn = von_neumann_entropy_ddm(ddm2d)
    return np.array(
        [phpr, phpr_db, les, tes, edge_ratio, shan, vn], dtype=np.float32
    )


def _self_test() -> None:
    """Run a synthetic sanity check: water-like DDM vs land-like DDM."""
    rng = np.random.default_rng(42)

    # Synthetic "water" DDM: sharp central peak, low diffuse background
    water = np.zeros((5, 40), dtype=np.float64)
    water[2, 20] = 1000.0
    water[2, 19] = 600.0
    water[2, 21] = 500.0
    water[1, 20] = 400.0
    water[3, 20] = 400.0
    water += rng.exponential(5.0, water.shape)

    # Synthetic "land" DDM: broad, spread-out energy
    land = np.zeros((5, 40), dtype=np.float64)
    for i in range(5):
        for j in range(15, 30):
            land[i, j] = 200.0 + rng.exponential(50.0)
    land += rng.exponential(30.0, land.shape)

    w_feats = extract_coherence_features(water.flatten())
    l_feats = extract_coherence_features(land.flatten())

    print("=== coherence_features._self_test ===")
    print(f"  feature names : {COHERENCE_FEATURE_NAMES}")
    print(f"  water features: {w_feats}")
    print(f"  land  features: {l_feats}")

    assert w_feats[0] > l_feats[0], (
        f"water PHPR ({w_feats[0]:.4f}) should exceed land PHPR ({l_feats[0]:.4f})"
    )
    assert w_feats[5] < l_feats[5], (
        f"water Shannon entropy ({w_feats[5]:.4f}) should be lower than land "
        f"({l_feats[5]:.4f})"
    )
    assert w_feats[6] < l_feats[6], (
        f"water VN entropy ({w_feats[6]:.4f}) should be lower than land "
        f"({l_feats[6]:.4f})"
    )
    print("  ✓ water vs land sanity assertions all passed")

    # Batched path matches per-sample path
    batch = np.stack([water.flatten(), land.flatten()])
    batch_feats = extract_coherence_features(batch)
    assert batch_feats.shape == (2, 7)
    np.testing.assert_allclose(batch_feats[0], w_feats, rtol=1e-6)
    np.testing.assert_allclose(batch_feats[1], l_feats, rtol=1e-6)
    print("  ✓ batched extraction matches per-sample output")


if __name__ == "__main__":
    _self_test()
