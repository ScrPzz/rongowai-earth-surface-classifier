import numpy as np
import pytest
from conftest import synthetic_ddm

from rongowai_ddm.features.coherence import COHERENCE_FEATURE_NAMES, coherence_features
from rongowai_ddm.features.layout import as_ddm, normalize_sum
from rongowai_ddm.features.peak_region import NEIGHBOURS_8, grow_peak_region
from rongowai_ddm.features.polarimetric import PAIR_FEATURE_NAMES, pair_features
from rongowai_ddm.features.registry import FEATURE_GROUPS, compute_ddm_features, feature_names


def test_layout_roundtrip():
    x = np.arange(200.0)
    ddm = as_ddm(x)
    assert ddm.shape == (1, 40, 5)
    assert ddm[0, 3, 4] == x[3 * 5 + 4]
    nx = normalize_sum(np.stack([x, x]))
    assert np.allclose(nx.sum((1, 2)), 1.0)


def test_group_shapes_and_finite(batch):
    ddms, _ = batch
    feats = compute_ddm_features(ddms, np.full(len(ddms), 1e7))
    for g, arr in feats.items():
        assert arr.shape == (len(ddms), len(FEATURE_GROUPS[g])), g
        assert np.isfinite(arr).all(), g
        assert arr.dtype == np.float32
    assert len(feature_names(["global", "peak"])) == len(FEATURE_GROUPS["global"]) + len(
        FEATURE_GROUPS["peak"]
    )
    with pytest.raises(KeyError):
        feature_names(["nope"])


def test_batch_matches_single(batch):
    """Vectorisation must not couple samples: batch features equal per-sample features."""
    ddms, _ = batch
    nf = np.full(len(ddms), 1e7)
    full = compute_ddm_features(ddms, nf)
    for i in range(len(ddms)):
        single = compute_ddm_features(ddms[i : i + 1], nf[i : i + 1])
        for g in full:
            np.testing.assert_allclose(full[g][i], single[g][0], rtol=1e-4, atol=1e-5, err_msg=g)


def test_water_land_ordering(batch):
    ddms, labels = batch
    feats = compute_ddm_features(ddms, np.full(len(ddms), 1e7))
    coh = feats["coherence"]
    names = list(COHERENCE_FEATURE_NAMES)
    water, land = labels == 0, labels == 1
    assert coh[water, names.index("phpr")].mean() > coh[land, names.index("phpr")].mean()
    assert (
        coh[water, names.index("shannon_entropy")].mean()
        < coh[land, names.index("shannon_entropy")].mean()
    )
    assert (
        coh[water, names.index("delay_width_hm")].mean()
        < coh[land, names.index("delay_width_hm")].mean()
    )
    asym = coh[:, names.index("edge_asymmetry")]
    assert np.all(asym >= -1) and np.all(asym <= 1)


def _is_connected(mask: np.ndarray) -> bool:
    rr, cc = np.where(mask)
    start = (rr[0], cc[0])
    seen, stack = {start}, [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBOURS_8:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < mask.shape[0]
                and 0 <= nc < mask.shape[1]
                and mask[nr, nc]
                and (nr, nc) not in seen
            ):
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def test_peak_region_is_connected_and_contains_peak(batch):
    ddms, _ = batch
    x = normalize_sum(ddms)
    mask = grow_peak_region(x, 40)
    assert mask.shape == x.shape
    assert (mask.sum((1, 2)) == 40).all()
    for i in range(len(x)):
        r, c = np.unravel_index(x[i].argmax(), x[i].shape)
        assert mask[i, r, c]
        assert _is_connected(mask[i])


def test_peak_region_small_budget(rng):
    x = normalize_sum(synthetic_ddm("water", rng)[None])
    mask = grow_peak_region(x, 1)
    assert mask.sum() == 1


def test_polarimetric_sign(rng):
    strong = synthetic_ddm("water", rng)[None]
    weak = np.maximum(strong / 8.0, 1.0)
    nf = np.full(1, 1e7 / 8.0)
    out = pair_features(
        strong,
        weak,
        np.full(1, 1e7),
        nf,
        np.array([5.0]),
        np.array([-3.0]),
        np.array([0.0]),
        np.array([-2.0]),
        np.array([5.0]),
        np.array([5.2]),
    )
    names = list(PAIR_FEATURE_NAMES)
    assert out.shape == (1, len(names))
    assert out[0, names.index("pol_peak_ratio_db")] > 0
    assert out[0, names.index("pol_snr_diff_db")] == pytest.approx(8.0)
    assert -1 <= out[0, names.index("pol_shape_corr")] <= 1


def test_coherence_handles_flat_ddm():
    flat = np.ones((2, 40, 5))
    out = coherence_features(normalize_sum(flat))
    assert np.isfinite(out).all()
