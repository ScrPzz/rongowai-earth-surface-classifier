"""Peak-region features (49 features).

The region is grown from the brightest pixel by repeatedly adding the brightest
8-connected neighbour until it holds ``n_pixels`` pixels (the "adaptive" method
of the original notebooks). Shape, position, intensity, texture, connectivity,
moment and orientation statistics of that region are then computed, all
vectorized over a batch of DDMs.
"""

from __future__ import annotations

import numpy as np

NEIGHBOURS_8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
NEIGHBOURS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GLCM_LEVELS = 8
HIST_BINS = 10

PEAK_FEATURE_NAMES: tuple[str, ...] = tuple(
    "peak_" + s
    for s in (
        "bbox_h",
        "bbox_w",
        "aspect_ratio",
        "perimeter",
        "compactness",
        "sphericity",
        "rectangularity",
        "centroid_delay",
        "centroid_doppler",
        "rel_pos_delay",
        "rel_pos_doppler",
        "dist_to_center",
        "spread_delay",
        "spread_doppler",
        "spread_total",
        "pdist_mean",
        "pdist_var",
        "pdist_max",
        "int_mean",
        "int_std",
        "int_min",
        "int_max",
        "int_range",
        "int_skew",
        "int_kurtosis",
        "int_entropy",
        "rel_intensity",
        "background_contrast",
        "local_contrast",
        "connectivity_8",
        "cluster_coeff",
        "mu20",
        "mu02",
        "mu11",
        "mu30",
        "mu03",
        "hu1",
        "hu2",
        "hu3",
        "orient_angle",
        "axis_ratio",
        "orient_consistency",
        "glcm_contrast",
        "glcm_dissimilarity",
        "glcm_homogeneity",
        "glcm_energy",
        "glcm_correlation",
        "glcm_asm",
        "energy_fraction",
    )
)


def shift(a: np.ndarray, dr: int, dc: int, fill=0) -> np.ndarray:
    """``out[r, c] = a[r + dr, c + dc]`` with ``fill`` outside the image; ``a`` is ``(N, H, W)``."""
    _, h, w = a.shape
    out = np.full_like(a, fill)
    r0, r1 = max(0, -dr), min(h, h - dr)
    c0, c1 = max(0, -dc), min(w, w - dc)
    if r1 > r0 and c1 > c0:
        out[:, r0:r1, c0:c1] = a[:, r0 + dr : r1 + dr, c0 + dc : c1 + dc]
    return out


def grow_peak_region(img: np.ndarray, n_pixels: int) -> np.ndarray:
    """Boolean ``(N, H, W)`` mask of the ``n_pixels`` brightest 8-connected region around the peak."""
    n, h, w = img.shape
    mask = np.zeros(img.shape, dtype=bool)
    flat_mask = mask.reshape(n, -1)
    ar = np.arange(n)
    flat_mask[ar, img.reshape(n, -1).argmax(1)] = True
    for _ in range(min(n_pixels, h * w) - 1):
        frontier = np.zeros_like(mask)
        for dr, dc in NEIGHBOURS_8:
            frontier |= shift(mask, dr, dc, False)
        frontier &= ~mask
        cand = np.where(frontier, img, -np.inf).reshape(n, -1)
        j = cand.argmax(1)
        ok = np.isfinite(cand[ar, j])
        flat_mask[ar[ok], j[ok]] = True
    return mask


def _region_coordinates(mask: np.ndarray) -> np.ndarray:
    """``(N, K, 2)`` float coordinates of region pixels, NaN-padded to the largest region."""
    n, h, w = mask.shape
    flat = mask.reshape(n, -1)
    k = int(flat.sum(1).max())
    order = np.argsort(~flat, axis=1, kind="stable")[:, :k]
    valid = np.take_along_axis(flat, order, axis=1)
    coords = np.stack([order // w, order % w], axis=-1).astype(np.float64)
    coords[~valid] = np.nan
    return coords


def _pairwise_stats(coords: np.ndarray, chunk: int = 2048) -> tuple[np.ndarray, ...]:
    n, k, _ = coords.shape
    iu = np.triu_indices(k, 1)
    out = np.zeros((n, 4))
    for s in range(0, n, chunk):
        p = coords[s : s + chunk]
        d = np.sqrt(((p[:, :, None, :] - p[:, None, :, :]) ** 2).sum(-1))[:, iu[0], iu[1]]
        with np.errstate(invalid="ignore"):
            out[s : s + chunk, 0] = np.nanmean(d, axis=1)
            out[s : s + chunk, 1] = np.nanvar(d, axis=1)
            out[s : s + chunk, 2] = np.nanmax(d, axis=1)
            out[s : s + chunk, 3] = np.nanmin(d, axis=1)
    return tuple(np.nan_to_num(out[:, i]) for i in range(4))


def _glcm_features(q: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Grey-level co-occurrence statistics over horizontal and vertical in-region pairs."""
    n = q.shape[0]
    levels = GLCM_LEVELS
    base = (np.arange(n) * levels * levels)[:, None, None]
    counts = np.zeros(n * levels * levels, dtype=np.float64)
    for a, b, v in (
        (q[:, :, :-1], q[:, :, 1:], mask[:, :, :-1] & mask[:, :, 1:]),
        (q[:, :-1, :], q[:, 1:, :], mask[:, :-1, :] & mask[:, 1:, :]),
    ):
        idx_ab = (base + a * levels + b)[v]
        idx_ba = (base + b * levels + a)[v]
        counts += np.bincount(np.concatenate([idx_ab, idx_ba]), minlength=counts.size)
    P = counts.reshape(n, levels, levels)
    total = P.sum((1, 2), keepdims=True)
    P = np.where(total > 0, P / np.where(total > 0, total, 1), 0.0)
    i, j = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")
    diff = (i - j).astype(np.float64)[None]
    contrast = (P * diff**2).sum((1, 2))
    dissim = (P * np.abs(diff)).sum((1, 2))
    homog = (P / (1.0 + diff**2)).sum((1, 2))
    asm = (P * P).sum((1, 2))
    energy = np.sqrt(asm)
    mu = (P * i[None]).sum((1, 2))
    var = (P * (i[None] - mu[:, None, None]) ** 2).sum((1, 2))
    cov = (P * (i[None] - mu[:, None, None]) * (j[None] - mu[:, None, None])).sum((1, 2))
    corr = np.where(var > 0, cov / np.where(var > 0, var, 1.0), 0.0)
    return np.column_stack([contrast, dissim, homog, energy, corr, asm])


def peak_region_features(x: np.ndarray, n_pixels: int = 40) -> np.ndarray:
    """``(N, 49)`` float32 features of normalized DDMs ``x`` of shape ``(N, 40, 5)``."""
    n, h, w = x.shape
    mask = grow_peak_region(x, n_pixels)
    area = mask.sum((1, 2)).astype(np.float64)
    R = np.broadcast_to(np.arange(h, dtype=np.float64)[None, :, None], x.shape)
    C = np.broadcast_to(np.arange(w, dtype=np.float64)[None, None, :], x.shape)

    rmin = np.where(mask, R, np.inf).min((1, 2))
    rmax = np.where(mask, R, -np.inf).max((1, 2))
    cmin = np.where(mask, C, np.inf).min((1, 2))
    cmax = np.where(mask, C, -np.inf).max((1, 2))
    bbox_h = rmax - rmin + 1
    bbox_w = cmax - cmin + 1
    aspect = bbox_w / bbox_h
    rect = area / (bbox_h * bbox_w)
    perimeter = np.zeros(n)
    for dr, dc in NEIGHBOURS_4:
        perimeter += (mask & ~shift(mask, dr, dc, False)).sum((1, 2))
    compact = 4 * np.pi * area / perimeter**2
    spher = 2 * np.sqrt(np.pi * area) / perimeter

    wgt = np.where(mask, x, 0.0)
    wsum = wgt.sum((1, 2))
    cr = (wgt * R).sum((1, 2)) / wsum
    cc = (wgt * C).sum((1, 2)) / wsum
    dist_center = np.sqrt((cr - h / 2) ** 2 + (cc - w / 2) ** 2)

    mr = (mask * R).sum((1, 2)) / area
    mc = (mask * C).sum((1, 2)) / area
    dR = np.where(mask, R - mr[:, None, None], 0.0)
    dC = np.where(mask, C - mc[:, None, None], 0.0)
    var_r = (dR**2).sum((1, 2)) / area
    var_c = (dC**2).sum((1, 2)) / area
    spread_r, spread_c = np.sqrt(var_r), np.sqrt(var_c)

    pd_mean, pd_var, pd_max, _ = _pairwise_stats(_region_coordinates(mask))

    vals = np.where(mask, x, np.nan)
    with np.errstate(invalid="ignore"):
        i_mean = np.nanmean(vals, axis=(1, 2))
        i_std = np.nanstd(vals, axis=(1, 2))
        i_min = np.nanmin(vals, axis=(1, 2))
        i_max = np.nanmax(vals, axis=(1, 2))
    dv = np.where(mask, x - i_mean[:, None, None], 0.0)
    m2 = (dv**2).sum((1, 2)) / area
    m3 = (dv**3).sum((1, 2)) / area
    m4 = (dv**4).sum((1, 2)) / area
    safe = np.where(m2 > 0, m2, 1.0)
    i_skew = np.where(m2 > 0, m3 / safe**1.5, 0.0)
    i_kurt = np.where(m2 > 0, m4 / safe**2 - 3.0, 0.0)
    rng = i_max - i_min
    rel = np.where(
        mask, (x - i_min[:, None, None]) / np.where(rng > 0, rng, 1.0)[:, None, None], 0.0
    )
    bins = np.clip(np.floor(rel * HIST_BINS), 0, HIST_BINS - 1).astype(np.int64)
    sample = np.broadcast_to(np.arange(n)[:, None, None], x.shape)
    hist = np.bincount((sample * HIST_BINS + bins)[mask], minlength=n * HIST_BINS).reshape(
        n, HIST_BINS
    )
    ph = hist / area[:, None]
    i_entropy = -(np.where(ph > 0, ph * np.log(np.where(ph > 0, ph, 1.0)), 0.0)).sum(1)

    img_mean = x.mean((1, 2))
    img_std = x.std((1, 2))
    rel_int = i_mean / img_mean
    bg_contrast = np.abs(i_mean - img_mean) / np.where(img_std > 0, img_std, 1.0)
    outside = ~mask
    nb_sum = np.zeros_like(x)
    nb_cnt = np.zeros_like(x)
    nbr_in = np.zeros_like(x)
    for dr, dc in NEIGHBOURS_8:
        nb_sum += shift(np.where(outside, x, 0.0), dr, dc, 0.0)
        nb_cnt += shift(outside.astype(np.float64), dr, dc, 0.0)
        nbr_in += shift(mask.astype(np.float64), dr, dc, 0.0)
    has_nb = mask & (nb_cnt > 0)
    nb_mean = nb_sum / np.where(nb_cnt > 0, nb_cnt, 1.0)
    lc = np.where(has_nb, np.abs(x - nb_mean), 0.0).sum((1, 2))
    lc_cnt = has_nb.sum((1, 2))
    local_contrast = np.where(lc_cnt > 0, lc / np.maximum(lc_cnt, 1), 0.0)
    conn8 = (mask * nbr_in).sum((1, 2)) / area
    pairs = (mask * nbr_in).sum((1, 2)) / 2.0
    max_pairs = area * (area - 1) / 2.0
    cluster = np.where(max_pairs > 0, pairs / np.where(max_pairs > 0, max_pairs, 1.0), 0.0)

    mu20 = (dR**2).sum((1, 2))
    mu02 = (dC**2).sum((1, 2))
    mu11 = (dR * dC).sum((1, 2))
    mu30 = (dR**3).sum((1, 2))
    mu03 = (dC**3).sum((1, 2))
    mu21 = (dR**2 * dC).sum((1, 2))
    mu12 = (dR * dC**2).sum((1, 2))
    a2, a25 = area**2, area**2.5
    eta = lambda m, order: m / area ** (1 + order / 2)  # noqa: E731
    e20, e02, e11 = eta(mu20, 2), eta(mu02, 2), eta(mu11, 2)
    e30, e03, e21, e12 = eta(mu30, 3), eta(mu03, 3), eta(mu21, 3), eta(mu12, 3)
    hu1 = e20 + e02
    hu2 = (e20 - e02) ** 2 + 4 * e11**2
    hu3 = (e30 - 3 * e12) ** 2 + (3 * e21 - e03) ** 2

    srr, scc, src = var_r, var_c, mu11 / area
    tr, det = srr + scc, srr * scc - src**2
    disc = np.sqrt(np.maximum(tr**2 / 4 - det, 0.0))
    lam1, lam2 = tr / 2 + disc, np.maximum(tr / 2 - disc, 0.0)
    vr = np.where(np.abs(src) > 1e-12, src, np.where(srr >= scc, 1.0, 0.0))
    vc = np.where(np.abs(src) > 1e-12, lam1 - srr, np.where(srr >= scc, 0.0, 1.0))
    angle = np.arctan2(vc, vr)
    axis_ratio = np.where(lam2 > 1e-9, lam1 / np.where(lam2 > 1e-9, lam2, 1.0), 1.0)
    axis_ratio = np.minimum(axis_ratio, 1e3)
    consistency = np.where(tr > 0, lam1 / np.where(tr > 0, tr, 1.0), 0.0)

    q = np.clip(np.floor(rel * (GLCM_LEVELS - 1) + 1e-9), 0, GLCM_LEVELS - 1).astype(np.int64)
    glcm = _glcm_features(q, mask)

    out = np.column_stack(
        [
            bbox_h,
            bbox_w,
            aspect,
            perimeter,
            compact,
            spher,
            rect,
            cr,
            cc,
            cr / h,
            cc / w,
            dist_center,
            spread_r,
            spread_c,
            np.sqrt(var_r + var_c),
            pd_mean,
            pd_var,
            pd_max,
            i_mean,
            i_std,
            i_min,
            i_max,
            rng,
            i_skew,
            i_kurt,
            i_entropy,
            rel_int,
            bg_contrast,
            local_contrast,
            conn8,
            cluster,
            mu20 / a2,
            mu02 / a2,
            mu11 / a2,
            mu30 / a25,
            mu03 / a25,
            hu1,
            hu2,
            hu3,
            angle,
            axis_ratio,
            consistency,
            glcm[:, 0],
            glcm[:, 1],
            glcm[:, 2],
            glcm[:, 3],
            glcm[:, 4],
            glcm[:, 5],
            wsum,
        ]
    ).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
