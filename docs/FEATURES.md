# Feature catalogue

All groups except `power`, `meta` and the raw pixels are computed on the
**sum-normalized** DDM, so they describe the shape of the map and not its
level. The DDM is a `40 × 5` image (delay × Doppler, 0.125 chip × 250 Hz per
pixel); the *delay waveform* is the map integrated over Doppler and the
*Doppler profile* is the map integrated over delay.

| group | count | module | what it captures |
|---|---:|---|---|
| `global` | 45 | `features/global_stats.py` | distribution of pixel values, 2-D position and spread of the energy, delay-band energies, derivative / autocorrelation / spectrum of the delay waveform |
| `quadrant` | 16 | `features/quadrant.py` | mean, std, max and energy of the four quadrants (delay bin 20, Doppler bin 2) |
| `peak` | 49 | `features/peak_region.py` | shape, position, intensity, texture, connectivity, moments and orientation of the 40-pixel region grown from the peak |
| `coherence` | 16 | `features/coherence.py` | coherence observables: peak-to-horseshoe ratio, edge slopes, entropies, widths, deviation from the ideal ambiguity function |
| `power` | 4 | `features/power.py` | absolute level: total and peak counts, peak excess over the noise floor, SNR from counts |
| `polarimetric` | 7 | `features/polarimetric.py` | LHCP/RHCP ratios of peak and excess power, SNR difference, shape correlation, PHPR and entropy differences, polarization flag |
| `meta` | 6 | L1 variables | `ddm_snr`, incidence angle, antenna gains, slant range, aircraft altitude |
| `raw` | 200 | `px_000` … `px_199` | the normalized pixels themselves |

## `global`

| feature | definition |
|---|---|
| `std`, `min`, `max`, `median`, `range` | statistics of the 200 normalized pixels |
| `skew`, `kurtosis` | third and fourth standardized moments (Fisher kurtosis) |
| `entropy` | Shannon entropy of the pixel distribution |
| `gini` | Gini coefficient of the pixel values (0 flat, 1 all energy in one pixel) |
| `p10`, `p25`, `p75`, `p90` | percentiles of the pixel values |
| `energy`, `moment3`, `moment4` | sum of squares, mean cube and mean fourth power |
| `peak_delay`, `peak_doppler` | position of the brightest pixel |
| `com_delay`, `com_doppler` | energy-weighted centre of mass along each axis |
| `inertia_delay`, `inertia_doppler` | energy-weighted second moment about the centre of mass |
| `band{0..4}_sum`, `band{0..4}_max` | energy and maximum in five delay bands of 8 bins |
| `wf_diff_mean/std/max/min`, `wf_n_pos_diff`, `wf_n_neg_diff` | first differences of the delay waveform |
| `wf_autocorr_lag1..3` | autocorrelation of the delay waveform |
| `wf_fft_peak_freq`, `wf_fft_max/median/mean` | amplitude spectrum of the delay waveform (DC bin excluded) |

## `peak`

The region is grown from the brightest pixel: at every step the brightest
8-connected neighbour of the current region is added, until 40 pixels are in.
The procedure is vectorized over a batch of DDMs (`grow_peak_region`).

| family | features |
|---|---|
| geometry | `bbox_h`, `bbox_w`, `aspect_ratio`, `perimeter` (exposed 4-neighbour edges), `compactness` (4πA/P²), `sphericity`, `rectangularity` (A / bounding-box area) |
| position | intensity-weighted `centroid_delay`, `centroid_doppler`, `rel_pos_*`, `dist_to_center` |
| spread | `spread_delay`, `spread_doppler`, `spread_total` (standard deviation of the pixel coordinates), `pdist_mean/var/max` (pairwise distances between region pixels) |
| intensity | `int_mean/std/min/max/range/skew/kurtosis`, `int_entropy` (10-bin histogram) |
| context | `rel_intensity` (region mean / image mean), `background_contrast`, `local_contrast` (mean absolute difference to the non-region neighbours) |
| connectivity | `connectivity_8` (mean number of region neighbours), `cluster_coeff` (adjacent pairs / all pairs) |
| moments | binary central moments `mu20`, `mu02`, `mu11`, `mu30`, `mu03` (area-normalized) and Hu invariants `hu1..3` |
| orientation | `orient_angle`, `axis_ratio`, `orient_consistency` from the covariance of the pixel coordinates |
| texture | grey-level co-occurrence on 8 quantized levels over horizontal and vertical in-region pairs: `glcm_contrast`, `glcm_dissimilarity`, `glcm_homogeneity`, `glcm_energy`, `glcm_correlation`, `glcm_asm` |
| energy | `energy_fraction`: share of the DDM energy inside the region |

## `coherence`

| feature | definition |
|---|---|
| `phpr`, `phpr_db` | energy in a 7 × 3 window around the peak over the energy outside it (peak-to-horseshoe power ratio) |
| `les`, `tes` | least-squares slopes of the peak-normalized delay waveform over the 5 bins before and after the peak |
| `edge_asymmetry` | (|LES| − |TES|) / (|LES| + |TES|) |
| `shannon_entropy` | entropy of the normalized DDM |
| `vn_entropy` | von Neumann entropy of the 5 × 5 Doppler density matrix DᵀD / tr(DᵀD) |
| `delay_width_hm`, `doppler_width_hm` | number of bins above half maximum of the delay waveform and of the Doppler profile |
| `peak_to_mean`, `peak_z` | peak over mean pixel value; peak minus median in units of the standard deviation |
| `wf_asymmetry` | waveform energy after the peak over the energy before it |
| `delay_spread`, `doppler_spread` | energy-weighted standard deviation along each axis |
| `wf_rmsd_ambiguity` | RMS deviation between the peak-normalized waveform and the squared triangular ambiguity function of the C/A code centred on the peak |
| `wf_trailing_ratio` | waveform value half a chip after the peak, relative to the peak |

## `polarimetric`

| feature | definition |
|---|---|
| `is_lhcp` | 1 for the LHCP DDM of the pair, 0 for the RHCP twin |
| `pol_peak_ratio_db` | 10 log10 of (peak − noise floor) LHCP over RHCP, clipped to ±40 dB |
| `pol_excess_ratio_db` | same ratio for the total power above the noise floor |
| `pol_snr_diff_db` | L1 SNR of the LHCP DDM minus that of the RHCP DDM |
| `pol_shape_corr` | Pearson correlation of the two pixel vectors |
| `pol_phpr_diff_db` | PHPR (dB) of LHCP minus RHCP |
| `pol_entropy_diff` | Shannon entropy of RHCP minus LHCP |

## What is deliberately not a feature

Latitude, longitude, distance to the coast, the flight route, the file size and
the L1 `coherence_state` are stored with every sample for analysis, but none of
them is ever passed to a model: the first four encode the label or the holdout
geography, the last is used as a rule-based baseline.
