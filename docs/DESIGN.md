# Design notes

This document records the requirements and the decisions behind the rework of the
repository. It was written before the code, in place of an interactive
requirements session, so every decision is stated together with the assumption
it rests on. Change the assumption and the decision should be revisited.

## Goal

Present the work done on the Rongowai land/water classifier in a form that a
reader can run, inspect and trust: one package, numbered scripts that reproduce
every number in the README, and results measured under a validation protocol
that cannot leak flight-level structure into the test folds.

The three areas the repository must cover in depth are

1. **feature extraction** from raw Delay-Doppler Maps (DDMs),
2. **dataset handling**: quality filtering, labelling, sampling, splitting, leakage checks,
3. **model selection and training**: screening, tuning, calibration, evaluation.

Everything else (deployment, export formats, experiment tracking servers) is out
of scope for this branch.

## Functional requirements

| ID | Requirement |
|---|---|
| F1 | Read Rongowai L1 NetCDF flights and extract every DDM with its label and instrument metadata. |
| F2 | Apply the quality filters of the original pipeline (antenna gains, slant range, optional SNR) with thresholds in one place. |
| F3 | Produce a binary label from `sp_surface_type` and keep the 8-class surface type for per-class analysis. |
| F4 | Sample a bounded number of DDMs per flight so the whole corpus fits on a workstation disk. |
| F5 | Split by flight, with a geographic and a temporal out-of-distribution holdout, and group folds for cross-validation. |
| F6 | Compute feature groups from the DDM alone (global statistics, peak region, coherence observables), from the polarimetric pair, and from instrument metadata. |
| F7 | Screen a set of model families under the same grouped cross-validation, then tune the best gradient-boosting model and the best neural tabular model. |
| F8 | Evaluate on every holdout with threshold-free and thresholded metrics, per polarization and per surface type; calibrate probabilities. |
| F9 | Regenerate every figure and table in the README from saved artefacts. |

## Non-functional requirements

- Full ETL of the 7,293 usable flights must run in well under an hour on 16 cores.
- Feature extraction must be vectorized over DDMs; no per-sample Python loops in the hot path.
- Every script is idempotent and resumable where it is expensive (the ETL writes per-chunk files).
- No partner or company names anywhere in the branch.
- Tests cover feature invariants, sampling, splitting and leakage checks on synthetic data; they do not need the dataset.

## Decisions and the assumptions behind them

**D1. DDM layout is 40 delay bins × 5 Doppler bins.** Verified on the files
(`raw_counts` dimensions are `sample, ddm, delay, doppler`). The original
notebooks flattened DDMs in C order and reshaped them to `(40, 5)`; the code keeps
that convention everywhere and names the axes.

**D2. Channels 0–9 and 10–19 are the same reflections in the two polarizations.**
Verified: for every valid pair the PRN, the specular point and the surface type
coincide, and `ddm_ant` is 2 (LHCP) for 0–9 and 3 (RHCP) for 10–19. A *reflection*
is therefore a pair of DDMs, and polarimetric features are ratios between the two.

**D3. Label rule.** Water = ocean (−1) or inland water (3); land = every other
surface class. The original pipeline put inland water on the land side. The
physical signature that the classifier learns (coherent specular return) belongs to
any calm water surface, so inland water is grouped with water here; the per-class
breakdown in the evaluation shows how that class behaves. The rule is a single
function and can be changed in one place.

**D4. Both polarizations are samples.** The original pipeline treated every DDM as a
sample. The rework keeps that, records the polarization, and reports metrics per
polarization. The RHCP channel is the weak (cross-polar) one and is expected to be
much harder.

**D5. Quality filters** as in the original pipeline: co-polar and cross-polar antenna
gain ≥ 5 dBi, slant range from the aircraft to the specular point between 2 and
10 km, no NaN in the filter variables, non-zero DDM. The SNR > 0 dB filter is off by
default (as in the final original ETL) because it removes about three quarters of the
samples; SNR is kept as metadata so performance can be reported as a function of it.

**D6. Sampling.** At most 300 reflections (600 DDMs) per flight, drawn at random with
class stratification inside the flight. Per-flight totals before sampling are kept in
the flight table, so the natural class prior is known.

**D7. Splits.** Flights touching the South Island form the geographic holdout; the
last six months form the temporal holdout; flights in both form a third holdout.
The remaining flights are split into five folds with `StratifiedGroupKFold` on the
per-flight dominant class. Flights whose origin or destination is `ZZZZ` are
dropped. File size is never a feature: it is a near-perfect proxy for the route.

**D8. Feature groups** are independent modules with a registry, so an ablation is a
list of group names. The 200 normalized pixels are also stored, which makes the
"raw pixels" and "learned latent" baselines possible without touching NetCDF again.

**D9. Model selection** is done on one feature set (all DDM-derived groups) with the
same grouped folds for every family, on a fixed subsample so that slow families are
affordable. Tuning is then done for XGBoost and TabNet only, with Optuna and the
same folds. The final models are trained on the whole train pool with early stopping
on one held-out fold.

**D10. Metrics.** ROC AUC and average precision as the threshold-free scores;
accuracy, precision, recall, F1 at a threshold chosen on the validation fold; Brier
score and expected calibration error before and after calibration.

## Open questions for the author

- Whether inland water should stay on the water side (D3).
- Whether the RHCP channel should be part of the deployment target or only of the analysis (D4).
- Whether a stricter quality flag (`quality_flags1`, `ddm_snr_flag`) should replace the gain/range filters.
