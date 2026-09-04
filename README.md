# Land or water? Classifying Rongowai GNSS-R Delay-Doppler Maps

A reproducible pipeline that turns the raw Delay-Doppler Maps (DDMs) of the
[Rongowai](https://doi.org/10.5067/RGOWA-S1A10) airborne GNSS-reflectometry
mission into a binary **water / land** classifier, and measures it in a way
that cannot leak flight-level structure into the test set.

The repository covers three things in depth:

1. **Feature extraction** from a 40 × 5 DDM: 137 hand-crafted features in six
   groups (global statistics, peak-region shape, coherence observables, absolute
   power, polarimetric ratios between the twin LHCP/RHCP channels, quadrant
   statistics), all vectorized, plus the raw pixels and a learned latent as
   baselines.
2. **Dataset handling**: quality filtering, labelling, bounded per-flight
   sampling of a 1.1 TiB corpus into a 4.4 M-row table, flight-level splits with
   a geographic and a temporal out-of-distribution holdout, leakage checks and
   adversarial validation.
3. **Model selection and training**: screening of eight model families under the
   same flight-grouped folds, Optuna tuning of XGBoost and TabNet, calibration,
   and an evaluation broken down by polarization, surface class, SNR and
   distance to the coast.

<!-- RESULTS_SUMMARY -->

## The data

Rongowai flies a GNSS-R receiver on an Air New Zealand Q300 that criss-crosses
New Zealand on scheduled routes. Every second, the receiver tracks up to ten
GPS reflections and produces, for each of them, a DDM in the two circular
polarizations: the reflected signal is mostly left-hand polarized (LHCP), the
right-hand component (RHCP) grows with surface roughness. Each L1 file is one
flight; the corpus used here is the full V1.0 release:

| | |
|---|---|
| flights (files) | 7,324, of which 7,293 with a known origin and destination |
| period | 2022-10-26 → 2026-05-23 |
| volume | 1.1 TiB of NetCDF |
| DDM | 40 delay bins × 5 Doppler bins (0.125 chip × 250 Hz), 1 s coherent integration |
| channels | 20 per epoch: channels 0–9 (LHCP) and 10–19 (RHCP) are the same ten reflections |
| label source | `sp_surface_type` from the LINZ land-cover database at the specular point |

The label is **water** for ocean and inland water, **land** for artificial,
barely vegetated, crop, grass, shrub and forest. The eight-class code is kept
with every sample, so the per-class behaviour is part of the evaluation.

## Pipeline

```
NetCDF flight ─► quality filters ─► label ─► per-flight stratified sample (≤ 300 reflections)
              ─► feature groups for the LHCP and the RHCP DDM + pair features ─► one parquet row per DDM
flights.parquet ─► geographic / temporal holdouts ─► 5 flight-grouped folds on the rest
```

| step | script | output |
|---|---|---|
| 0 | `scripts/00_build_flight_index.py` | `results/flights.parquet`: one row per flight with its split |
| 1 | `scripts/01_etl.py` | `data/samples/part-*.parquet` (one row per DDM), fold ids, `results/etl_report.json` |
| 2 | `scripts/02_adversarial_validation.py` | how separable train and holdouts are, at flight and at sample level |
| 3 | `scripts/03_model_selection.py` | screening of the model families on the same folds |
| 4 | `scripts/04_feature_ablation.py` | which feature groups matter, in and out of distribution |
| 5 | `scripts/05_tune_xgboost.py` | Optuna search for XGBoost |
| 6 | `scripts/06_tune_tabnet.py` | Optuna search for TabNet |
| 7 | `scripts/07_train_final.py` | final models, holdout metrics, calibration, breakdowns, importances |
| 8 | `scripts/08_make_figures.py` | every figure and table in this README |
| 9 | `scripts/09_train_autoencoder.py` | optional learned latent for the ablation |

The full ETL of 7,293 flights takes about 18 minutes on 16 cores (0.15 s per
flight, feature extraction included). Every script reads `configs/default.yaml`
and writes under `results/`; the ETL is resumable chunk by chunk.

### Quality filters and sampling

A DDM is kept when the co-polar and cross-polar antenna gains at the specular
point are at least 5 dBi, the slant range from the aircraft to the specular
point is between 2 and 10 km, and both DDMs of the reflection are non-empty.
The SNR filter of the original pipeline is off (it would drop about three
quarters of the samples); SNR is kept as metadata and the evaluation reports
performance as a function of it. About 52 % of the reflections survive.

A flight has tens of thousands of valid reflections, and neighbouring
reflections are nearly identical. Each flight therefore contributes at most 300
reflections (600 DDMs), drawn at random with class stratification inside the
flight; the per-flight totals before sampling are stored in the flight table,
so the natural prior is known.

### Splits that respect the flight

All DDMs of a flight share the weather, the geometry, the route and the sensor
state. A random split of DDMs measures memorisation, so every split here is
made at the flight level:

- **geographic holdout**: every flight that touches the South Island;
- **temporal holdout**: the last six months of the corpus;
- **combined holdout**: flights in both;
- **train pool**: the rest, cut into five folds by flight (stratified on the
  per-flight land fraction). Fold 0 is the validation fold of the final models.

| split | flights | DDM rows | what it tests |
|---|---:|---:|---|
| train pool (5 folds) | 3,550 | 2,115,000 | in-distribution skill, model selection, tuning |
| geographic holdout | 2,726 | 1,619,888 | a region never seen in training (South Island) |
| temporal holdout | 557 | 332,400 | the last six months |
| combined holdout | 460 | 272,400 | both at once |

Before sampling, the corpus holds 177.3 M reflections, of which 89.0 M pass the
quality filters (30.8 % water, 69.2 % land); the table above keeps 4.34 M DDMs
from 7,233 flights (60 flights had no valid reflection).

Adversarial validation (`02`) makes the design explicit. A classifier is
trained to tell two pools apart; an AUC near 0.5 means the pools are
exchangeable, an AUC near 1 means they are trivially different.

| level | pools | AUC | most useful variables |
|---|---|---:|---|
| flight metadata | train vs geographic holdout | 1.000 | airport codes, **file size** |
| flight metadata | train vs temporal holdout | 0.696 | hour of day, file size, airports |
| DDM features | fold 0 vs the other train folds | 0.544 | none stands out |
| DDM features | train vs geographic holdout | 0.730 | delay-band energy, delay centroid, polarimetric excess ratio |
| DDM features | train vs temporal holdout | 0.774 | spectral median of the waveform, peak-region connectivity |

The folds are exchangeable, as they should be. The holdouts are different by
construction at the flight level, and the file size alone gives away the
island, because South-Island routes are short: file size, route, coordinates
and distance to the coast are stored with every sample for analysis and are
never features. At the feature level the drift is real but moderate: the
geographic holdout differs mostly in *where* the peak sits in delay (a geometry
effect), the temporal holdout in the waveform's spectral content (a sensor or
processing effect). Both are the kind of shift a deployed model would meet.

## Feature extraction

Every DDM is sum-normalized before the shape features are computed, so that
the level of the signal is carried by one small group (`power`) and everything
else describes the shape. The groups are documented feature by feature in
[`docs/FEATURES.md`](docs/FEATURES.md).

| group | n | idea |
|---|---:|---|
| `global` | 45 | distribution of the pixel values, energy centroid and inertia, delay-band energies, derivative / autocorrelation / spectrum of the delay waveform |
| `peak` | 49 | the 40-pixel region grown from the peak: bounding box, perimeter, compactness, spread, intensity statistics, texture (GLCM), connectivity, moments, orientation |
| `coherence` | 16 | peak-to-horseshoe ratio, leading/trailing edge slopes, Shannon and von Neumann entropy, widths at half maximum, deviation from the ideal ambiguity function |
| `power` | 4 | total and peak counts, peak excess over the noise floor, SNR from counts |
| `polarimetric` | 7 | LHCP/RHCP ratios of peak and excess power, SNR difference, shape correlation, PHPR and entropy differences, polarization flag |
| `quadrant` | 16 | mean, std, max, energy of the four quadrants |
| `meta` | 6 | instrument and geometry metadata (SNR, incidence angle, gains, slant range, altitude): known at inference time, tested as an ablation |
| `raw` | 200 | the normalized pixels |

The extractor is vectorized over batches of DDMs (the peak region is grown for
a whole batch at once), which is what makes the full-corpus ETL a matter of
minutes rather than hours.

<!-- FEATURE_FIGURES -->

## Model selection and training

<!-- MODEL_SELECTION -->

### Which features matter

One fixed model (XGBoost, default hyper-parameters, 600 trees) is trained on
each feature set: five flight-grouped folds on a 500,000-row subsample of the
train pool (CV AUC), and once on the whole train pool to score the geographic
and the temporal holdout. Rows marked "default" use the four DDM-only groups
plus the polarimetric one; "default + latent" adds the 20-D code of an
autoencoder trained on the raw pixels (`09`).

| feature set | features | CV AUC | geo AUC | time AUC |
|---|---:|---:|---:|---:|
| raw pixels | 200 | 0.8865 | 0.9083 | 0.8723 |
| global | 45 | 0.8859 | 0.9061 | 0.8704 |
| peak | 49 | 0.8719 | 0.8971 | 0.8584 |
| coherence | 16 | 0.8707 | 0.8956 | 0.8558 |
| power | 4 | 0.8428 | 0.8769 | 0.8608 |
| quadrant | 16 | 0.8690 | 0.8903 | 0.8613 |
| polarimetric | 7 | 0.9257 | 0.9555 | 0.9174 |
| meta | 6 | 0.9223 | 0.9061 | 0.9364 |
| DDM set (global+peak+coherence+power) | 114 | 0.9085 | 0.9225 | 0.9019 |
| default (DDM set + polarimetric) | 121 | 0.9613 | 0.9698 | 0.9592 |
| default + quadrant | 137 | 0.9614 | 0.9700 | 0.9595 |
| default + raw pixels | 321 | 0.9611 | 0.9701 | 0.9596 |
| default + meta | 127 | 0.9822 | 0.9744 | 0.9821 |
| default - global | 76 | 0.9588 | 0.9679 | 0.9567 |
| default - peak | 72 | 0.9614 | 0.9698 | 0.9588 |
| default - coherence | 105 | 0.9611 | 0.9696 | 0.9590 |
| default - power | 117 | 0.9537 | 0.9685 | 0.9461 |
| default - polarimetric | 114 | 0.9085 | 0.9225 | 0.9019 |
| default - delay position | 106 | 0.9594 | 0.9684 | 0.9570 |

![Feature-set ablation](results/figures/feature_ablation.png)

Four findings, in decreasing order of size:

1. **The polarimetric pair is the strongest signal in the data.** Seven
   features computed from the LHCP/RHCP twin DDMs reach an AUC of 0.926 on
   their own, more than the 200 raw pixels (0.887) or any shape group, and
   adding them to the DDM-only set moves the AUC from 0.909 to 0.961. The
   dual-polarization receiver is what makes the problem tractable at this
   accuracy.
2. **Absolute power is the second lever.** Removing the four `power` features
   costs 0.008 AUC in cross-validation and 0.013 on the temporal holdout;
   the shape of a DDM says less than the shape plus how strong it is.
3. **The elaborate groups are redundant.** The 49 peak-region features and the
   16 coherence observables score around 0.87 alone, but removing either from
   the default set changes nothing; the quadrant statistics and the raw pixels
   add nothing either. The information they carry is already in the global
   statistics, the power and the polarimetric ratios.
4. **Metadata helps, but it is a prior, not a measurement.** Receiver SNR,
   incidence angle, antenna gains, slant range and aircraft altitude add
   0.02 AUC in cross-validation and on the temporal holdout, and 0.005 on the
   geographic one. Altitude and range encode the phase of the flight, and
   airports are on land: the gain is real and would hold in production, but it
   is knowledge about where the aircraft is rather than about the surface. The
   final models therefore use the DDM-derived default set, and the metadata
   variant is reported as an option.

The delay position of the peak, which the adversarial validation flagged as
the main difference between the train pool and the South Island, is worth
0.002 AUC in and out of distribution: keeping it is right.

### Tuning

XGBoost was tuned with Optuna over depth, learning rate, child weight,
sub-sampling, column sampling, the three regularisation terms and the number
of trees (69 trials, 52 complete and 16 pruned, each scored on three
flight-grouped folds of a 400,000-row subsample). The optimum is a flat
plateau: the ten best trials sit between 0.9624 and 0.9627 CV AUC, against
0.9613 for the default parameters, with depth 9–11, learning rate 0.03–0.05
and 1,000–1,500 trees. TabNet was tuned on one flight-grouped split (train on
folds 1–3, early stopping on fold 4, score on fold 0) over width, number of
steps, sparsity, learning rate and batch size.

<!-- TUNING_TABNET -->

<!-- FINAL -->

## Reproducing

```bash
git clone https://github.com/ScrPzz/rongowai-earth-surface-classifier -b rework
cd rongowai-earth-surface-classifier
uv sync --extra dev                      # or: pip install -e ".[dev]"
export RONGOWAI_DATA_DIR=/path/to/RONGOWAI_L1_SDR_V1.0
python scripts/00_build_flight_index.py
python scripts/01_etl.py --workers 8
python scripts/02_adversarial_validation.py
python scripts/03_model_selection.py
python scripts/04_feature_ablation.py
python scripts/05_tune_xgboost.py --n-trials 60
python scripts/06_tune_tabnet.py --n-trials 15
python scripts/09_train_autoencoder.py
python scripts/07_train_final.py
python scripts/08_make_figures.py
pytest                                   # unit tests on synthetic DDMs, no data needed
```

`torch` is pinned to the CUDA 12.8 build in `pyproject.toml`; change the index
for a CPU-only install. The L1 files are available from PO.DAAC
(doi:10.5067/RGOWA-S1A10).

## Repository layout

```
configs/default.yaml        thresholds, sampling, split and feature parameters
src/rongowai_ddm/
  netcdf.py                 one flight → arrays
  data/                     flight index, quality, labels, sampling, splits, etl, dataset
  features/                 layout, global_stats, quadrant, peak_region, coherence, power, polarimetric, registry
  models/                   xgb, tabnet_model, zoo, autoencoder, calibration, rule_baseline
  evaluation/               metrics, cv, adversarial, plots
scripts/                    numbered, reproducible steps
tests/                      unit tests on synthetic DDMs
notebooks/                  one flight, end to end (executed)
docs/                       DESIGN.md (requirements and decisions), FEATURES.md (feature catalogue)
results/                    reports, tables and figures produced by the scripts
```

## Acknowledgements and data

Rongowai Science Payloads Operations Centre (2024). *Rongowai Level 1 Science
Data Record Version 1.0*. PO.DAAC, CA, USA. https://doi.org/10.5067/RGOWA-S1A10

The original notebook-based version of this work lives on the `main` branch.

## License

MIT.
