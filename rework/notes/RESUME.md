# Resume Point — Rongowai DDM Classifier Rework

**Last update:** 2026-05-25 (session interrupted by PC shutdown ~00:36 UTC the same day).

This file is the single entry point to pick the work back up. For deeper background see the auto-memory files (`project_rongowai_improvements.md`, `project_rongowai_blog.md` in `~/.claude/projects/.../memory/`).

---

## TL;DR — Where we stopped

- **Phase 0 (validation harness)**: P0.1 + P0.2 **complete and on disk**. P0.3 baseline **partial**: the only saved baseline metrics are from a 10-flight smoke-test run (May 24, 17:44), not from the full dataset.
- **Phase 1 modules**: 4 files written but **never executed** (no self-tests run, no real-data integration). See "Untested modules" below.
- **Big ETL crash**: a full-dataset ETL run was in progress when the PC died. It got through ~57% of flights; the output parquet files **are corrupted** (no parquet footer) and must be deleted before retrying.
- **No code is broken** — just unfinished. Memory + git are clean.

---

## Disk state inventory (verified 2026-05-25)

### Safe / usable

| Path | What it is | Notes |
|------|------------|-------|
| `rework/results/p0_1_flight_index.parquet` | 7,293 flights with split assignments | P0.1 output, final |
| `rework/results/p0_1_fold_assignments.parquet` | Fold assignments for train_cv | P0.1 output, final |
| `rework/results/p0_1_report.json` | P0.1 summary | Final |
| `rework/results/p0_2_adversarial_report.json` | P0.2 adversarial validation results | Final, see memory for verdict |
| `rework/results/p0_2_feature_importance.csv` | Adversarial classifier importances | Final |
| `rework/results/p0_3_baseline_*.{json,joblib}` | XGBoost smoke-test artefacts | **From 10-flight smoke run, not full data — replace when full P0.3 completes** |
| `rework/results/p0_3_etl.log` | 16 MB log of the crashed ETL | Useful for debugging if you want to know where it died |
| All `rework/{data,validation,calibration,models,experiments}/*.py` | Source code | Intact, last modified May 24 |

### **CORRUPTED — delete before retry**

| Path | Size | Status |
|------|------|--------|
| `rework/results/p0_3_features/train_cv.parquet` | 5.0 GB | No footer — `pyarrow` errors out on read |
| `rework/results/p0_3_features/val_geographic_holdout.parquet` | 6.9 GB | Same |

```bash
rm rework/results/p0_3_features/{train_cv,val_geographic_holdout}.parquet
```

### Git state

- Branch: `geok-integration` (clean against origin)
- All of `rework/` is **untracked** — nothing committed yet. Consider an initial commit before resuming.

---

## Crashed ETL — autopsy

- PID 1291806 launched May 24, ran ~5h 42min before the PC powered off.
- Last log line: 4170 / 7293 flights, 29.6M samples extracted, throughput ~0.20 flights/s (slowing as files got bigger).
- Script writes the `p0_3_etl_report.json` only at the end → its absence confirms incomplete run.
- Parquet writer uses appending Arrow `ParquetWriter`; without a clean `.close()` the footer is never written and the file is unrecoverable.

### Resilience to-do for the next run

Before relaunching the ETL, ideally:

1. Add a **checkpoint mechanism**: flush + close the parquet writer every N flights (e.g. N=500), producing `train_cv_chunk_001.parquet`, `train_cv_chunk_002.parquet`, etc. Concatenate at the end. This way a power loss only loses the in-flight chunk.
2. Add **multiprocessing**: the script is single-threaded but the NetCDF + feature extraction is CPU-bound. With ~8 workers, the 5.5h job should drop to ~45 min.
3. Run under `PYTHONUNBUFFERED=1` and `nohup` so the log flushes in real time and survives terminal close.

Example (current behaviour, single-threaded):
```bash
cd /home/atogni/Scrivania/rongowai-earth-surface-classifier
PYTHONUNBUFFERED=1 nohup python rework/experiments/p0_3_etl.py \
    > rework/results/p0_3_etl.log 2>&1 &
echo $! > rework/results/p0_3_etl.pid
```

---

## Untested modules — Phase 1+ skeletons

All four files were written during the autonomous session, **never run, never integrated**. Each has a `_self_test()` callable via `python -m rework.data.<module>` (or equivalent for `calibration`).

### `rework/data/coherence_features.py` — Phase 1.3

7 scalar features per DDM: `phpr`, `phpr_db`, `les`, `tes`, `edge_ratio_les_abs_tes`, `shannon_entropy`, `vn_entropy`. References: Wang/Hu/Li 2022 (PHPR), Russo 2022 (Von Neumann entropy).

Self-test: synthetic water-like vs land-like DDM, asserts water has higher PHPR and lower entropies.

```bash
python -m rework.data.coherence_features
```

### `rework/data/conv_encoder.py` — Phase 1.2

Conv2d autoencoder (1→16→32 channels → Linear → 20-D latent), symmetric ConvTranspose decoder. Drop-in replacement for the production MLP encoder. Reference: Wang et al. IGARSS 2021.

```bash
python -m rework.data.conv_encoder
```

### `rework/data/polarimetric.py` — Phase 1.1 (SKELETON)

4 scalar features: `polarisation_ratio_db`, `polarisation_purity`, `phpr_lhcp`, `phpr_rhcp`. Reference: Bai & Ruf 2025 (arXiv:2501.10334).

**Blocking issue documented in the module docstring:** the NetCDF channel layout for LHCP/RHCP split is not yet inspected at the byte level. Before this module can run on real data, write a quick inspection script (`rework/experiments/inspect_polarimetric_layout.py`) that opens one NetCDF and answers:

1. How are LHCP and RHCP arranged within `raw_counts`'s `n_samples` axis? (block-by-polarisation vs interleaved?)
2. Is there an auxiliary variable (`ddm_ant`, `polarisation_type`, etc.) that labels each sample's polarisation?
3. Does the existing `ddm_snr` / `sp_rx_gain_*` filter pipeline already split polarisations?

The self-test runs on synthetic data only — it does **not** validate the NetCDF wiring.

### `rework/calibration/beta_calibration.py` — Phase 2.2

`BetaCalibrator` (sklearn-compatible). Fits 3-parameter Beta calibration as logistic regression on `(log p, -log(1-p))`. Reference: Kull, Silva Filho & Flach AISTATS 2017.

```bash
python -m rework.calibration.beta_calibration
```

---

## Other unfinished bits surfaced during the session

### `DDMFeatureExtractorV2` + `PeakRegionDetector` — Task #31 enabler

An Explore agent extracted the full code for these from the notebooks (`[f_eng]adaptive_features_extraction.ipynb`, `peak_detector_demonstator.ipynb`). The code was returned but **not yet written to files in `rework/`**. When you resume, the agent's output gives you the ready-to-paste classes; create:

- `rework/data/peak_detector.py` (PeakRegionDetector, default `method='statistical'`, N=10, produces 65 features)
- `rework/data/feature_extractor_v2.py` (DDMFeatureExtractorV2, `data_format='40x5'`, produces 78 DDM features)

Combined: 78 + 65 = 143 features. This finally explains the "78 features" number in `deliver/README_ONNX.md`: it's the V2 extractor alone, not V1 + peaks.

To re-extract, re-run the Explore agent with this prompt template:

> Read `deliver/[f_eng]adaptive_features_extraction.ipynb` and `deliver/peak_detector_demonstator.ipynb`. Extract the full source of class `DDMFeatureExtractorV2` and class `PeakRegionDetector` (default constructor args, all methods, all helper functions they call). Return as a ready-to-paste Python module per class.

---

## Task tracker snapshot

| ID | Status | Notes |
|----|--------|-------|
| #1–#2, #5, #8, #9 | completed | Design, inventory, references, P0.1, P0.2 |
| #10 | in_progress | P0.3 baseline — code complete, full data run never finished |
| #11 | in_progress | Polarimetric — skeleton only, NetCDF layout TBD |
| #12 | in_progress | Conv2d AE — code only, never trained |
| #13 | in_progress | Coherence features — code only, never integrated |
| #16 | in_progress | Beta calibration — code only, never used end-to-end |
| #31 | pending | 78-feature comparison (V2 extractor) — blocked on #10 + needs files created |
| All others | pending | Phase 1.4 ablation onward |

When resuming, the recommended order is: **(a)** finish P0.3 (rerun ETL with checkpoints + multiproc, then HP search), **(b)** smoke-test the 4 written modules, **(c)** integrate coherence features into the ETL for the first real Phase 1 measurement.

---

## Key constraints to remember (do NOT relearn the hard way)

- **Do not modify `deliver/`.** All new code goes in `rework/`. (User instruction, 2026-05-24.)
- **Drop `file_size_log` from any feature set.** P0.2 showed it's a near-perfect South Island proxy (importance 0.486 in the geographic discriminator). Including it leaks the geographic holdout.
- **Use group K-fold from P0.1 for every measurement.** Random K-fold numbers are not comparable to the new baseline.
- **ZZZZ flights (31 of them) are excluded.** `INVALID_ICAO_CODES = frozenset({"ZZZZ"})` in `rework/validation/geographic_holdout.py`.
- **`MinMaxScaler.fit_transform` is called per-file**, matching production. Do not change to a global scaler — the production model expects this quirk.
- **Feature count is 72 for the current baseline** (20 encoder + 52 stats), not 78 as the README_ONNX claims. See Task #31 for the resolution path.
- **Working dir must be the project root** (`/home/atogni/Scrivania/rongowai-earth-surface-classifier`), not `rework/`, for the experiments to find their relative imports.

---

## Quick command cheatsheet

```bash
# Project root
cd /home/atogni/Scrivania/rongowai-earth-surface-classifier

# Run the (untested) Phase 1 self-tests
python -m rework.data.coherence_features
python -m rework.data.conv_encoder
python -m rework.data.polarimetric
python -m rework.calibration.beta_calibration

# Restart the ETL (single-threaded, ~5.5h). Delete corrupted parquet first.
rm -f rework/results/p0_3_features/{train_cv,val_geographic_holdout}.parquet
PYTHONUNBUFFERED=1 nohup python rework/experiments/p0_3_etl.py \
    > rework/results/p0_3_etl.log 2>&1 &

# Train the baseline once ETL completes
python rework/experiments/p0_3_train_baseline.py --n-iter 250 --n-splits 5

# P0.1 / P0.2 — already done, but if you ever need to rerun:
python rework/experiments/p0_1_build_harness.py
python rework/experiments/p0_2_adversarial_validation.py
```

---

## Pointers to deeper context

- **Full plan + rationale**: `~/.claude/projects/-home-atogni-Scrivania-rongowai-earth-surface-classifier/memory/project_rongowai_improvements.md`
- **Blog post scope**: `~/.claude/projects/-home-atogni-Scrivania-rongowai-earth-surface-classifier/memory/project_rongowai_blog.md`
- **References (35 verified citations)**: `rework/references.md`
- **Dataset**: `/media/atogni/dati/rongowai/data` (7,324 NetCDFs, 1.12 TiB, 2022-10-26 → 2026-05-23)
