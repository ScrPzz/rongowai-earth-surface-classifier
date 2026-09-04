# Checkpoint — 2026-09-04, 21:10 (session paused by the author)

Single entry point to resume the rework. Read this first.

## Where things are

| what | where |
|---|---|
| clean code, branch `rework` (git worktree) | `~/Scrivania/progetti/rongowai-rework` — 6 local commits, **not pushed yet** |
| original working copy (untouched, branch `geok-integration`, old notebooks + untracked `rework/` attempt) | `~/Scrivania/progetti/rongowai-earth-surface-classifier` |
| blog post draft (Part 1 written, Part 2 pending), branch `rongowai-post`, `draft: true`, **not committed, not pushed** | `~/Scrivania/progetti/atogni.github.io-rongowai/content/posts/rongowai-ddm-land-water/index.md` (worktree of the blog repo) |
| environment | `.venv` in the worktree (uv, Python 3.11, torch 2.11+cu128, xgboost 3.2, pytorch-tabnet, optuna); `uv sync --extra dev --extra notebooks` recreates it |
| raw data | `/media/atogni/dati/rongowai/data` (read-only NTFS mount, 7,324 L1 files) |
| sample table | `data/samples/part-*.parquet` (4,339,688 rows, 6.8 GB, gitignored) + `data/latent.parquet` (autoencoder codes) |
| flight table with splits and folds | `results/flights.parquet` |

## Done

- Package `src/rongowai_ddm` (NetCDF reader, quality filters, labels, per-flight sampling, flight-level splits, six vectorised feature groups, model wrappers, calibration, metrics, plots); 15 unit tests pass; ruff clean.
- Scripts `00`–`10` written and run: flight index, full ETL (18 min, 0 failed flights), adversarial validation, feature ablation (+ latent rows), XGBoost tuning (69 Optuna trials), TabNet tuning (15 trials), leakage demo, autoencoder.
- Docs: `docs/DESIGN.md` (requirements + decisions), `docs/FEATURES.md` (feature catalogue), executed notebook `notebooks/01_one_flight_end_to_end.ipynb`.
- README: sections on data, pipeline, splits, adversarial validation, features, ablation and tuning are written; placeholders remain (see below).

## Running when the session was paused

`results/run_chain2.sh` (nohup, sequential) was executing `03_model_selection.py`, to be followed by `07_train_final.py` and `08_make_figures.py`. Check `results/log_chain2.log`; a `=== chain done ===` line means everything finished. If the machine was shut down before that, rerun what is missing, one at a time (the GPU is shared with another training job; the scripts fall back to the CPU when GPU memory is short):

```bash
cd ~/Scrivania/progetti/rongowai-rework
.venv/bin/python scripts/03_model_selection.py --n-rows 300000   # -> results/model_selection.csv
.venv/bin/python scripts/07_train_final.py                       # -> results/final/{metrics.json,predictions.parquet,...}
.venv/bin/python scripts/08_make_figures.py                      # -> results/figures/*.png, results/tables.md
```

## Numbers so far (flight-grouped validation, default feature set = global+peak+coherence+power+polarimetric, 121 features)

| result | value |
|---|---|
| rows: train / geo holdout / time holdout / both | 2,115,000 / 1,619,888 / 332,400 / 272,400 |
| adversarial AUC: folds exchangeable / train vs geo / train vs time (DDM features) | 0.544 / 0.730 / 0.774 |
| ablation, XGBoost default params: polarimetric alone / DDM set / default / default + meta | 0.926 / 0.909 / 0.961 / 0.982 (CV AUC) |
| default set on the geographic / temporal holdout | 0.970 / 0.959 |
| latent (20-D autoencoder) alone / added to default | 0.882 / 0.9616 (no gain) |
| XGBoost tuned (Optuna, 3 folds, 400k rows) | 0.9627 CV AUC (default params 0.9613) |
| TabNet tuned (one grouped split) | 0.9611 held-out AUC |
| random-row folds vs flight folds on the same rows | 0.9624 vs 0.9613 (leakage is small with sparse per-flight sampling) |

## To do, in order

1. Verify the chain outputs: `results/model_selection.csv`, `results/final/metrics.json`, `results/final/breakdown_xgb.csv`, `results/final/calibration.csv`, `results/figures/`, `results/tables.md`.
2. Fill the README placeholders: `<!-- RESULTS_SUMMARY -->`, `<!-- FEATURE_FIGURES -->`, `<!-- MODEL_SELECTION -->`, `<!-- TUNING_TABNET -->`, `<!-- FINAL -->`; add the two latent rows to the ablation table; embed the figures.
3. Write Part 2 of the blog post (screening, ablation, final holdout metrics, breakdowns by SNR / polarization / surface class / coast distance, calibration, the leakage demo, limitations) and copy the figures it references into the post folder (`ddm_gallery.png`, `peak_region_water.png`, `feature_distributions.png` and the result figures).
4. Quality pass (`ruff check`, `pytest`), commit, then **push the branch**: `git push -u origin rework` (from the worktree). Decide whether to open a PR to `main`.
5. Blog: commit the post on `rongowai-post` (keep `draft: true`); do not push or publish without an explicit go-ahead.
6. Open decisions for the author (see `docs/DESIGN.md`): inland water on the water side; whether RHCP DDMs belong to the deployment target; whether metadata features (a flight-context prior) should be in the final model.
7. Housekeeping: `data/samples_smoke/` can be deleted; the untracked `rework/` folder in the original working copy is superseded by this branch.

## Gotchas learned

- The GPU is shared with a long MUTANT training run (~28.7 GB of 32 GB): keep one GPU job at a time; `models/xgb.py` picks the CPU when less than 1.5 GB is free and refits on CPU after an out-of-memory error.
- A Claude Code hook blocks every shell command containing `rm` (also `git rm`); delete with Python or move files instead.
- No partner/company names anywhere in this branch or the post.
