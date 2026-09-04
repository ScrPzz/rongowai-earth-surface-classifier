#!/bin/bash
# Sequential chain: one GPU job at a time (the GPU is shared with another training run).
cd /home/atogni/Scrivania/progetti/rongowai-rework
export PYTHONUNBUFFERED=1
while pgrep -f "^[.]venv/bin/python scripts/" > /dev/null; do sleep 15; done
run() {
  local name=$1; shift
  echo "=== $(date '+%H:%M:%S') start $name $* ==="
  .venv/bin/python scripts/$name "$@" > results/log_${name%%.py}.log 2>&1
  local rc=$?
  echo "=== $(date '+%H:%M:%S') end $name exit $rc ==="
  if [ $rc -ne 0 ]; then echo "--- tail of results/log_${name%%.py}.log ---"; tail -5 results/log_${name%%.py}.log | cut -c1-300; fi
}
run 05_tune_xgboost.py --n-trials 60 --n-rows 400000
run 04_feature_ablation.py --n-rows 500000
run 09_train_autoencoder.py --epochs 20
run 04_feature_ablation.py --sets latent "default + latent" --latent data/latent.parquet --append
run 06_tune_tabnet.py --n-trials 15
run 10_leakage_demo.py
run 03_model_selection.py --n-rows 300000
run 07_train_final.py
run 08_make_figures.py
echo "=== chain done $(date '+%H:%M:%S') ==="
