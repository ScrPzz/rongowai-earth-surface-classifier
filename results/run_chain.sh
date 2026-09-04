#!/bin/bash
# Sequential chain of the GPU-heavy steps; one XGBoost/TabNet process at a time.
cd /home/atogni/Scrivania/progetti/rongowai-rework
export PYTHONUNBUFFERED=1
run() { echo "=== $(date '+%H:%M:%S') start $1 ==="; .venv/bin/python scripts/$1 "${@:2}" > results/log_${1%%.py}.log 2>&1; echo "=== $(date '+%H:%M:%S') end $1 exit $? ==="; }
run 05_tune_xgboost.py --n-trials 60 --n-rows 400000
run 04_feature_ablation.py --n-rows 500000
run 09_train_autoencoder.py --epochs 20
run 06_tune_tabnet.py --n-trials 15
run 10_leakage_demo.py
run 07_train_final.py
run 08_make_figures.py
echo "=== chain done $(date '+%H:%M:%S') ==="
