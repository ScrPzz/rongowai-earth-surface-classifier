#!/bin/bash
# Remaining steps after the 2026-09-04 checkpoint: screening, final training, figures (one GPU job at a time).
cd /home/atogni/Scrivania/progetti/rongowai-rework
export PYTHONUNBUFFERED=1
run() {
  local name=$1; shift
  echo "=== $(date '+%H:%M:%S') start $name $* ==="
  .venv/bin/python scripts/$name "$@" > results/log_${name%%.py}.log 2>&1
  local rc=$?
  echo "=== $(date '+%H:%M:%S') end $name exit $rc ==="
  if [ $rc -ne 0 ]; then echo "--- tail of results/log_${name%%.py}.log ---"; tail -5 results/log_${name%%.py}.log | cut -c1-300; fi
}
run 03_model_selection.py --n-rows 300000
run 07_train_final.py
run 08_make_figures.py
echo "=== chain done $(date '+%H:%M:%S') ==="
