#!/bin/bash
# Kaggle session 1 (docs/deviations.md 2026-11-11): the V3 baselines and H1 tasks of jobs/v3_baselines.sh
# (jobs/lib/v3_baseline_task.sh, unchanged), fp32, two queues on the two T4s. Then zips results and checkpoints to
# /kaggle/working. Run from the repo root after kaggle/setup.sh session1. Re-running skips finished runs.
set -uo pipefail
export ASSETS=${ASSETS:-/kaggle/input/ipi-assets}
export BASE=/tmp/hf/deberta-v3-base AGENT_DIR=data/transfer/sets_adliu V2_CKPT_ROOT=$ASSETS/v2_ckpts
source jobs/lib/v3_baseline_task.sh
queue() {  # queue GPU TASK...
  local gpu=$1; shift
  ( export CUDA_VISIBLE_DEVICES=$gpu
    for t in "$@"; do
      echo "$(date -Is) gpu $gpu task $t start"
      if v3_baseline_task "$t" >> "logs/kaggle_s1_task$t.log" 2>&1; then echo "$(date -Is) task $t done"
      else echo "$(date -Is) task $t FAILED (logs/kaggle_s1_task$t.log)"; fi
    done )
}
queue 0 0 1 2 7 &
queue 1 3 4 5 6 8 9 &
wait
bash kaggle/package.sh session1
