#!/bin/bash
# Kaggle session 1 (docs/deviations.md 2026-11-11): the V3 baselines and H1 tasks of jobs/v3_baselines.sh
# (jobs/lib/v3_baseline_task.sh, unchanged), fp32, two queues on the two T4s. Outputs are zipped to /kaggle/working
# on exit, also after a failure. Exits non-zero if any task failed. Re-running skips finished runs.
set -euo pipefail
: "${ASSETS:?set ASSETS}"
export NLTK_DATA=/tmp/nltk_data BASE=/tmp/hf/deberta-v3-base AGENT_DIR=data/transfer/sets_adliu
export V2_CKPT_ROOT=$ASSETS/v2_ckpts
trap 'bash kaggle/package.sh session1 || echo "packaging failed"' EXIT
source jobs/lib/v3_baseline_task.sh
mkdir -p logs; rm -f logs/kaggle_s1_failed
queue() {  # queue GPU TASK...: one task after another on one GPU; a failed task is logged, the rest still run
  local gpu=$1; shift
  ( export CUDA_VISIBLE_DEVICES=$gpu
    for t in "$@"; do
      echo "$(date "+%Y-%m-%dT%H:%M:%S") gpu $gpu task $t start"
      if ( set -e; v3_baseline_task "$t" ) >> "logs/kaggle_s1_task$t.log" 2>&1; then echo "$(date "+%Y-%m-%dT%H:%M:%S") task $t done"
      else echo "$(date "+%Y-%m-%dT%H:%M:%S") task $t FAILED"; echo "$t" >> logs/kaggle_s1_failed; tail -20 "logs/kaggle_s1_task$t.log"; fi
    done )
}
TASKS0=${TASKS0:-"0 1 2 7"}; TASKS1=${TASKS1:-"3 4 5 6 8 9"}
queue 0 $TASKS0 & p0=$!
queue 1 $TASKS1 & p1=$!
wait $p0 $p1
if [ -s logs/kaggle_s1_failed ]; then echo "SESSION 1 FAILED tasks: $(tr '\n' ' ' < logs/kaggle_s1_failed)"; exit 1; fi
echo "SESSION 1 COMPLETE"
