# Shared setup for the transfer scoring jobs (jobs/transfer_scores_*.sh; paper_plan §3, A3, A16, A20;
# docs/deviations.md 2026-10-30 and 2026-10-31). Source right after the #SBATCH header.
# Builds data/transfer/sets_adliu if absent (recovered AD pairing + calibration pool), then each job scores its
# detectors on the six sets below into $W (finished score files are skipped, so a resubmit resumes) and writes a
# run record results/transfer/<exp>/ with eval/transfer/record_scores.py (our env).
source jobs/lib/job_env.sh
set -e
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv
LIU_PY=$PWD/.venv-liu/bin/python
OUR_PY=$PWD/.venv/bin/python
DET=eval/transfer/detectors.py
SETS=$PWD/data/transfer/sets_adliu
W=$SCRATCH/nlp_harness/transfer_scores
SETNAMES="admatched adshift tbeval adclean tbclean calpool"
mkdir -p "$W"
export TORCHDYNAMO_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [ ! -f "$SETS/manifest.json" ]; then  # jobs may start together: build privately, then move into place once
  "$OUR_PY" eval/transfer/sets_adliu.py --base data/transfer/sets \
    --pairing_run results/transfer/gate2_followup_pairing/20261030-201440_seed0 --out "$SETS.$SLURM_JOB_ID"
  mv -T "$SETS.$SLURM_JOB_ID" "$SETS" 2>/dev/null || rm -rf "$SETS.$SLURM_JOB_ID"
fi
# record EXP OUT SCORER_PY DETECTORS...
record() {
  local exp=$1 out=$2 py=$3; shift 3
  "$OUR_PY" eval/transfer/record_scores.py --exp "$exp" --out "$out" --sets_dir "$SETS" --detectors "$@" \
    --sets $SETNAMES --scorer_py "$py"
}
