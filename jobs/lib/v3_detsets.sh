# Shared setup for the V3 published-detector jobs (jobs/v3_detectors.sh, jobs/v3_judges.sh; paper_plan A30).
# Builds data/transfer/v3_sets from the committed data/v3 (scripts/v3_detector_sets.py, deterministic) privately,
# moves it into place, refuses a stale dir, and defines record() for eval/transfer/record_scores.py.
source jobs/lib/job_env.sh
set -e
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv
LIU_PY=$PWD/.venv-liu/bin/python
OUR_PY=$PWD/.venv/bin/python
DET=eval/transfer/detectors.py
SETS=$PWD/data/transfer/v3_sets
export TORCHDYNAMO_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$OUR_PY" scripts/v3_detector_sets.py --out "$SETS.$SLURM_JOB_ID"
version() { "$OUR_PY" -c "import json, sys; print(json.load(open(sys.argv[1]))['data_version'])" "$1/manifest.json"; }
NEW=$(version "$SETS.$SLURM_JOB_ID")
mv -T "$SETS.$SLURM_JOB_ID" "$SETS" 2>/dev/null || rm -rf "$SETS.$SLURM_JOB_ID"
[ "$(version "$SETS")" = "$NEW" ] || { echo "$SETS is $(version "$SETS"), code builds $NEW: move it aside"; exit 1; }
W=$SCRATCH/nlp_harness/v3_detscores/$NEW
mkdir -p "$W"
record() {
  local exp=$1 out=$2 py=$3; shift 3
  "$OUR_PY" eval/transfer/record_scores.py --version v3 --exp "$exp" --out "$out" --sets_dir "$SETS" --detectors "$@" \
    --sets v3all --scorer_py "$py"
}
