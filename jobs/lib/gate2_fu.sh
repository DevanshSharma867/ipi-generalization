# Shared setup for the Gate 2 follow-up jobs (jobs/gate2_fu_{batch,pairing,judge}.sh; docs/deviations.md 2026-10-30,
# paper_plan A19, A20). Source right after the #SBATCH header. Each job writes its own run record
# (results/transfer/gate2_followup_<part>/) and skips finished score files, so a resubmit resumes.
source jobs/lib/job_env.sh
set -e
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv
AD_PY=$PWD/.venv-agentdojo/bin/python
LIU_PY=$PWD/.venv-liu/bin/python
OUR_PY=$PWD/.venv/bin/python
G2=$SCRATCH/nlp_harness/gate2          # Gate 2 work dir (jobs/gate2_liu_repro.sh): run-2 scores
W=$SCRATCH/nlp_harness/gate2_followup
SETS=$PWD/data/transfer/sets
FU=eval/transfer/gate2_followup.py
DET=eval/transfer/detectors.py
mkdir -p "$W"
export TORCHDYNAMO_DISABLE=1 PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1  # as in Gate 2
report() { "$OUR_PY" $FU report --part "$1" --work "$W" --gate2 "$G2" --liu_py "$LIU_PY" --our_py "$OUR_PY"; }
