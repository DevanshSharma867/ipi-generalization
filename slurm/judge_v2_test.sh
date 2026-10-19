#!/bin/bash
# ==============================================================================
# SLURM: Qwen2.5-7B judge (frozen score_detectors.JUDGE_SYSTEM prompt, same settings as slurm/probe_v2.sh)
# on the 288-row V2 test set, which contains the 72 untouched raw clean rows.
# Gives the judge's natural false-positive rate and V2 test macro-F1; probe_report.py merges this run with
# the judge's earlier probe run (latest run per eval_set).
# Submit from the repo root:  sbatch slurm/judge_v2_test.sh
# ==============================================================================

#SBATCH --job-name=ipi_judge_v2test
#SBATCH --output=logs/slurm_judge_v2test_%j.log
#SBATCH --error=logs/slurm_judge_v2test_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=gpu_rtx_pro_6000_6_csis_hyd
#SBATCH --account=bits
#SBATCH --qos=gpu_csis_course
#SBATCH --gres=mps:50

MODULES=""
VENV_PATH=".venv"

set -euo pipefail
mkdir -p logs
for m in $MODULES; do module load "$m"; done
source "$VENV_PATH/bin/activate"
export HF_HOME="$PWD/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
echo "Job $SLURM_JOB_ID on $(hostname) at $(date); commit $(git rev-parse HEAD)"

python scripts/score_probes.py --model judge --checkpoint Qwen/Qwen2.5-7B-Instruct --name judge_qwen2.5_7b \
    --checkpoint_origin hf_hub --sets v2_test

echo "Done at $(date)"
