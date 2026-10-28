#!/bin/bash
# ==============================================================================
# SLURM: score the V2 validation split (data/processed/val.csv) with the public detectors and the judge.
# ==============================================================================
# These scores are the calibration set of the "calibrated" threshold protocol (src/metrics.py,
# tpr_at_fpr_protocols): thresholds are fixed on the clean val rows, then applied to every eval set.
# Writes standard-schema predictions via runlog to results/v2_probes/score_<name>_v2val/<run_id>/.
# Each step is skipped if its results directory already holds a predictions.csv (resumable).
# DeBERTa val scores already exist (results/v2_std/v2_val/); TF-IDF val is scored locally on CPU.
#
# Prerequisites (login node): repo at the commit that added this file, clean tree; hf_cache holds
# protectai/deberta-v3-base-prompt-injection-v2, leolee99/PIGuard, Qwen/Qwen2.5-7B-Instruct.
# Submit from the repo root:  sbatch jobs/score_v2_val.sh
# ==============================================================================

#SBATCH --job-name=ipi_v2_val
#SBATCH --output=logs/slurm_v2_val_%j.log
#SBATCH --error=logs/slurm_v2_val_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
# BITS Hyderabad Sharanga student QoS: cpu=4, mem=80G, mps=50 (half a card)
#SBATCH --partition=gpu_rtx_pro_6000_6_csis_hyd
#SBATCH --account=bits
#SBATCH --qos=gpu_csis_course
#SBATCH --gres=mps:50

MODULES=""                               # e.g. "cuda/12.8"; empty = none
VENV_PATH=".venv"

set -euo pipefail
mkdir -p logs
for m in $MODULES; do module load "$m"; done
source "$VENV_PATH/bin/activate"
export HF_HOME="$PWD/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
echo "Job $SLURM_JOB_ID on $(hostname) at $(date); commit $(git rev-parse HEAD)"

done_already() { compgen -G "results/v2_probes/score_$1/*/predictions.csv" > /dev/null; }

score() {  # score <name> <args...>
    local name=$1; shift
    if done_already "$name"; then echo ">>> Skipping (exists): $name"; return; fi
    echo ">>> Scoring $name"
    python scripts/score_probes.py --name "$name" --sets v2_val "$@"
}

score protectai_v2val --model hf --checkpoint protectai/deberta-v3-base-prompt-injection-v2 --checkpoint_origin hf_hub
score piguard_v2val --model hf --checkpoint leolee99/PIGuard --checkpoint_origin hf_hub
score judge_qwen2.5_7b_v2val --model judge --checkpoint Qwen/Qwen2.5-7B-Instruct --checkpoint_origin hf_hub

echo "Done at $(date)"
