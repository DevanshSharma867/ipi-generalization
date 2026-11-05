#!/bin/bash
# ==============================================================================
# SLURM: GPU health probe (no real compute). Lists every GPU with its uncorrected ECC count, then runs
# jobs/lib/gpu_guard.sh (1-tensor CUDA alloc + sync, known-bad UUID check). Writes only its log.
#   sbatch jobs/gpu_probe.sh
# ==============================================================================
#SBATCH --job-name=ipi_gpu_probe
#SBATCH --output=logs/slurm_gpu_probe_%j.log
#SBATCH --error=logs/slurm_gpu_probe_%j.log
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=gpu_rtx_pro_6000_6_csis_hyd
#SBATCH --account=bits
#SBATCH --qos=gpu_csis_course
#SBATCH --gres=mps:50
#SBATCH --requeue
#SBATCH --open-mode=append

set -uo pipefail
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date); commit $(git rev-parse HEAD)"
nvidia-smi --query-gpu=index,uuid,ecc.errors.uncorrected.volatile.total --format=csv
env | grep -E '^CUDA_(VISIBLE_DEVICES|MPS_)' | sort
source .venv/bin/activate
source jobs/lib/gpu_guard.sh
echo "probe PASS"
