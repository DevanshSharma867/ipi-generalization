#!/bin/bash
# ==============================================================================
# SLURM: transfer scoring, encoders (shared setup and set list in jobs/lib/transfer_scores.sh).
# Every published classifier Liu scored, in his env (transformers 4.57.6), except Wolf Defender (its own job) and
# protectai-small-v2 (access pending). ~7.8k rows per detector; Gate 2 rates give ~45 min (prismor-1.5b slowest).
# Submit from the repo root: sbatch jobs/transfer_scores_encoders.sh
# ==============================================================================
#SBATCH --job-name=ts_encoders
#SBATCH --output=logs/slurm_ts_encoders_%j.log
#SBATCH --error=logs/slurm_ts_encoders_%j.log
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=gpu_rtx_pro_6000_6_csis_hyd
#SBATCH --account=bits
#SBATCH --qos=gpu_csis_course
#SBATCH --gres=mps:50

source jobs/lib/transfer_scores.sh
D="protectai-v1 protectai-v2 piguard jasperls testsavant-large horizon-base sheltron sheltron-taskaware \
sheltron-generic prismor-1.5b promptguard2-86m promptguard2-22m promptguard1-86m qualifire-sentinel rogue-sentinel-v2"
"$LIU_PY" $DET --detectors $D --sets $SETNAMES --sets_dir "$SETS" --out "$W/liuenv"
record transfer_scores_liuenv "$W/liuenv" "$LIU_PY" $D
