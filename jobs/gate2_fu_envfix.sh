#!/bin/bash
# ==============================================================================
# SLURM: Gate 2 follow-up, part "envfix" (eval/transfer/gate2_followup.py envfix; shared setup in jobs/lib/gate2_fu.sh).
# Our env (transformers 5) on the 500-row BIPIA subset with the two loader differences undone (repo tokenizer.json for
# PIGuard and PG2; ModernBERT local RoPE theta for Horizon), vs Liu's env at bs 1 (jobs/gate2_fu_batch.sh output).
# Time: ~8 min estimated (6 model loads, 500 rows each). Submit from the repo root: sbatch jobs/gate2_fu_envfix.sh
# ==============================================================================
#SBATCH --job-name=g2fu_envfix
#SBATCH --output=logs/slurm_g2fu_envfix_%j.log
#SBATCH --error=logs/slurm_g2fu_envfix_%j.log
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu_rtx_pro_6000_6_csis_hyd
#SBATCH --account=bits
#SBATCH --qos=gpu_csis_course
#SBATCH --gres=mps:50

source jobs/lib/gate2_fu.sh
"$OUR_PY" $FU envfix --work "$W" --device cuda
