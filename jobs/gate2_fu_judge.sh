#!/bin/bash
# ==============================================================================
# SLURM: Gate 2 follow-up, part "judge" (eval/transfer/gate2_followup.py; shared setup in jobs/lib/gate2_fu.sh).
# Qwen judge (Liu's env, his Yes/No rule): 4-bit repeat on BIPIA (same config as Gate 2 run 2), bf16 vs 4-bit on
# the seeded AD-Matched, then one 4-bit run at GPU_MEM=6GiB. Time: ~37 min estimated (BIPIA 4-bit took ~6 min in
# Gate 2; the 6 GiB run may offload layers to CPU).
# Submit from the repo root: sbatch jobs/gate2_fu_judge.sh
# ==============================================================================
#SBATCH --job-name=g2fu_judge
#SBATCH --output=logs/slurm_g2fu_judge_%j.log
#SBATCH --error=logs/slurm_g2fu_judge_%j.log
#SBATCH --time=00:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=gpu_rtx_pro_6000_6_csis_hyd
#SBATCH --account=bits
#SBATCH --qos=gpu_csis_course
#SBATCH --gres=mps:50

source jobs/lib/gate2_fu.sh

"$LIU_PY" $DET --detectors judge-qwen --sets bipia --sets_dir "$SETS" --out "$W/judge_repeat" --gpu_mem 18GiB
"$LIU_PY" $DET --detectors judge-qwen --sets admatched adclean --sets_dir "$SETS" --out "$W/judge_bf16_seeded" \
  --precision bf16 --gpu_mem 40GiB
"$LIU_PY" $DET --detectors judge-qwen --sets bipia --sets_dir "$SETS" --out "$W/judge_6gib" --gpu_mem 6GiB
report judge
