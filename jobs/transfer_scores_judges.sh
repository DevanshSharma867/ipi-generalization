#!/bin/bash
# ==============================================================================
# SLURM: transfer scoring, judges (shared setup and set list in jobs/lib/transfer_scores.sh).
# Liu's two LLM judges (Qwen2.5-7B, Llama-3.1-8B; his Yes/No rule) in his env, 4-bit at GPU_MEM=18GiB (his
# reproduce.sh setting) and bf16 (A20 precision for judges). ~7 min per judge and precision estimated (Gate 2 rates).
# Submit from the repo root: sbatch jobs/transfer_scores_judges.sh
# ==============================================================================
#SBATCH --job-name=ts_judges
#SBATCH --output=logs/slurm_ts_judges_%j.log
#SBATCH --error=logs/slurm_ts_judges_%j.log
#SBATCH --time=00:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=gpu_rtx_pro_6000_6_csis_hyd
#SBATCH --account=bits
#SBATCH --qos=gpu_csis_course
#SBATCH --gres=mps:50

source jobs/lib/transfer_scores.sh
D="judge-qwen judge-llama"
"$LIU_PY" $DET --detectors $D --sets $SETNAMES --sets_dir "$SETS" --out "$W/judges_nf4" --gpu_mem 18GiB
record transfer_scores_judges_nf4 "$W/judges_nf4" "$LIU_PY" $D
"$LIU_PY" $DET --detectors $D --sets $SETNAMES --sets_dir "$SETS" --out "$W/judges_bf16" --precision bf16 \
  --gpu_mem 40GiB
record transfer_scores_judges_bf16 "$W/judges_bf16" "$LIU_PY" $D
