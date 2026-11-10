#!/bin/bash
# ==============================================================================
# SLURM: published detectors on V3 (paper_plan A30), wave 2: the two task-aware LLM judges (Qwen2.5-7B,
# Llama-3.1-8B) in bf16 (A20) with our Yes/No variant rule (A12), on the same v3_all.jsonl as jobs/v3_detectors.sh.
# Submit only after wave 1 (jobs/v3_baselines.sh, v3_tfidf.sh, v3_detectors.sh) is submitted.
# Time: F021's bf16+nf4 judge pass over ~7.8k rows took 12 min for both judges (job 426037); ~9.0k rows here, bf16 only.
# Submit from the repo root: sbatch jobs/v3_judges.sh
# ==============================================================================
#SBATCH --job-name=v3_judges
#SBATCH --output=logs/slurm_v3_judges_%j.log
#SBATCH --error=logs/slurm_v3_judges_%j.log
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=gpu_rtx_pro_6000_6_csis_hyd
#SBATCH --account=bits
#SBATCH --qos=gpu_csis_course
#SBATCH --gres=mps:50
#SBATCH --requeue
#SBATCH --open-mode=append

source jobs/lib/v3_detsets.sh
D="judge-qwen judge-llama"
"$LIU_PY" $DET --detectors $D --sets v3all --sets_dir "$SETS" --out "$W/judges_bf16" --precision bf16 \
  --judge_rule variants --gpu_mem 40GiB
record v3_detscores_judges_bf16 "$W/judges_bf16" "$LIU_PY" $D
