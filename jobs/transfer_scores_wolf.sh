#!/bin/bash
# ==============================================================================
# SLURM: transfer scoring, wolf (shared setup and set list in jobs/lib/transfer_scores.sh).
# Wolf Defender in our env (transformers 5.20.1; Liu: its own transformers >= 5.6 env). ~5 min estimated.
# Submit from the repo root: sbatch jobs/transfer_scores_wolf.sh
# ==============================================================================
#SBATCH --job-name=ts_wolf
#SBATCH --output=logs/slurm_ts_wolf_%j.log
#SBATCH --error=logs/slurm_ts_wolf_%j.log
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=gpu_rtx_pro_6000_6_csis_hyd
#SBATCH --account=bits
#SBATCH --qos=gpu_csis_course
#SBATCH --gres=mps:50

source jobs/lib/transfer_scores.sh
"$OUR_PY" $DET --detectors wolf-defender --sets $SETNAMES --sets_dir "$SETS" --out "$W/wolf"
record transfer_scores_wolf "$W/wolf" "$OUR_PY" wolf-defender
