#!/bin/bash
# ==============================================================================
# SLURM: score val-selected ICT checkpoints on held-out sets (src/models/ict.py score). One array task per seed.
# Test sets and EVAL pairs are sealed: submit only after every ICT/ablation training run that will be reported has
# finished and nothing more is selected (CLAUDE.md rule 1). Writes results/v3/<exp>_score/<run_id>/.
# Agent sets (AD/TB) need the transfer harness's row format first; this script takes V3 csv and pair jsonl.
# Submit from the repo root, e.g.
#   EXP=ict_full EVALS="test=data/v3/split_seed_0/test.csv pairs=data/pairs/eval_split0.jsonl" sbatch jobs/ict_score.sh
# ==============================================================================
#SBATCH --job-name=ict_score
#SBATCH --output=logs/slurm_ict_score_%A_%a.log
#SBATCH --error=logs/slurm_ict_score_%A_%a.log
#SBATCH --array=0-4
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=gpu_rtx_pro_6000_6_csis_hyd
#SBATCH --account=bits
#SBATCH --qos=gpu_csis_course
#SBATCH --gres=mps:50

source jobs/lib/job_env.sh
export TORCHDYNAMO_DISABLE=1
: "${EXP:?set EXP, e.g. ict_full}" "${EVALS:?set EVALS as name=path ...}"
SPLIT=${SPLIT:-0}
SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-0}}
CKPT=checkpoints/ict/$EXP/split_seed_${SPLIT}_seed$SEED/best
[ -f "$CKPT/ict_config.json" ] || { echo "no checkpoint at $CKPT"; exit 1; }
ARGS=(--exp "$EXP" --ckpt "$CKPT" --seed "$SEED")
for e in $EVALS; do ARGS+=(--eval "$e"); done
python src/models/ict.py score "${ARGS[@]}"
