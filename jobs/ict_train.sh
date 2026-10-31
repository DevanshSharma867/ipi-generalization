#!/bin/bash
# ==============================================================================
# SLURM: train ICT or one of its ablations (src/models/ict.py; paper_plan C4, A18). One array task per training
# seed; paper_plan A18: split seed 0, training seeds 0-4 for everything compared on pairs.
# Prerequisites: data/v3/split_seed_<SPLIT>/ (frozen V3), TRAIN pairs for that split, the backbone in HF_HOME at
# its configs/detectors.yaml pin (scripts/prefetch_models.sh), and the paper_plan amendment that defines
# ablations (b)-(e) and the checkpoint-selection metric, committed before the first submit.
# Resumable: state.pt in checkpoints/ict/<exp>/split_seed_<k>_seed<s>/ is picked up by a resubmit; a seed whose
# run already finished cleanly is skipped. Writes results/v3/<exp>/<run_id>/ via runlog.
# Submit from the repo root, e.g.
#   ABLATION=full TRAIN_PAIRS=data/pairs/train_split0.jsonl VAL_PAIRS=data/pairs/val_split0.jsonl sbatch jobs/ict_train.sh
#   ABLATION=a sbatch jobs/ict_train.sh
#   ABLATION=full BACKBONE=modernbert-large TRAIN_PAIRS=... sbatch --time=08:00:00 jobs/ict_train.sh
# Time is a placeholder until a 1-seed timing run; ModernBERT-large needs more and may need a smaller batch.
# ==============================================================================
#SBATCH --job-name=ict_train
#SBATCH --output=logs/slurm_ict_train_%A_%a.log
#SBATCH --error=logs/slurm_ict_train_%A_%a.log
#SBATCH --array=0-4
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=gpu_rtx_pro_6000_6_csis_hyd
#SBATCH --account=bits
#SBATCH --qos=gpu_csis_course
#SBATCH --gres=mps:50

source jobs/lib/job_env.sh
export TORCHDYNAMO_DISABLE=1   # ModernBERT compiles with Inductor, which needs nvcc (absent on compute nodes)
ABLATION=${ABLATION:-full}
BACKBONE=${BACKBONE:-deberta-v3-base}
SPLIT=${SPLIT:-0}
SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-0}}
SPLIT_DIR=data/v3/split_seed_$SPLIT
EXP=ict_$ABLATION; [ "$BACKBONE" = deberta-v3-base ] || EXP=${EXP}_$BACKBONE
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

if python - "$EXP" "$SEED" "$SPLIT_DIR" <<'PY'; then echo "$EXP seed $SEED on $SPLIT_DIR already finished cleanly; skipping"; exit 0; fi
import json, sys
from pathlib import Path
exp, seed, split_dir = sys.argv[1:]
for d in Path("results/v3", exp).glob(f"*_seed{seed}"):
    c = json.loads((d / "config.json").read_text())
    if (d / "finished.json").exists() and not c["git_dirty"] and Path(c["args"]["split_dir"]) == Path(split_dir):
        sys.exit(0)
sys.exit(1)
PY

ARGS=(--split_dir "$SPLIT_DIR" --ablation "$ABLATION" --backbone "$BACKBONE" --seed "$SEED")
[ -n "${TRAIN_PAIRS:-}" ] && ARGS+=(--train_pairs "$TRAIN_PAIRS")
[ -n "${VAL_PAIRS:-}" ] && ARGS+=(--val_pairs "$VAL_PAIRS")
python src/models/ict.py train "${ARGS[@]}" ${EXTRA_ARGS:-}
