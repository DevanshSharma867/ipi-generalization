#!/bin/bash
# ==============================================================================
# SLURM: train ICT, its grid or one of its ablations (src/models/ict.py; paper_plan C4, A18, A22).
# Order (A22):
#   1. Grid, full ICT, training seed 0, one array task per grid point (12):
#        GRID=1 TRAIN_PAIRS=... VAL_PAIRS=... sbatch --array=0-11 jobs/ict_train.sh
#   2. python src/models/ict.py select   (login node) -> configs/ict_hparams.json; commit it.
#   3. Full ICT and ablations a-e, training seeds 0-4 on split seed 0, with the selected hyperparameters:
#        ABLATION=full TRAIN_PAIRS=... VAL_PAIRS=... sbatch jobs/ict_train.sh
#        ABLATION=a VAL_PAIRS=... sbatch jobs/ict_train.sh
# VAL_PAIRS: held-out TRAIN-side pairs on val docs (selection), never EVAL pairs. AGENT_DIR defaults to
# data/agent_v1/split_seed_<SPLIT> (src/agent_pipeline.py); AGENT_PAIRS is optional.
# Prerequisites: frozen V3 split, agent-style data, TRAIN pairs, the backbone in HF_HOME at its pin.
# Resumable: state.pt under checkpoints/ict/<exp>/ is picked up by a resubmit; a task whose run already finished
# cleanly is skipped. Time is a placeholder until a 1-seed timing run (ModernBERT-large needs more).
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
: "${VAL_PAIRS:?set VAL_PAIRS (held-out TRAIN-side pairs on val docs)}"
ABLATION=${ABLATION:-full}
BACKBONE=${BACKBONE:-deberta-v3-base}
SPLIT=${SPLIT:-0}
SPLIT_DIR=data/v3/split_seed_$SPLIT
AGENT_DIR=${AGENT_DIR:-data/agent_v1/split_seed_$SPLIT}
TASK=${SLURM_ARRAY_TASK_ID:-0}
if [ -n "${GRID:-}" ]; then
  SEED=0; GRID_INDEX=$TASK; EXP=ict_grid; ABLATION=full
else
  SEED=${SEED:-$TASK}; GRID_INDEX=""; EXP=ict_$ABLATION
fi
[ "$BACKBONE" = deberta-v3-base ] || EXP=${EXP}_$BACKBONE
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

if python - "$EXP" "$SEED" "$SPLIT_DIR" "$GRID_INDEX" <<'PY'; then echo "$EXP seed $SEED grid '$GRID_INDEX' on $SPLIT_DIR already finished cleanly; skipping"; exit 0; fi
import json, sys
from pathlib import Path
exp, seed, split_dir, grid = sys.argv[1:]
for d in Path("results/v3", exp).glob(f"*_seed{seed}"):
    c = json.loads((d / "config.json").read_text())
    a = c["args"]
    if (d / "finished.json").exists() and not c["git_dirty"] and Path(a["split_dir"]) == Path(split_dir) \
            and str(a.get("grid_index") if a.get("grid_index") is not None else "") == grid:
        sys.exit(0)
sys.exit(1)
PY

ARGS=(--split_dir "$SPLIT_DIR" --val_pairs "$VAL_PAIRS" --ablation "$ABLATION" --backbone "$BACKBONE" --seed "$SEED")
[ -n "$GRID_INDEX" ] && ARGS+=(--grid_index "$GRID_INDEX")
[ -d "$AGENT_DIR" ] && ARGS+=(--agent_dir "$AGENT_DIR")
[ -n "${TRAIN_PAIRS:-}" ] && ARGS+=(--train_pairs "$TRAIN_PAIRS")
[ -n "${AGENT_PAIRS:-}" ] && ARGS+=(--agent_pairs "$AGENT_PAIRS")
python src/models/ict.py train "${ARGS[@]}" ${EXTRA_ARGS:-}
