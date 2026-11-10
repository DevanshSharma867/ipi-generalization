#!/bin/bash
# ==============================================================================
# SLURM: held-out TRAIN-side pairs on V3 val docs (paper_plan A22 selection, A32; docs/pairs_spec.md v3 §8;
# config pairs_gen_v5.yaml, mode val). gen-train-mistral (Mistral 7B bf16, TRAIN prompt v3, unchanged), batch 8,
# on the val docs of split seed 0 only (data/pairs/docs/val_s0.jsonl: 132 docs, 13 email, 105 table, 14 code;
# manifest checks them against the frozen data/v3/split_seed_0). Quads per doc as in train (6/3/6 email/table/code):
# 477 quads. Failed quads are regenerated up to 3 times (A29). Two array tasks of ~240 quads, ~0.45 h each at the
# re-timing's 6.29 s/quad (results/pairs/gen_timing/20261109-094809_seed0). Then jobs/pairs_selfcheck.sh filters them.
# Output (home, untracked): data/pairs/val_gen/v5_s0_<code sha>/shard_<k>_of_<n>.jsonl + .meta.json + commit_<k>.txt.
# Resumable: resubmitting the same array finishes what a timeout left (quads already written are skipped).
# Submit from the repo root (~/nlp_harness): sbatch --array=0-1%1 jobs/pairs_gen_val.sh
# ==============================================================================
#SBATCH --job-name=pairs_val
#SBATCH --output=logs/slurm_pairs_val_%A_%a.log
#SBATCH --error=logs/slurm_pairs_val_%A_%a.log
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

source jobs/lib/job_env.sh
N_SHARDS=2
K=${SLURM_ARRAY_TASK_ID:?submit as an array: --array=0-$((N_SHARDS - 1))%1}
DOCS=data/pairs/docs/val_s0.jsonl
CODE_SHA=$(cat configs/pairs_gen_v5.yaml src/pairs_gen.py scripts/gen_pairs.py | sha256sum | cut -c1-12)
OUT=data/pairs/val_gen/v5_s0_$CODE_SHA
mkdir -p "$OUT"
echo "$(git rev-parse HEAD) job ${SLURM_JOB_ID} array ${SLURM_ARRAY_JOB_ID:-}_$K $(date -Is)" >> "$OUT/commit_$K.txt"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

python scripts/gen_pairs.py generate --mode val --generator gen-train-mistral --config configs/pairs_gen_v5.yaml \
  --docs "$DOCS" --out_dir "$OUT" --shard "$K" --n_shards "$N_SHARDS" --batch_size 8
echo "shard $K done: $OUT"
