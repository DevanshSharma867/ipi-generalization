#!/bin/bash
# ==============================================================================
# SLURM: TRAIN intent-pair generation (paper_plan A18 Q3, A29; docs/pairs_spec.md v3 §8; config pairs_gen_v4.yaml).
# gen-train-mistral (Mistral 7B bf16, TRAIN prompt v3), batch 8, on the TRAIN docs of split seed 0 only
# (data/pairs/docs/train_s0.jsonl: 609 docs, 50 email, 491 table, 68 code; manifest checks them against the frozen
# data/v3/split_seed_0). Quads per doc 6/3/6 (email/table/code): 2,181 quads. Failed quads are regenerated up to 3
# times (A29). One array task = one shard of docs (round robin over sorted doc ids), sized to stay well under 1 h:
# re-timing run results/pairs/gen_timing/20261109-094809_seed0 measured 6.29 s/quad at batch 8 including retries,
# so about 273 quads and 0.5 h per shard.
# Output (home, not scratch; untracked): data/pairs/train_gen/v4_s0_<code sha>/shard_<k>_of_<n>.jsonl + .meta.json,
# plus commit_<k>.txt with the commit and job id of each attempt. Resumable: a shard skips quads already written
# (the shard file is rewritten after every batch), so resubmitting the same array finishes what a timeout left.
# The self-consistency filter (spec §8, TRAIN) runs later on these records; it is not part of this job.
# Submit from the repo root (~/nlp_harness): sbatch --array=0-7%1 jobs/pairs_gen_train.sh
# ==============================================================================
#SBATCH --job-name=pairs_train
#SBATCH --output=logs/slurm_pairs_train_%A_%a.log
#SBATCH --error=logs/slurm_pairs_train_%A_%a.log
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
N_SHARDS=8
K=${SLURM_ARRAY_TASK_ID:?submit as an array: --array=0-$((N_SHARDS - 1))%1}
DOCS=data/pairs/docs/train_s0.jsonl
CODE_SHA=$(cat configs/pairs_gen_v4.yaml src/pairs_gen.py scripts/gen_pairs.py | sha256sum | cut -c1-12)
OUT=data/pairs/train_gen/v4_s0_$CODE_SHA
mkdir -p "$OUT"
echo "$(git rev-parse HEAD) job ${SLURM_JOB_ID} array ${SLURM_ARRAY_JOB_ID:-}_$K $(date -Is)" >> "$OUT/commit_$K.txt"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

python scripts/gen_pairs.py generate --mode train --generator gen-train-mistral --config configs/pairs_gen_v4.yaml \
  --docs "$DOCS" --out_dir "$OUT" --shard "$K" --n_shards "$N_SHARDS" --batch_size 8
echo "shard $K done: $OUT"
