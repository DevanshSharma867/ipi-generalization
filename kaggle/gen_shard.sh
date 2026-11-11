#!/bin/bash
# One generation shard on Kaggle, as jobs/pairs_gen_{val,train}.sh do it on the cluster:
#   bash kaggle/gen_shard.sh MODE DOCS OUT_DIR N_SHARDS K   (config pairs_gen_v6.yaml, gen-train-mistral, batch 8)
set -euo pipefail
MODE=$1 DOCS=$2 OUT=$3 N=$4 K=$5
echo "$(git rev-parse HEAD) kaggle ${KAGGLE_KERNEL_RUN_TYPE:-} $(date "+%Y-%m-%dT%H:%M:%S")" >> "$OUT/commit_$K.txt"
python scripts/gen_pairs.py generate --mode "$MODE" --generator gen-train-mistral --config configs/pairs_gen_v6.yaml \
  --docs "$DOCS" --out_dir "$OUT" --shard "$K" --n_shards "$N" --batch_size 8 ${GEN_DEVICE:+--device "$GEN_DEVICE"}
