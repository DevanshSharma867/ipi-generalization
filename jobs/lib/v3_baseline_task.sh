# One V3-baseline/H1 task (paper_plan §7 step 4, A30), shared by jobs/v3_baselines.sh (cluster) and
# kaggle/session1.sh (Kaggle). Usage: v3_baseline_task <T>, with python on PATH and
#   BASE          microsoft/deberta-v3-base snapshot 8ccc9b6 (the one the V2 runs used)
#   AGENT_DIR     data/transfer/sets_adliu (agent sets + calibration pool)
#   V2_CKPT_ROOT  directory holding deberta_s42, deberta_s123, deberta_s999 (existing V2 checkpoints)
# Tasks: 0-4 V3 DeBERTa split seed T (train, then score V3 sets + V3 probes + agent sets); 5-6 V2 DeBERTa seed T-5
# (train on frozen V2, score agent sets); 7-9 score V2 seeds 42/123/999. A finished clean run is skipped; training
# resumes from <ckpt>/state.pt.
AGENT_SETS="admatched adshift tbeval adclean tbclean calpool"

source "$(dirname "${BASH_SOURCE[0]}")/done_run.sh"

v3_baseline_task() {
  local T=$1 K S CK SPLIT
  [ -f "$AGENT_DIR/manifest.json" ] || { echo "missing $AGENT_DIR/manifest.json"; return 1; }
  if [ "$T" -le 4 ]; then
    K=$T; CK=checkpoints/v3/deberta_v3/split_seed_${K}_seed${K}; SPLIT=data/v3/split_seed_$K
    done_run v3 deberta_v3_train $K || python src/models/deberta.py --mode train --data_dir $SPLIT --output_dir $CK \
        --model_name "$BASE" --seed $K --skip_test --run_exp deberta_v3_train --run_version v3 --data_version v3 || return 1
    done_run v3 score_deberta_v3 $K || python scripts/score_baselines.py --model deberta --ckpt $CK \
        --exp score_deberta_v3 --seed $K --model_name deberta_v3_s$K --v3_split_dir $SPLIT \
        --probes data/v3_probes/split_seed_$K.csv --agent_dir "$AGENT_DIR" --agent_sets $AGENT_SETS \
        --data_version "v3 + $(basename "$AGENT_DIR")" || return 1
  elif [ "$T" -le 6 ]; then
    S=$((T - 5)); CK=checkpoints/h1/deberta_v2_s$S
    done_run h1 deberta_v2_train $S || python src/models/deberta.py --mode train --data_dir data/processed \
        --output_dir $CK --model_name "$BASE" --seed $S --skip_test --run_exp deberta_v2_train --run_version h1 \
        --data_version v2-frozen || return 1
    done_run h1 score_deberta_v2 $S || python scripts/score_baselines.py --model deberta --ckpt $CK \
        --exp score_deberta_v2 --version h1 --seed $S --model_name deberta_v2_s$S --agent_dir "$AGENT_DIR" \
        --agent_sets $AGENT_SETS --data_version "$(basename "$AGENT_DIR")" || return 1
  else
    S=$(echo "42 123 999" | cut -d' ' -f$((T - 6))); CK=$V2_CKPT_ROOT/deberta_s$S
    done_run h1 score_deberta_v2 $S || python scripts/score_baselines.py --model deberta --ckpt "$CK" \
        --exp score_deberta_v2 --version h1 --seed $S --model_name deberta_v2_s$S --agent_dir "$AGENT_DIR" \
        --agent_sets $AGENT_SETS --data_version "$(basename "$AGENT_DIR")" || return 1
  fi
}
