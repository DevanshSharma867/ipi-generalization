# TF-IDF V3 steps (paper_plan A30), shared by jobs/v3_tfidf.sh (cluster) and kaggle/tfidf.sh (Kaggle CPU).
# Per split seed: train with C selected on V3 val (scripts/tfidf_v3.py), then score V3 sets + V3 probes. A finished
# clean run is skipped. Needs python on PATH and nltk_data (stopwords, wordnet, punkt) at the repo root.
source "$(dirname "${BASH_SOURCE[0]}")/done_run.sh"
v3_tfidf_steps() {
  local K
  for K in 0 1 2 3 4; do
    done_run v3 tfidf_v3_train $K || python scripts/tfidf_v3.py --split_seed $K || return 1
    done_run v3 score_tfidf_v3 $K || python scripts/score_baselines.py --model tfidf \
        --ckpt checkpoints/v3/tfidf_lr/split_seed_$K.pkl --exp score_tfidf_v3 --seed $K --model_name tfidf_v3_s$K \
        --v3_split_dir data/v3/split_seed_$K --probes data/v3_probes/split_seed_$K.csv || return 1
  done
}
