#!/bin/bash
# ==============================================================================
# SLURM: TF-IDF+LR on V3 (paper_plan A30): per split seed 0-4, C in {1,10,100} selected on V3 val
# (scripts/tfidf_v3.py; test never read), then scored on the V3 sets and V3 probes (scripts/score_baselines.py).
# CPU work, but the QoS needs --gres=mps:50. Resumable: finished clean runs are skipped. ~5 s train + ~5 s score per seed locally.
# Submit from the repo root: sbatch jobs/v3_tfidf.sh
# ==============================================================================
#SBATCH --job-name=v3_tfidf
#SBATCH --output=logs/slurm_v3_tfidf_%j.log
#SBATCH --error=logs/slurm_v3_tfidf_%j.log
#SBATCH --time=00:30:00
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
[ -e nltk_data ] || ln -s "$HOME/nlp/nltk_data" nltk_data   # stopwords + wordnet for preprocess_for_tfidf (gitignored)
done_run() {
  python - "$@" <<'PY'
import json, sys
from pathlib import Path
exp, seed = sys.argv[1:]
for d in Path("results", "v3", exp).glob(f"*_seed{seed}"):
    if (d / "finished.json").exists() and not json.loads((d / "config.json").read_text())["git_dirty"]:
        sys.exit(0)
sys.exit(1)
PY
}
for K in 0 1 2 3 4; do
  done_run tfidf_v3_train $K || python scripts/tfidf_v3.py --split_seed $K
  done_run score_tfidf_v3 $K || python scripts/score_baselines.py --model tfidf \
      --ckpt checkpoints/v3/tfidf_lr/split_seed_$K.pkl --exp score_tfidf_v3 --seed $K --model_name tfidf_v3_s$K \
      --v3_split_dir data/v3/split_seed_$K --probes data/v3_probes/split_seed_$K.csv
done
