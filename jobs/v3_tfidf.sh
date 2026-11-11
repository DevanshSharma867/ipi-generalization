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
source jobs/lib/v3_tfidf_steps.sh
v3_tfidf_steps
