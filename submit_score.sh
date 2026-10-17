#!/bin/bash
# ==============================================================================
# SLURM: zero-shot scoring of all detectors on BIPIA test and out-of-distribution sets
# ==============================================================================
# Prerequisites (login node, has internet):
#   python score_detectors.py build --dataset pairs|injecagent|deepset   (or rsync data/)
#   HF_HOME=$PWD/hf_cache hf download Qwen/Qwen2.5-7B-Instruct
#   HF_HOME=$PWD/hf_cache hf download protectai/deberta-v3-base-prompt-injection-v2
#   HF_HOME=$PWD/hf_cache hf download leolee99/PIGuard
# Submit:  sbatch submit_score.sh
# ==============================================================================

#SBATCH --job-name=ipi_score
#SBATCH --output=logs/slurm_score_%j.log
#SBATCH --error=logs/slurm_score_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
# BITS Hyderabad Sharanga student QoS: cpu=4, mem=80G, mps=50 (half a card)
#SBATCH --partition=gpu_rtx_pro_6000_6_csis_hyd
#SBATCH --account=bits
#SBATCH --qos=gpu_csis_course
#SBATCH --gres=mps:50

mkdir -p logs
source .venv/bin/activate
export HF_HOME="$PWD/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

score() {  # score <dataset> <model> <checkpoint> <name>; skipped if its scores exist
    [ -f "results/scores/$1/$4.csv" ] && { echo ">>> Skipping (exists): $1/$4"; return; }
    echo ">>> Scoring $1 with $4"
    python score_detectors.py score --dataset "$1" --model "$2" --checkpoint "$3" --name "$4"
}

for ds in bipia_test pairs injecagent deepset; do
    for seed in 42 123 999; do
        score "$ds" deberta "checkpoints/deberta_s${seed}" "deberta_s${seed}"
    done
    score "$ds" hf protectai/deberta-v3-base-prompt-injection-v2 protectai
    score "$ds" hf leolee99/PIGuard piguard
done
score pairs judge Qwen/Qwen2.5-7B-Instruct judge_qwen2.5_7b
score injecagent judge Qwen/Qwen2.5-7B-Instruct judge_qwen2.5_7b
python score_detectors.py summarize
