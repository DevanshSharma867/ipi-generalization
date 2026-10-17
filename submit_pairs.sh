#!/bin/bash
# ==============================================================================
# SLURM: score query-counterfactual pairs with DeBERTa seeds and an LLM judge
# ==============================================================================
# Prerequisites (login node, has internet):
#   python score_pairs.py build
#   HF_HOME=$PWD/hf_cache hf download Qwen/Qwen2.5-7B-Instruct
# Submit:  sbatch submit_pairs.sh
# ==============================================================================

#SBATCH --job-name=ipi_pairs
#SBATCH --output=logs/slurm_pairs_%j.log
#SBATCH --error=logs/slurm_pairs_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
# BITS Hyderabad Sharanga student QoS: cpu=4, mem=80G, mps=50 (half a card)
#SBATCH --partition=gpu_rtx_pro_6000_6_csis_hyd
#SBATCH --account=bits
#SBATCH --qos=gpu_csis_course
#SBATCH --gres=mps:50

mkdir -p logs results/pairs
source .venv/bin/activate
export HF_HOME="$PWD/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

for seed in 42 123 999; do
    python score_pairs.py score --model deberta --checkpoint "checkpoints/deberta_s${seed}" --name "deberta_s${seed}"
done
python score_pairs.py score --model judge --checkpoint Qwen/Qwen2.5-7B-Instruct --name judge_qwen2.5_7b
python score_pairs.py summarize
