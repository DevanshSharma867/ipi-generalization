#!/bin/bash
# ==============================================================================
# SLURM: score the V2 position probes (data/probes_v2/) with the GPU detectors.
# ==============================================================================
# Writes standard-schema predictions via runlog to results/v2_probes/score_<name>/<run_id>/.
# DeBERTa: uses checkpoints/deberta_s<seed> (the path submit_train.sh wrote) if it exists;
# otherwise retrains with the identical V2 config and seed into checkpoints/v2_retrain/deberta_s<seed>
# (persistent; the path and origin are recorded in each scoring run's config.json).
# Each scoring step is skipped if its results directory already holds a predictions.csv.
#
# Prerequisites (login node): repo at the commit that added data/probes_v2/; hf_cache holds
# microsoft/deberta-v3-base, protectai/deberta-v3-base-prompt-injection-v2, leolee99/PIGuard,
# Qwen/Qwen2.5-7B-Instruct (as for submit_score.sh).
# Submit from the repo root:  sbatch slurm/probe_v2.sh
# ==============================================================================

#SBATCH --job-name=ipi_probe_v2
#SBATCH --output=logs/slurm_probe_v2_%j.log
#SBATCH --error=logs/slurm_probe_v2_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
# BITS Hyderabad Sharanga student QoS: cpu=4, mem=80G, mps=50 (half a card)
#SBATCH --partition=gpu_rtx_pro_6000_6_csis_hyd
#SBATCH --account=bits
#SBATCH --qos=gpu_csis_course
#SBATCH --gres=mps:50

# ------------------------------------------------------------------------------
# SITE SETTINGS
# ------------------------------------------------------------------------------
MODULES=""                               # e.g. "cuda/12.8"; empty = none
VENV_PATH=".venv"
SEEDS="42 123 999"
JUDGE_DOC_FRAC="${JUDGE_DOC_FRAC:-1.0}"  # fraction of docs the judge scores in P2/P3 (0.25 if compute is tight)

set -euo pipefail
mkdir -p logs results/v2_probes/deberta_retrain checkpoints/v2_retrain
for m in $MODULES; do module load "$m"; done
source "$VENV_PATH/bin/activate"
export HF_HOME="$PWD/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
echo "Job $SLURM_JOB_ID on $(hostname) at $(date); commit $(git rev-parse HEAD)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

done_already() { compgen -G "results/v2_probes/score_$1/*/predictions.csv" > /dev/null; }

score() {  # score <name> <args...>
    local name=$1; shift
    if done_already "$name"; then echo ">>> Skipping (exists): $name"; return; fi
    echo ">>> Scoring $name"
    python scripts/score_probes.py --name "$name" "$@"
}

# 1. DeBERTa V2, 3 seeds
for seed in $SEEDS; do
    ckpt="checkpoints/deberta_s${seed}"
    origin="hpc_original"
    if [ ! -f "$ckpt/config.json" ]; then
        ckpt="checkpoints/v2_retrain/deberta_s${seed}"
        origin="retrained_identical_config"
        if [ ! -f "$ckpt/config.json" ]; then
            echo ">>> Retraining DeBERTa seed $seed (identical V2 config) -> $ckpt"
            python src/models/deberta.py --mode train --data_dir data/processed --batch_size 8 \
                --grad_accum_steps 2 --epochs 3 --lr 2e-5 --seed "$seed" --output_dir "$ckpt" \
                --output "results/v2_probes/deberta_retrain/s${seed}_results.json"
        fi
    fi
    score "deberta_s${seed}" --model deberta --checkpoint "$ckpt" --checkpoint_origin "$origin" --seed "$seed"
done

# 2. Public detectors (untrusted text only, as in score_detectors.py)
score protectai --model hf --checkpoint protectai/deberta-v3-base-prompt-injection-v2 --checkpoint_origin hf_hub
score piguard --model hf --checkpoint leolee99/PIGuard --checkpoint_origin hf_hub

# 3. Qwen2.5-7B judge with the frozen score_detectors.JUDGE_SYSTEM prompt
score judge_qwen2.5_7b --model judge --checkpoint Qwen/Qwen2.5-7B-Instruct --checkpoint_origin hf_hub \
    --sets p1_neutral_insertion p4_whitespace_only p2_benign_moved p3_attack_moved --doc_frac "$JUDGE_DOC_FRAC"

echo "Done at $(date)"
