#!/bin/bash
# ==============================================================================
# SLURM BATCH SUBMISSION SCRIPT: DeBERTa-v3 Multi-Seed & LOTO Training
# ==============================================================================
#
# INSTRUCTIONS FOR RUNNING ON CLUSTER:
# 1. On the login node (with active internet access), pre-download weights:
#      python3 -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; AutoTokenizer.from_pretrained('microsoft/deberta-v3-base'); AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-base', num_labels=2)"
# 2. Create the venv in the repo root (.venv) with a CUDA 12.8 torch build (Blackwell GPUs).
# 3. Submit the job using:
#      sbatch submit_train.sh
# 4. Monitor execution using:
#      squeue -u $USER
#      tail -f logs/slurm_deberta_%j.log
# ==============================================================================

#SBATCH --job-name=deberta_ipi
#SBATCH --output=logs/slurm_deberta_%j.log
#SBATCH --error=logs/slurm_deberta_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# ------------------------------------------------------------------------------
# CLUSTER CONFIGURATION (FILL IN YOUR SITE-SPECIFIC SETTINGS)
# ------------------------------------------------------------------------------
# BITS Hyderabad Sharanga: student accounts may only use this partition
#SBATCH --partition=gpu_rtx_pro_6000_6_csis_hyd

# Student QoS caps each job at cpu=4, mem=80G, mps=50 (half a card), gpu=0
#SBATCH --account=bits
#SBATCH --qos=gpu_csis_course

# Half an RTX Pro 6000 via MPS (100 MPS units per card); --gres=gpu:N is denied
#SBATCH --gres=mps:50

# ------------------------------------------------------------------------------
# ENVIRONMENT SETUP
# ------------------------------------------------------------------------------
echo "=============================================================================="
echo "Starting DeBERTa Training Job on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "=============================================================================="

# Create log and checkpoint directories
mkdir -p logs
mkdir -p checkpoints
mkdir -p results

# TODO: Load required modules (uncomment and adjust for your HPC cluster)
# module load python/3.10
# module load cuda/12.1

# TODO: Activate your Python virtual environment
# source /path/to/your/venv/bin/activate
source .venv/bin/activate

# Enforce offline HuggingFace mode (compute nodes usually lack egress access)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

# Print environment verification
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA Available: {torch.cuda.is_available()} | Device Count: {torch.cuda.device_count()}')"

# ------------------------------------------------------------------------------
# TRAINING RUNS
# ------------------------------------------------------------------------------
# Each run is skipped if its results JSON already exists, so resubmitting the
# script only fills in missing runs.
SEEDS="42 123 999"
TRAIN_ARGS="--mode train --data_dir data/processed --batch_size 8 --grad_accum_steps 2 --epochs 3 --lr 2e-5"

run() {  # run <results_json> <args...>
    local out_json=$1; shift
    if [ -f "$out_json" ]; then
        echo ">>> Skipping (exists): $out_json"
        return
    fi
    echo ""
    echo ">>> Running: $out_json"
    python src/models/deberta.py $TRAIN_ARGS --output "$out_json" "$@"
}

# 1. Multi-seed standard evaluation
for seed in $SEEDS; do
    run "results/deberta_seed${seed}_results.json" \
        --seed "$seed" --output_dir "checkpoints/deberta_s${seed}"
done

# 2. Leave-one-task-out, every fold x every seed
# Seed 42 keeps the unsuffixed names used by earlier results.
for task in email table code; do
    for seed in $SEEDS; do
        suffix=$([ "$seed" = 42 ] && echo "" || echo "_s${seed}")
        run "results/deberta_loto_${task}${suffix}_results.json" \
            --seed "$seed" --test_task "$task" \
            --output_dir "checkpoints/deberta_loto_${task}${suffix}"
    done
done

# 3. Size-matched control for the TableQA LOTO fold: train on 496 rows (the size
# of the email+code LOTO training set) drawn from all three tasks, test on the
# full test set. Compare its TableQA score with the held-out-TableQA fold.
for seed in $SEEDS; do
    run "results/deberta_sub496_s${seed}_results.json" \
        --seed "$seed" --train_subsample 496 --output_dir "checkpoints/deberta_sub496_s${seed}"
done

echo ""
echo "=============================================================================="
echo "All DeBERTa multi-seed and LOTO training runs complete at $(date)"
echo "=============================================================================="
