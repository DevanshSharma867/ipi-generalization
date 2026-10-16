#!/bin/bash
# ==============================================================================
# SLURM BATCH SUBMISSION SCRIPT: DeBERTa-v3 Multi-Seed & LOTO Training
# CS F429 NLP Course Project (Phase 2)
# ==============================================================================
#
# INSTRUCTIONS FOR RUNNING ON CLUSTER:
# 1. On the login node (with active internet access), pre-download weights:
#      python3 -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; AutoTokenizer.from_pretrained('microsoft/deberta-v3-base'); AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-base', num_labels=2)"
# 2. Fill in the TODO lines below (partition, account, GPU specification, venv).
# 3. Submit the job using:
#      sbatch submit_train.sh
# 4. Monitor execution using:
#      squeue -u $USER
#      tail -f logs/slurm_deberta_%j.log
# ==============================================================================

#SBATCH --job-name=deberta_ipi_v3
#SBATCH --output=logs/slurm_deberta_%j.log
#SBATCH --error=logs/slurm_deberta_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# ------------------------------------------------------------------------------
# CLUSTER CONFIGURATION (FILL IN YOUR SITE-SPECIFIC SETTINGS)
# ------------------------------------------------------------------------------
# TODO: Specify your cluster partition (e.g., gpu, general, batch)
#SBATCH --partition=TODO_PARTITION

# TODO: Specify your project allocation / account name
#SBATCH --account=TODO_ACCOUNT

# TODO: Specify GPU resource request (e.g., --gres=gpu:1 or --gres=gpu:a100:1 or --gres=gpu:v100:1)
#SBATCH --gres=gpu:1

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
# 1. MULTI-SEED STANDARD EVALUATION (Seeds 42, 123, 999)
# ------------------------------------------------------------------------------
echo ""
echo ">>> [1/6] Running DeBERTa Seed 42 Standard Training..."
python src/models/deberta.py \
    --mode train \
    --data_dir data/processed \
    --output_dir checkpoints/deberta_s42 \
    --seed 42 \
    --batch_size 8 \
    --grad_accum_steps 2 \
    --epochs 3 \
    --lr 2e-5

echo ""
echo ">>> [2/6] Running DeBERTa Seed 123 Standard Training..."
python src/models/deberta.py \
    --mode train \
    --data_dir data/processed \
    --output_dir checkpoints/deberta_s123 \
    --seed 123 \
    --batch_size 8 \
    --grad_accum_steps 2 \
    --epochs 3 \
    --lr 2e-5

echo ""
echo ">>> [3/6] Running DeBERTa Seed 999 Standard Training..."
python src/models/deberta.py \
    --mode train \
    --data_dir data/processed \
    --output_dir checkpoints/deberta_s999 \
    --seed 999 \
    --batch_size 8 \
    --grad_accum_steps 2 \
    --epochs 3 \
    --lr 2e-5

# ------------------------------------------------------------------------------
# 2. LEAVE-ONE-TASK-OUT (LOTO) CROSS-VALIDATION (All 3 Folds)
# ------------------------------------------------------------------------------
echo ""
echo ">>> [4/6] Running LOTO Fold: Held-out EmailQA (Train on Table + Code)..."
python src/models/deberta.py \
    --mode train \
    --data_dir data/processed \
    --test_task email \
    --train_tasks table code \
    --output_dir checkpoints/deberta_loto_email \
    --seed 42 \
    --batch_size 8 \
    --grad_accum_steps 2 \
    --epochs 3 \
    --lr 2e-5

echo ""
echo ">>> [5/6] Running LOTO Fold: Held-out TableQA (Train on Email + Code)..."
python src/models/deberta.py \
    --mode train \
    --data_dir data/processed \
    --test_task table \
    --train_tasks email code \
    --output_dir checkpoints/deberta_loto_table \
    --seed 42 \
    --batch_size 8 \
    --grad_accum_steps 2 \
    --epochs 3 \
    --lr 2e-5

echo ""
echo ">>> [6/6] Running LOTO Fold: Held-out CodeQA (Train on Email + Table)..."
python src/models/deberta.py \
    --mode train \
    --data_dir data/processed \
    --test_task code \
    --train_tasks email table \
    --output_dir checkpoints/deberta_loto_code \
    --seed 42 \
    --batch_size 8 \
    --grad_accum_steps 2 \
    --epochs 3 \
    --lr 2e-5

echo ""
echo "=============================================================================="
echo "All DeBERTa multi-seed and LOTO training runs complete at $(date)"
echo "=============================================================================="
