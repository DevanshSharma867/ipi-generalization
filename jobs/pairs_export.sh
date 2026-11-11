#!/bin/bash
# ==============================================================================
# SLURM (CPU work; the QoS needs --gres=mps:50 even so): export the self-check-kept TRAIN-side pairs to the files ICT
# reads (scripts/export_pairs.py; paper_plan A22, A32): data/pairs/ict_s0/train_pairs.jsonl and val_pairs.jsonl,
# each with a .manifest.json (counts, input and output sha256, commit). Untracked (gitignored) on the cluster.
# Needs both generate dirs finished and self-checked (jobs/pairs_selfcheck.sh). Idempotent: an identical rerun is a
# no-op; a different existing file is never overwritten.
# Submit from the repo root (~/nlp_harness): sbatch --dependency=afterok:<train selfcheck>:<val selfcheck> jobs/pairs_export.sh
# ==============================================================================
#SBATCH --job-name=pairs_export
#SBATCH --output=logs/slurm_pairs_export_%j.log
#SBATCH --error=logs/slurm_pairs_export_%j.log
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu_rtx_pro_6000_6_csis_hyd
#SBATCH --account=bits
#SBATCH --qos=gpu_csis_course
#SBATCH --gres=mps:50
#SBATCH --requeue
#SBATCH --open-mode=append

source jobs/lib/job_env.sh
python scripts/export_pairs.py --in_root data/pairs/train_gen --out data/pairs/ict_s0/train_pairs.jsonl
python scripts/export_pairs.py --in_root data/pairs/val_gen --out data/pairs/ict_s0/val_pairs.jsonl
echo "done: data/pairs/ict_s0/"
