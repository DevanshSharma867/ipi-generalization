#!/bin/bash
# ==============================================================================
# SLURM: V3 baselines and H1 (paper_plan §7 step 4, §3 H1, A30). One array task per DeBERTa run:
#   0-4  V3 DeBERTa-v3-base, split seed k = training seed k: train on data/v3/split_seed_k (V3 val selection,
#        test never read), then score V3 sets + V3 probes + agent sets (H1).
#   5-6  V2 DeBERTa seeds 0 and 1 on frozen V2 (data/processed; V2 val selection), then score the agent sets (H1).
#   7-9  existing V2 DeBERTa seeds 42, 123, 999 (~/nlp/checkpoints/deberta_s<seed>, reused as-is): score the agent sets.
# Hyperparameters are deberta.py's defaults (lr 2e-5, bs 8x2, 3 epochs, max_len 512). Base model: the
# microsoft/deberta-v3-base snapshot 8ccc9b6 the V2 runs used.
# Resumable: a finished clean run (train or score) is skipped; training resumes from <ckpt>/state.pt (per epoch).
# Submit from the repo root: sbatch jobs/v3_baselines.sh   (wave 1; judges: jobs/v3_judges.sh)
# Time: V2 training took 7-8 min (jobs 399716, 399738); V3 train is ~1.8x larger; scoring ~12k rows.
# ==============================================================================
#SBATCH --job-name=v3_base
#SBATCH --output=logs/slurm_v3_base_%A_%a.log
#SBATCH --error=logs/slurm_v3_base_%A_%a.log
#SBATCH --array=0-9
#SBATCH --time=01:00:00
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
source jobs/lib/v3_baseline_task.sh
BASE=$SCRATCH/hf_cache/hub/models--microsoft--deberta-v3-base/snapshots/8ccc9b6f36199bec6961081d44eb72fb3f7353f3
AGENT_DIR=data/transfer/sets_adliu
V2_CKPT_ROOT=$HOME/nlp/checkpoints
v3_baseline_task "${SLURM_ARRAY_TASK_ID:?array task}"
