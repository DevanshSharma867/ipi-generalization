#!/bin/bash
# ==============================================================================
# SLURM: Gate 2 of the transfer harness, the Liu (2026) reproduction (paper_plan A17, protocol A19).
# Scores PIGuard, Prompt Guard 2 86M, Horizon-Labs and the 4-bit Qwen2.5-7B judge on AD-Clean, AD-Matched, BIPIA
# and TB-Matched three times (Liu's code in his env; our port in his env; our port in ours), then compares with his
# released results (eval/transfer/gate2.py). Score files go to scratch; the run record (with every score file)
# to results/transfer/liu_repro_gate2/ in home. Resumable: resubmit and finished score files are skipped.
# Needs the rebuilt sets (jobs/build_transfer_sets.sh), the prefetched models (scripts/prefetch_models.sh) and
# bitsandbytes in our venv (requirements-hpc.txt). Submit from the repo root:
#   sbatch jobs/gate2_liu_repro.sh
# ==============================================================================
#SBATCH --job-name=gate2_liu
#SBATCH --output=logs/slurm_gate2_liu_%j.log
#SBATCH --error=logs/slurm_gate2_liu_%j.log
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=gpu_rtx_pro_6000_6_csis_hyd
#SBATCH --account=bits
#SBATCH --qos=gpu_csis_course
#SBATCH --gres=mps:50

source jobs/lib/job_env.sh
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv
LIU_PY=$PWD/.venv-liu/bin/python
OUR_PY=$PWD/.venv/bin/python
WORK=$SCRATCH/nlp_harness/gate2
mkdir -p "$WORK"
"$OUR_PY" -c "import bitsandbytes, transformers, torch; print('ours', torch.__version__, transformers.__version__, bitsandbytes.__version__)"
"$LIU_PY" -c "import bitsandbytes, transformers, torch; print('liu ', torch.__version__, transformers.__version__, bitsandbytes.__version__)"

# metric port vs numbers_many.json on Liu's released files first (seconds); nothing is scored if it fails
"$OUR_PY" eval/transfer/gate2.py refcheck
"$OUR_PY" eval/transfer/gate2.py score --work "$WORK" --liu_py "$LIU_PY" --our_py "$OUR_PY"
"$OUR_PY" eval/transfer/gate2.py compare --work "$WORK" --liu_py "$LIU_PY" --our_py "$OUR_PY"
