# Shared setup for sbatch jobs in jobs/ (source it right after the #SBATCH header).
#
# Standard header (BITS Sharanga, course account; QoS gpu_csis_course allows per job cpu=4, mem=80G, mps=50):
#   #SBATCH --partition=gpu_rtx_pro_6000_6_csis_hyd
#   #SBATCH --account=bits
#   #SBATCH --qos=gpu_csis_course
#   #SBATCH --gres=mps:50          # half a card's compute (MPS active-thread share)
#   #SBATCH --cpus-per-task=4
#   #SBATCH --mem=48G
#   #SBATCH --nodes=1
#   #SBATCH --ntasks=1
# VRAM is not capped by the gres; CUDA_MPS_PINNED_DEVICE_MEM_LIMIT below caps it at half the card (48G).
#
# Variables (set before sourcing): VENV_PATH (default .venv), MODULES (default none), ALLOW_DIRTY=1 to skip the
# clean-tree check (runlog refuses dirty trees anyway unless --allow-dirty is passed).

set -euo pipefail
mkdir -p logs
for m in ${MODULES:-}; do module load "$m"; done
source "${VENV_PATH:-.venv}/bin/activate"
export HF_HOME="$PWD/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export CUDA_MPS_PINNED_DEVICE_MEM_LIMIT="0=48G"
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date); commit $(git rev-parse HEAD); python $(command -v python)"

if [ -z "${ALLOW_DIRTY:-}" ]; then
  python -c "import sys; sys.path.insert(0, 'src'); import runlog; sys.exit(runlog.code_dirty())" \
    || { echo "Dirty tree; commit, stash or exclude these first:"; git status --porcelain | grep -v ' results/'; exit 1; }
fi
