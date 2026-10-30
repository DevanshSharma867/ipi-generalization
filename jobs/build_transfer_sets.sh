#!/bin/bash
#SBATCH --job-name=build_transfer
#SBATCH --partition=gpu_rtx_pro_6000_6_csis_hyd
#SBATCH --account=bits
#SBATCH --qos=gpu_csis_course
#SBATCH --gres=mps:50          # unused: the QoS rejects jobs without it; the build is CPU only
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/csisnlp_18/nlp_harness/logs/build_transfer_sets.log
#SBATCH --open-mode=append
# ==============================================================================
# Transfer harness step a on the cluster: rebuild the Liu (2026) sets (scripts/build_transfer_sets.sh, venv
# creation included) and run Gate 1 (eval/transfer/gate1.py, logged to results/transfer/set_build_gate1/ in home).
# Venvs, caches and the rebuilt sets live on scratch (jobs/lib/scratch.sh). Resumable: resubmit after a failure
# and finished steps are skipped. Submit from the repo root:
#   sbatch jobs/build_transfer_sets.sh
# ==============================================================================
set -euo pipefail
echo "=== job ${SLURM_JOB_ID:-local} on $(hostname) at $(date); commit $(git rev-parse HEAD)"
# the build clones, installs and streams HF datasets: fail now if any host it downloads from is unreachable
# (any HTTP status counts as reachable; 000 means no connection)
for u in https://github.com https://huggingface.co https://cdn-lfs.hf.co https://cas-bridge.xethub.hf.co \
         https://pypi.org/simple/uv/ https://files.pythonhosted.org https://download.pytorch.org/whl/cu128/; do
  code=$(curl -sS -m 20 -o /dev/null -w '%{http_code}' "$u" 2>/dev/null || true)
  echo "net $u ${code:-000}"; [ "${code:-000}" != 000 ] || { echo "UNREACHABLE from $(hostname): $u"; exit 3; }
done
source jobs/lib/scratch.sh
check_home_free 2
touch_scratch
mkdir -p "$VENVS"
export VENV_AD=$VENVS/agentdojo VENV_LIU=$VENVS/liu PYTHONUNBUFFERED=1
# repo-local names point at the scratch venvs (later harness steps use them); never over a real directory
for pair in ".venv-agentdojo:$VENV_AD" ".venv-liu:$VENV_LIU"; do
  link=${pair%%:*}; [ -e "$link" ] && [ ! -L "$link" ] && { echo "$link is a real directory"; exit 1; }
  ln -sfn "${pair#*:}" "$link"
done

# our venv (system Python 3.11 base, so it survives the move) goes to scratch behind a symlink; the home copy is
# deleted only after the moved one imports torch and transformers
if [ ! -L .venv ]; then
  rsync -a --delete .venv/ "$VENVS/nlp_harness/"
  mv .venv .venv.moving && ln -s "$VENVS/nlp_harness" .venv
  if .venv/bin/python -c "import torch, transformers"; then rm -rf .venv.moving
  else rm .venv && mv .venv.moving .venv && { echo "moved venv fails to import; restored"; exit 1; }; fi
fi
.venv/bin/python -c "import torch, transformers; print('venv', torch.__file__, torch.__version__, transformers.__version__)"

# uv is installed only on the login node, and compute nodes cannot reach GitHub's release-asset host: the same
# version's release binary is downloaded and checksum-verified on the login node into $SCRATCH/tools (commands in the
# header of scripts/build_transfer_sets.sh)
command -v uv >/dev/null || export PATH="$SCRATCH/tools/uv-0.13.1:$PATH"
command -v uv >/dev/null || { echo "no uv: fetch it on the login node first (scripts/build_transfer_sets.sh header)"; exit 1; }
uv --version

bash scripts/build_transfer_sets.sh
.venv/bin/python eval/transfer/gate1.py
