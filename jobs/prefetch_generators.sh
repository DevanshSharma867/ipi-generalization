#!/bin/bash
#SBATCH --job-name=prefetch_gen
#SBATCH --partition=gpu_rtx_pro_6000_6_csis_hyd
#SBATCH --account=bits
#SBATCH --qos=gpu_csis_course
#SBATCH --gres=mps:50          # unused: the QoS rejects jobs without it; the download is network only
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_prefetch_gen_%j.log
# ==============================================================================
# Download the intent-pair generators (paper_plan A18, A24) at their configs/detectors.yaml revisions into HF_HOME
# on scratch: gen-eval-gemma (google/gemma-3-27b-it @ 005ad34, ~55 GB) and gen-train-mistral
# (mistralai/Mistral-7B-Instruct-v0.3 @ c170c70, ~29 GB). The token is read from HF_TOKEN_PATH (home), never copied.
# Resumable: resubmit after a timeout or failure; snapshot_download skips finished files. Fails unless both are
# complete. Submit from the repo root:
#   sbatch jobs/prefetch_generators.sh
# ==============================================================================
set -euo pipefail
echo "=== job ${SLURM_JOB_ID:-local} on $(hostname) at $(date); commit $(git rev-parse HEAD)"
for u in https://huggingface.co https://cdn-lfs.hf.co https://cas-bridge.xethub.hf.co; do
  code=$(curl -sS -m 20 -o /dev/null -w '%{http_code}' "$u" 2>/dev/null || true)
  echo "net $u ${code:-000}"; [ "${code:-000}" != 000 ] || { echo "UNREACHABLE from $(hostname): $u"; exit 3; }
done
source jobs/lib/scratch.sh
touch_scratch
mkdir -p logs
export HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1
echo "HF_HOME=$HF_HOME HF_TOKEN_PATH=$HF_TOKEN_PATH"
df -h "$HF_HOME" | tail -1

NAMES="gen-eval-gemma gen-train-mistral"
out=$(.venv/bin/python eval/transfer/prefetch.py fetch --only $NAMES | tee /dev/stderr)
for n in $NAMES; do
  grep -Eq "^$n +granted +complete" <<<"$out" || { echo "NOT COMPLETE: $n (resubmit to resume)"; exit 1; }
done
echo "=== both generators complete at $(date)"
