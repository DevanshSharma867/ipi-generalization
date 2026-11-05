# Fail fast on a bad GPU instead of crashing mid-run. Sourced by jobs/lib/job_env.sh (after scratch.sh and the venv),
# so every GPU job runs it. Needs python with torch on PATH.
# Exits 3 if the 1-tensor CUDA alloc + sync fails. On a known-bad card it requeues the job (`scontrol requeue`, the
# job needs #SBATCH --requeue and --open-mode=append so every attempt's line stays in the log) and exits 0, at most
# MAX_GPU_REQUEUES times (SLURM_RESTART_COUNT), then exits 3.
# Exports GPU_UUID; src/runlog.py records it in config.json (hardware.gpu_uuid).
# Known-bad cards (uncorrectable ECC; jobs 429434, 429443 on gpunode8, 2026-11-02; still in service 2026-11-05,
# jobs 432581, 432585):
BAD_GPU_UUIDS="GPU-cfb39ac3-d342-8391-b482-bf2ce8a41587"

GPU_UUID=$(timeout 120 python - <<'PY'
import torch
x = torch.ones(1, device="cuda")
torch.cuda.synchronize()
assert x.item() == 1.0
print(f"GPU-{torch.cuda.get_device_properties(0).uuid}")
PY
) || { echo "gpu_guard: 1-tensor CUDA test failed on $(hostname) (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}); not starting." >&2; exit 3; }
MAX_GPU_REQUEUES=5
attempt=${SLURM_RESTART_COUNT:-0}
echo "gpu_guard: attempt $attempt, GPU $GPU_UUID on $(hostname) at $(date)"
if [[ " $BAD_GPU_UUIDS " == *" $GPU_UUID "* ]]; then
  if [ -n "${SLURM_JOB_ID:-}" ] && [ "$attempt" -lt "$MAX_GPU_REQUEUES" ]; then
    echo "gpu_guard: known-bad GPU $GPU_UUID; requeueing job $SLURM_JOB_ID (requeue $((attempt + 1))/$MAX_GPU_REQUEUES)." >&2
    scontrol requeue "$SLURM_JOB_ID" && exit 0
    echo "gpu_guard: scontrol requeue failed; not starting." >&2; exit 3
  fi
  echo "gpu_guard: known-bad GPU $GPU_UUID on $(hostname) after $attempt requeues (max $MAX_GPU_REQUEUES); giving up. Resubmit later or ask the admins to drain the card." >&2
  exit 3
fi
export GPU_UUID
echo "gpu_guard: OK, GPU $GPU_UUID on $(hostname)"
