# Fail fast on a bad GPU instead of crashing mid-run. Sourced by jobs/lib/job_env.sh (after scratch.sh and the venv),
# so every GPU job runs it. Needs python with torch on PATH.
# Exits non-zero if the 1-tensor CUDA alloc + sync fails or the job's GPU is a known-bad card.
# Exports GPU_UUID; src/runlog.py records it in config.json (hardware.gpu_uuid).
# Known-bad cards (uncorrectable ECC; jobs 429434, 429443 on gpunode8, 2026-11-02):
BAD_GPU_UUIDS="GPU-cfb39ac3-d342-8391-b482-bf2ce8a41587"

GPU_UUID=$(timeout 120 python - <<'PY'
import torch
x = torch.ones(1, device="cuda")
torch.cuda.synchronize()
assert x.item() == 1.0
print(f"GPU-{torch.cuda.get_device_properties(0).uuid}")
PY
) || { echo "gpu_guard: 1-tensor CUDA test failed on $(hostname) (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}); not starting." >&2; exit 3; }
if [[ " $BAD_GPU_UUIDS " == *" $GPU_UUID "* ]]; then
  echo "gpu_guard: job landed on known-bad GPU $GPU_UUID on $(hostname); not starting." >&2; exit 3
fi
export GPU_UUID
echo "gpu_guard: OK, GPU $GPU_UUID on $(hostname)"
