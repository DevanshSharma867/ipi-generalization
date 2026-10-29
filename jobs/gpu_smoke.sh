#!/bin/bash
# ==============================================================================
# SLURM: 2-minute GPU smoke test before any real transfer-harness job. Reports the GPU, the MPS share and
# VRAM cap seen by CUDA, and allocates / frees memory with torch. Writes nothing outside logs/.
#   sbatch jobs/gpu_smoke.sh
# ==============================================================================
#SBATCH --job-name=ipi_gpu_smoke
#SBATCH --output=logs/slurm_gpu_smoke_%j.log
#SBATCH --error=logs/slurm_gpu_smoke_%j.err
#SBATCH --time=00:08:00
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
env | grep -E '^CUDA_(VISIBLE_DEVICES|MPS_)' | sort
cat > "${TMPDIR:-/tmp}/smoke_$$.py" <<'PY'
import time
t0 = time.time()
print("python start", flush=True)
import torch
print(f"import torch {time.time() - t0:.1f}s; torch {torch.__version__} cuda {torch.version.cuda}", flush=True)
t1 = time.time()
ok = torch.cuda.is_available()
print(f"cuda available {ok} after {time.time() - t1:.1f}s", flush=True)
free, total = torch.cuda.mem_get_info()
print(f"device {torch.cuda.get_device_name(0)} cc {torch.cuda.get_device_capability(0)}; "
      f"mem_get_info free {free / 2**30:.1f} GiB total {total / 2**30:.1f} GiB ({time.time() - t1:.1f}s)", flush=True)
x = torch.empty(int(8 * 2**30), dtype=torch.uint8, device="cuda")   # 8 GiB allocation
y = torch.randn(4096, 4096, device="cuda") @ torch.randn(4096, 4096, device="cuda")
torch.cuda.synchronize()
print(f"allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB; matmul ok {bool(torch.isfinite(y).all())}", flush=True)
del x, y
torch.cuda.empty_cache()
print(f"smoke test OK in {time.time() - t0:.1f}s", flush=True)
PY
# Stage 1 without the VRAM cap variable, stage 2 with it, so a hang can be attributed
echo "== without CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"
timeout 120 env -u CUDA_MPS_PINNED_DEVICE_MEM_LIMIT python "${TMPDIR:-/tmp}/smoke_$$.py" || echo "stage 1 exit $?"
echo "== with CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=$CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"
timeout 120 python "${TMPDIR:-/tmp}/smoke_$$.py" || echo "stage 2 exit $?"
rm -f "${TMPDIR:-/tmp}/smoke_$$.py"
