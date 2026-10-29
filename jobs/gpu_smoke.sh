#!/bin/bash
# ==============================================================================
# SLURM: 2-minute GPU smoke test before any real transfer-harness job. Reports the GPU, the MPS share and
# VRAM cap seen by CUDA, and allocates / frees memory with torch. Writes nothing outside logs/.
#   sbatch jobs/gpu_smoke.sh
# ==============================================================================
#SBATCH --job-name=ipi_gpu_smoke
#SBATCH --output=logs/slurm_gpu_smoke_%j.log
#SBATCH --error=logs/slurm_gpu_smoke_%j.err
#SBATCH --time=00:05:00
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
timeout 90 python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
free, total = torch.cuda.mem_get_info()
print(f"device {torch.cuda.get_device_name(0)} cc {torch.cuda.get_device_capability(0)}; "
      f"mem_get_info free {free / 2**30:.1f} GiB total {total / 2**30:.1f} GiB")
x = torch.empty(int(8 * 2**30), dtype=torch.uint8, device="cuda")   # 8 GiB allocation
y = torch.randn(4096, 4096, device="cuda") @ torch.randn(4096, 4096, device="cuda")
torch.cuda.synchronize()
print(f"allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB; matmul ok {bool(torch.isfinite(y).all())}")
del x, y
torch.cuda.empty_cache()
print("smoke test OK")
PY
