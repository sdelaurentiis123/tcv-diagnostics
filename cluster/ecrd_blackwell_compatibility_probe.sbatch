#!/bin/bash
# Non-scientific CUDA compatibility probe for Rocky 9 RTX Blackwell nodes.
# This reads no simulation data and cannot generate a scientific artifact.
#SBATCH --job-name=p0_ecrd_bw_probe
#SBATCH --partition=gpupreempt
#SBATCH --qos=gpupreempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtxblackwell
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/ecrd_blackwell_probe_%j.log

set -euo pipefail

readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"

source /etc/os-release
if [[ "${ID}" != "rocky" || "${VERSION_ID%%.*}" != "9" ]]; then
  echo "Blackwell compatibility probe requires Rocky Linux 9" >&2
  exit 2
fi
if [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -ne 1 ]]; then
  echo "Blackwell compatibility probe requires exactly one GPU" >&2
  exit 2
fi
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

"${PYTHON}" - <<'PY'
import json
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("one CUDA device is required")
device = torch.device("cuda", 0)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
x = torch.randn((1, 5, 64, 32, 88), device=device, dtype=torch.float32)
layer = torch.nn.Conv3d(5, 8, kernel_size=3, padding=1).to(device)
y = layer(x)
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    z = layer(x)
torch.cuda.synchronize(device)
if y.shape != (1, 8, 64, 32, 88) or z.shape != y.shape:
    raise RuntimeError("Blackwell Conv3d output shape differs")
if not torch.isfinite(y).all() or not torch.isfinite(z).all():
    raise FloatingPointError("Blackwell Conv3d result is non-finite")
print(json.dumps({
    "scope": "non_scientific_Rocky9_Blackwell_PyTorch_compatibility_probe",
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "device_name": torch.cuda.get_device_name(device),
    "device_capability": list(torch.cuda.get_device_capability(device)),
    "bfloat16_supported": bool(torch.cuda.is_bf16_supported()),
    "float32_Conv3d_finite": True,
    "bfloat16_autocast_Conv3d_finite": True,
    "simulation_data_read": False,
    "scientific_result": False,
}, sort_keys=True))
PY
