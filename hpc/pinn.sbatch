#!/bin/bash
# Train the PINN (placeholder — see docs/02_roadmap.md step 3).
#
# GPU request differs by platform:
#   COMP cluster  : NVIDIA A800 via MIG, e.g.  --gres=gpu:2g.20gb:1
#   Student HPC   : use the queue's GPU resource name shown by `scontrol show node`

#SBATCH --job-name=fdm-pinn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

module load anaconda3 2>/dev/null || true
ENV_NAME=fdm-pinn
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda env create -f environment.yml
fi
source activate "$ENV_NAME"

# Quick environment smoke test — replace with the real training entry point.
python - <<'PY'
import numpy, scipy, matplotlib, torch
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY

echo "[done]"
