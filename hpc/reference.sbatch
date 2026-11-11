#!/bin/bash
# Generate Schrödinger–Poisson reference data with PyUltraLight on the PolyU HPC.
#
# Platforms (both use SLURM):
#   COMP cluster  : hpclogin.comp.polyu.edu.hk     `module load slurm` first
#   Student HPC   : studenthpc.polyu.edu.hk        conda already available
# Adjust the `module load` / `source activate` lines to your platform if needed.

#SBATCH --job-name=fdm-reference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# --- conda / python environment -----------------------------------------
# COMP cluster: anaconda3 is a module.  Student HPC Platform: conda is on PATH.
module load anaconda3 2>/dev/null || true

ENV_NAME=fdm-pinn
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[setup] creating conda env $ENV_NAME ..."
    conda env create -f environment.yml
fi
# `source activate` works on both PolyU clusters (the student platform does not
# support `conda activate`).
source activate "$ENV_NAME"

# --- fetch the reference code once (git-ignored, see NOTICE.md) ----------
mkdir -p external
if [ ! -f external/PyUltraLight/PyUltraLight.py ]; then
    git clone https://github.com/auckland-cosmology/PyUltraLight.git external/PyUltraLight
fi

# --- run -----------------------------------------------------------------
python scripts/run_reference.py --pyul external/PyUltraLight --out data/reference

echo "[done] reference data in data/reference"
