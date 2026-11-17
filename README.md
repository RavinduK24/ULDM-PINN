# fdm-pinn — Physics-informed neural networks for fuzzy dark matter

A PINN framework for modelling **ultralight (fuzzy) dark matter** candidates via
the **Schrödinger–Poisson (SP) equations**, validated against the established
[PyUltraLight](https://github.com/auckland-cosmology/PyUltraLight) spectral solver.

This repository grew out of a code review of a set of Schrödinger–Poisson PINN
notebooks. The review (in `docs/`) found that those notebooks solved the *wrong
equation in the wrong units* — see [`docs/01_code_review.md`](docs/01_code_review.md).
This repo contains the corrected physics, a unit system that maps every solution
to a physical dark-matter mass, and the harness to generate reference data on the
PolyU HPC cluster.

## Repository layout

```
src/fdm/            physics + diagnostics library
  evolution.py      periodic 1-D Cartesian SPINN
  units.py          physical units, de Broglie parameter, soliton scaling
  diagnostics.py    the quantitative checks from the code review
  reference.py      loader for the external PyUltraLight solver
scripts/
  run_diagnostics.py      re-run the notebook diagnostics
  run_reference.py        run PyUltraLight and emit reference data
  train_evolution.py      train the Cartesian SPINN
hpc/
  reference.sbatch        SLURM job: generate reference data
  pinn.sbatch             SLURM job: train the PINN (placeholder)
tests/                self-contained unit tests (no pytest needed)
docs/                 code review + roadmap
data/                 generated data (git-ignored)
external/             third-party code (git-ignored, see NOTICE.md)
```

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tests/test_units.py
python scripts/run_diagnostics.py
```

Run the 1-D Cartesian periodic problem (no external data required):

```bash
python scripts/train_evolution.py --adam 200 --lbfgs 20 --colloc 2000
```

Results are written to `data/evolution/`: `evolution.png`, `metrics.json`,
`solution.npz`, and `model.pt`. The model uses `L=8`, `lambda=1`, the normalized
wavefunction initial condition `Psi(x,0) = sqrt(2) sin(2 pi x/L)`, and the
background-subtracted Poisson equation. The initial condition and periodic field
and derivative boundary conditions are minimized as explicit soft loss terms,
alongside the Schrödinger and Poisson residual losses. Their sample counts and
weights can be controlled with `--initial-points`, `--boundary-points`,
`--ic-weight`, and `--bc-weight`. Adam training defaults to four expanding
causal time windows and also minimizes global mass drift. Configure these with
`--time-windows`, `--mass-weight`, `--mass-times`, and `--mass-grid`; setting
`--time-windows 1` disables the curriculum and `--mass-weight 0` disables the
mass penalty.

Validate a trained checkpoint against the independent second-order split-step
Fourier solver:

```bash
python scripts/validate_evolution.py --run data/evolution_64x4
```

The validation directory contains relative L2 errors for density, phase-aligned
wavefunction, and zero-mean potential, plus mass-conservation curves and a
side-by-side comparison figure.

## Generating reference data with PyUltraLight

PyUltraLight is third-party code and is **not** distributed here (see
[`NOTICE.md`](NOTICE.md)). Clone it once, then run:

```bash
mkdir -p external && cd external
git clone https://github.com/auckland-cosmology/PyUltraLight.git
cd ..
python scripts/run_reference.py --pyul external/PyUltraLight --out data/reference
```

## Running on the PolyU HPC (SLURM)

The Student HPC Platform (`studenthpc.polyu.edu.hk`) and the COMP cluster
(`hpclogin.comp.polyu.edu.hk`) both use SLURM. After creating a conda
environment from `environment.yml`, submit a batch job:

```bash
# on the Student HPC Platform use `source activate` instead of `conda activate`
sbatch hpc/reference.sbatch
squeue -l
```

See [`docs/02_roadmap.md`](docs/02_roadmap.md) for details.

## Status

- ✅ Physics + units (tested)
- ✅ Notebook diagnostics (verified numbers in `docs/01_code_review.md`)
- ⚠️ Reference harness (PyUltraLight driver written; runs pending cluster time)
- 🛠️ Time-dependent Cartesian SPINN baseline implemented; reference validation pending

Before making this repository public: add a `LICENSE`, and confirm the
notebooks under review are cleared to publish (they are not committed here).

## References

- Mishra & Tolley (2025), *SPINN: Advancing Cosmological Simulations of Fuzzy
  Dark Matter with Physics Informed Neural Networks*, [arXiv:2506.02957](https://arxiv.org/abs/2506.02957).
- Mocz, Vogelsberger, Robles, Smith, Hernquist & Marinacci (2017), *Galaxy
  formation with BECDM*, MNRAS — the PyUltraLight code paper, [arXiv:1807.04037](https://arxiv.org/abs/1807.04037).
