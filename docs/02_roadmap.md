# Roadmap: from the audited notebooks to a correct FDM PINN

## Step 1 — Fix the formulation ✅ (done, `src/fdm/units.py`)

- Use the dimensionless SP system with the background term `∇²V = |Ψ|² − ⟨|Ψ|²⟩`.
- Make `λ = ħ/(m L² √(4πGρ̄))` an explicit physics input.
- Keep the soliton scaling symmetry (`ψ → a²ψ(ax, a²t)`, `V → a³V(ax)`) intact:
  this is what lets a single trained network cover every physical mass.

## Step 2 — Generate ground truth ⚠️ (harness written, cluster run pending)

```bash
# on the cluster (see hpc/reference.sbatch)
sbatch hpc/reference.sbatch
```

Produce, with PyUltraLight in 3-D:

- **stationary soliton** — the PINN must hold it still (this also validates the
  reference setup);
- **two-soliton merger** — non-trivial dynamics, interference fringes, and the
  mass/energy conservation curves;
- the **soliton core–mass relation** `M·r_c = const`, the key observable.

Acceptance: mass drift `≲ 1e-10`, energy drift `≲ 1e-3`, single soliton profile
change `≲ 1e-3`.

## Step 3 — Build the corrected PINN baseline

Start with the *simplest case that is also the real object*: the **3-D radially
reduced** Schrödinger–Poisson equation (a 1+1-D problem), which is validated
directly against PyUltraLight's spherically-symmetric run:

```
i Ψ_t = −½ (Ψ_rr + 2 Ψ_r / r) + V Ψ      (λ = 1 first)
V_rr  + 2 V_r / r = |Ψ|² − ⟨|Ψ|²⟩
BC:   Ψ_r(0,t)=0, V_r(0,t)=0, Ψ(R,t)=0
```

Pin the known failure modes from the review:

| Failure mode in the notebooks | Fix |
|---|---|
| float32 through 2nd derivatives | float64 (or double-precision MLP), mixed-precision only as a later optimisation |
| unweighted heterogeneous loss | gradient-norm / self-adaptive weights per term |
| fixed collocation points | resample every epoch (or RAR) |
| undertrained, tiny tanh net | Fourier features + hard-encoded IC/BC, L-BFGS polish after Adam |
| no conservation | add the norm `∂_t ∫|Ψ|² = 0` as an explicit regulariser / diagnostic |
| wrong Poisson | subtract the mean; the zero mode is the density normalisation, not a DOF |
| no physical input | take `λ` as a network input (parametric PINN) |

## Step 4 — Validate against the reference

Report, for each `λ` and each initial condition:

- relative L2 error in `|Ψ|²` and `V` vs PyUltraLight;
- conservation of mass, energy, and the soliton core–mass relation;
- extrapolation in time (train `[0,T]`, test `[T, 2T]`).

## Step 5 — Scale to 3-D (optional, on GPU)

A full 3-D PINN is a research-level target (cost scales as `(L/r_c)³` collocation
points). On the PolyU cluster request an MIG GPU:

```bash
# COMP cluster (A800, MIG slice)
srun --gres=gpu:2g.20gb:1 --pty bash
```

## PolyU HPC quick reference

- **Student HPC Platform** — portal `studenthpc.polyu.edu.hk`, SLURM, 70 GB quota,
  conda via `source activate <env>`, jobs via `sbatch`/`squeue -l`.
- **COMP cluster** — `hpclogin.comp.polyu.edu.hk` (use Research VPN from off-campus),
  `module load slurm`, `module load anaconda3`, MIG GPUs `--gres=gpu:2g.20gb:N`.
- Create the environment once: `conda env create -f environment.yml`.
- Clone the reference once (git-ignored): `external/PyUltraLight`.
