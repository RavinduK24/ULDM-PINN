# Code review: the coupled Schrödinger–Poisson PINN notebooks

**Status:** diagnosis complete and quantitatively verified; the corrected
re-implementation is tracked in [`02_roadmap.md`](02_roadmap.md).

**Reviewed files** (in `Project Docs/`):

| Notebook | Framework | Claimed problem |
|---|---|---|
| `SchrodingerNewton_1D_Coupled.ipynb` | TensorFlow/Keras | 1-D, two coupled wavefunctions + potential |
| `SchrodingerNewton_1D.ipynb` | TensorFlow/Keras | 1-D, one wavefunction |
| `SchrodingerNewton_1D-jason.ipynb` | TensorFlow/Keras | 1-D, one wavefunction |
| `SchrodingerNewton_2D.ipynb` | TensorFlow/Keras | 2-D |
| `SchrodingerNewton_3D.ipynb` | TensorFlow/Keras | 3-D |
| `Schrodinger_poisson_with_DeepXDE_.ipynb` | DeepXDE/PyTorch | 1-D |
| `Schrodinger_Poisson.ipynb` | PyTorch (Madelung form) | "1-D radial" with a dummy angle |
| `Copy_of_Schrodinger_Poisson_1.ipynb` | PyTorch | 1-D |

---

## 1. Summary verdict

Every notebook solves a **different, undocumented equation in dimensionless
units with no connection to a physical dark-matter particle**. The most severe
problems are in the *physics* (wrong Poisson problem, spurious self-interaction,
ignored dimensions), not in the neural-network tuning. None of the outputs can
be mapped to an ultralight-DM candidate.

| # | Defect | Severity | Where |
|---|---|---|---|
| P1 | `V_xx = |ψ|²` with periodic `V` has **no solution** (needs `⟨ρ⟩ = 0`) | fatal | all notebooks |
| P2 | 1-D Cartesian Poisson is a *sheet* geometry, not 3-D gravity; the result depends on the arbitrary box length | fatal | all 1-D/2-D/3-D notebooks |
| P3 | The 3-D notebook **ignores the z-derivative** (it is 2-D in disguise) | fatal | `SchrodingerNewton_3D` |
| P4 | A quartic `|ψ|²ψ` self-interaction is added, absent from SP and from its own sibling notebook; it breaks the soliton scaling symmetry | fatal | `_1D_Coupled` |
| P5 | No dimensionful units anywhere → no `λ`, no particle mass | fatal | all notebooks |
| P6 | DeepXDE Poisson source uses `r² − i²` instead of `|ψ|² = r² + i²`, and two Schrödinger residuals disagree by a factor 2 | fatal | DeepXDE |
| P7 | "Gravity" is a spatially **constant** force (trapz over an unsorted batch → scalar) | fatal | `Schrodinger_Poisson` (Madelung) |
| N1 | `float32` through second derivatives | severe | TF notebooks |
| N2 | Loss is an unweighted sum of heterogeneous terms | severe | all |
| N3 | Fixed collocation points, no resampling | severe | TF notebooks |
| N4 | Grossly undertrained (1000 steps), tiny net (8×20 tanh) | severe | TF notebooks |
| N5 | No mass/energy conservation constraint or diagnostic | severe | all |
| N6 | Periodic BC imposed on a *localised* soliton + an IC that is not a soliton | moderate | all |
| E1 | Dead code (`fun_u_b`, `u_b`, `output_layer`), copy-paste bugs | moderate | all |
| E2 | Plot/animation bugs (wrong domain, same frame every t) | minor | `Copy_of_...`, TF notebooks |

---

## 2. What each notebook actually solves

Decoding the residual functions (the only statement of "the PDE"):

**`SchrodingerNewton_1D_Coupled`** — two fields `ψ_j = u_j + i v_j`, one potential `Ψ`:

```
u_t + 0.5 v_xx + |ψ|² v − Ψ v = 0
v_t − 0.5 u_xx − |ψ|² u + Ψ u = 0      (same for the second pair)
Ψ_xx − |ψ1|² − |ψ2|² = 0
```

i.e. `i ψ_t = −½ ψ_xx + (Ψ − |ψ|²) ψ`. The `−|ψ|²ψ` term is a quartic
self-interaction that does not appear in the FDM Schrödinger–Poisson system
(`i ψ_t = −½ ψ_xx + Ψ ψ`), and its sibling `SchrodingerNewton_1D` has it switched
**off** (`0*h2*v`) — evidence the author knew it was wrong but left it in the
"coupled" file. Two identical initial conditions are used, so "coupling" is only
through the shared Poisson potential.

**`SchrodingerNewton_1D` / `_1D-jason`** — kinetic coefficient differs between the
two files (`0.5 v_xx` vs `1.0 v_xx`), and `_1D-jason` adds an ad-hoc gauge term
`V(t,0) = −1`. So three of five notebooks disagree on `ħ/2m` by a factor 2.

**`SchrodingerNewton_2D`** — drops the ½ (`u_t + (v_xx+v_yy) + ...`), a different
equation again.

**`SchrodingerNewton_3D`** — the residual function receives `u_zz, v_zz, Ψ_zz` and
**never uses them**:

```python
return u_t + (v_xx + v_yy) + h2*v - Psi*v, \
       v_t - (u_xx + u_yy) - h2*u + Psi*u, \
       (Psi_xx + Psi_yy) - h2          # ← z absent
```

so the z-axis is unconstrained: any z-independent solution of the 2-D system is a
"solution", and the network is trained to be flat in z by nothing.

**`Schrodinger_poisson_with_DeepXDE_`** — four "PDEs", of which `f3 = V_xx − 4πG(r²−i²)`
uses the real part of `ψ²` instead of `|ψ|²`, and `f2` has `0.25 r_xx` against
`f1`'s `0.5 i_xx`. `f4 = i r_xx − r i_xx` is not part of the SP system. The IC
`exp(−x)` on `[−5,5]` jumps by a factor `e^10 ≈ 2.2e4` across the periodic wrap.

**`Schrodinger_Poisson` (Madelung)** — network `forward()` returns the first two
hidden activations `x[:,0], x[:,1]` instead of `output_layer(x)`; the IC is
`rho = exp(2x²)` (a *density dip*, min at the centre); the "potential" is
`phi_x = 4πG · trapz(rho, dx=0.001, dim=0)` — a **single scalar** broadcast over the
whole batch, i.e. a uniform acceleration, with a bias of `dx·N/L = 5×`.

---

## 3. Verified quantitative diagnostics

These numbers are produced by `python scripts/run_diagnostics.py` (numpy only).

### 3.1 Poisson compatibility floor

For periodic `V`, the equation `V'' = ρ` is solvable only if `⟨ρ⟩ = 0`; otherwise
the residual `r = V'' − ρ` has `⟨r⟩ = −⟨ρ⟩` for *any* `V`, so
`‖r‖ ≥ |⟨ρ⟩|`. Measured for the eight initial conditions:

| Notebook | `⟨ρ⟩` | `rms(ρ)` | irreducible `‖r‖` floor | `floor / rms(ρ)` |
|---|---:|---:|---:|---:|
| coupled_1D (2×0.06 sech x) | +7.2e-04 | 1.86e-03 | 7.2e-04 | 39 % |
| SNewton_1D (sech x) | +0.40 | 0.73 | 0.40 | 55 % |
| SNewton_1D-jason (2 exp(−x²)) | +0.50 | 1.19 | 0.50 | 42 % |
| Copy_of_SP_1 (sech x) | +0.10 | 0.26 | 0.10 | 39 % |
| SNewton_2D (3 exp(…)) | +0.44 | 1.41 | 0.44 | 31 % |
| SNewton_3D (3 exp(−r²)) | +0.88 | 1.99 | 0.88 | 44 % |
| DeepXDE (exp(−x)) | +1101 | 3483 | 1101 | 32 % |
| Madelung (exp(2x²)) | +2.36 | 2.87 | 2.36 | 82 % |

In Fourier space this is the `k = 0` divergence of the Poisson Green function
`V̂ = −ρ̂/k²`: the constant mode makes the linear system singular/inconsistent.
The correct handling — subtract the mean (drop the `k = 0` mode) — is exactly what
both SPINN (Eq. 5: `∇²V = |Ψ|² − 1`) and PyUltraLight (`phik[0,0,0] = 0`) do, and
what no notebook does.

### 3.2 1-D sheet gravity is box-dominated

For one Gaussian lump, the 1-D periodic potential depth grows linearly with the
box length, while the 3-D Newtonian potential `Φ(0) = −GM/r_c` is box-independent:

| L | 1-D potential depth | depth × L |
|---|---:|---:|
| 5 | 0.290 | 1.45 |
| 10 | 0.919 | 9.19 |
| 20 | 2.03 | 40.5 |
| 40 | 4.15 | 165.9 |
| 80 | 8.34 | 667.3 |

Equivalently, the 1-D "force" `V'` does **not** decay with distance (it is the field
of an infinite sheet), while the 3-D force `∝ M(<r)/r²`. A 1-D Cartesian Poisson
solve therefore cannot represent a compact object; it is only meaningful as a
*planar cosmological* model (with the mean density subtracted). Modelling a
soliton/star/halo requires the 3-D (or 3-D radially-reduced) Poisson operator.

### 3.3 The ground state the PINN should reproduce

The repository already contains the correct target: PyUltraLight's radial soliton
profile `Soliton Profile Files/initial_f.npy` (a solution of the SP ODE in code
units `ħ = m = 1, 4πG = 1`). Its dimensionless mass `M_code = 3.883` and half-density
radius `r_half = 0.690`, giving the invariant

```
M · r_half = 2.679   (dimensionless;  PyUltraLight uses 5.35854 / M as its size)
```

which reproduces the Schive–Yu–Chiueh profile `ρ = ρ_c [1 + 0.091 (r/r_c)²]^{-8}` to
< 0.5 % across `0 ≤ r ≤ 3 r_c`. In physical units (`r_c = K ħ²/(G m² M)`,
`K = 2.679`, `ρ_c = 0.0846 M/r_c³`):

| m [eV] | M [M☉] | r_c [pc] | ρ_c [M☉/pc³] |
|---|---:|---:|---:|
| 1e-22 | 1e7 | 2.29e4 | 4.2e-7 |
| 1e-22 | 1e9 | 2.29e2 | 41.7 |
| 1e-22 | 1e10 | 2.29e1 | 4.2e5 |
| 1e-21 | 1e9 | 2.29e0 | 4.2e7 |
| 1e-20 | 1e9 | 2.29e-2 | 4.2e13 |

The soliton is the single object whose *existence and mass–radius relation* is the
observable FDM signature — and the notebooks never touch it.

---

## 4. What the correct formulation looks like

**Equation (dimensionless, SPINN convention):**

```
i ∂_t Ψ = − (λ/2) ∇²Ψ + (1/λ) V Ψ
∇²V     = |Ψ|² − ⟨|Ψ|²⟩
```

with the normalisation `⟨|Ψ|²⟩ = 1` and periodic boundary conditions. The **single
dimensionless parameter**

```
λ = ħ / ( m L² √(4πG ρ̄) )
```

is what encodes the particle mass `m` for a box size `L` and background density
`ρ̄`. For the fiducial `m = 10⁻²² eV`, `L = 1 kpc`, `ρ̄ = 0.4 GeV/cm³`,
`λ ≈ 0.8` (computed in `fdm.units.de_broglie_parameter`). `λ → ∞` is the free
Schrödinger equation, `λ → 0` is strongly self-gravitating. A PINN that is to
"model ultralight DM candidates" must take `λ` as its physics input and be trained
over a range of `λ` (a *parametric* PINN), because one network trained at `λ = 1`
carries no mass information.

**Reference solver:** PyUltraLight (3-D, spectral, mass/energy conserving —
measured mass drift `≈ 2e-14`). `scripts/run_reference.py` drives it; the corrected
PINN is validated against its soliton profile and merger evolution.

---

## 5. References

- Mishra & Tolley (2025), *SPINN*, [arXiv:2506.02957](https://arxiv.org/abs/2506.02957).
- Mocz et al. (2017), *Galaxy formation with BECDM*, [arXiv:1807.04037](https://arxiv.org/abs/1807.04037) — PyUltraLight.
- Schive, Chiueh & Broadhurst (2014), *Cosmic structure as the quantum interference
  of a coherent dark wave*, Nature Phys. 10, 496.
- Hui, Ostriker, Tremaine & Witten (2017), *Ultralight scalars as cosmological dark
  matter*, [arXiv:1610.08297](https://arxiv.org/abs/1610.08297).
