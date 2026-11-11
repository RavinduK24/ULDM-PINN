"""
Quantitative diagnostics from the code review of the Schrödinger–Poisson PINN
notebooks.  Each function returns plain dicts so the numbers can be re-verified
with nothing but numpy.

The headline result of every check below:

  1. `V_xx = rho` with a *periodic* V has no solution unless <rho> = 0 exactly,
     and its residual cannot be pushed below |<rho>|.
  2. 1-D Cartesian Poisson is a *sheet* geometry: the force does not fall off,
     and the potential depth is set by the (arbitrary) box length.
  3. The "trapz over a random batch" used by one notebook is a scalar, i.e. a
     spatially constant gravitational force, biased by dx * N / L.
"""
from __future__ import annotations

import os

import numpy as np


def _trapz(y, x):
    return float(np.trapezoid(y, x))


# --------------------------------------------------------------------------- #
# 1. Poisson compatibility
# --------------------------------------------------------------------------- #
def poisson_compatibility(rho: np.ndarray) -> dict:
    """
    For the periodic equation V'' = rho, the residual r = V'' - rho has
    <r> = -<rho> for every V, so the least-squares floor is |<rho>|.
    Returns floor (absolute) and floor_rel = |<rho>| / rms(rho).
    """
    mean = float(rho.mean())
    rms = float(np.sqrt((rho**2).mean()))
    return dict(rho_mean=mean, rho_rms=rms, rho_max=float(rho.max()),
                floor=abs(mean), floor_rel=abs(mean) / rms)


def notebook_cases() -> list[dict]:
    """The eight initial conditions from the notebooks under review."""
    cases = []
    x = np.linspace(-10, 10, 200001)
    cases.append(dict(name="coupled_1D (2 x 0.06 sech x, L=20)",
                      rho=2 * (0.06 / np.cosh(x))**2))
    x = np.linspace(-5, 5, 200001)
    cases.append(dict(name="SNewton_1D (u=v=sech x, L=10)",
                      rho=2 * (1 / np.cosh(x))**2))
    cases.append(dict(name="SNewton_1D-jason (2 exp(-x^2), L=10)",
                      rho=(2 * np.exp(-x**2))**2))
    x = np.linspace(-10, 10, 200001)
    cases.append(dict(name="Copy_of_SP_1 (sech x, L=20)",
                      rho=(1 / np.cosh(x))**2))
    g = np.linspace(-2, 2, 2001)
    XX, YY = np.meshgrid(g, g, indexing="ij")
    cases.append(dict(name="SNewton_2D (3exp(-x^2-4y^2), 4x4 box)",
                      rho=(3 * np.exp(-(XX**2 + 4 * YY**2)))**2))
    cases.append(dict(name="SNewton_3D (3exp(-r^2), 4^3 box)",
                      rho=(3 * np.exp(-(XX**2 + YY**2)))**2))
    x = np.linspace(-5, 5, 200001)
    cases.append(dict(name="DeepXDE (exp(-x), L=10)", rho=np.exp(-2 * x)))
    x = np.linspace(-1, 1, 200001)
    cases.append(dict(name="Madelung_torch (exp(2x^2), L=2)",
                      rho=np.exp(2 * x**2)))
    return cases


# --------------------------------------------------------------------------- #
# 2. 1-D sheet vs 3-D Newtonian gravity
# --------------------------------------------------------------------------- #
def _cumtrapz(y, x):
    return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))])


def one_d_vs_three_d(sigma: float = 1.0, L: float = 20.0) -> dict:
    """Solve the Poisson problem for one Gaussian lump in 1-D (periodic sheet)
    and in 3-D (radial), and compare the potential depth and the force."""
    x = np.linspace(-L / 2, L / 2, 4096, endpoint=False)
    dx = x[1] - x[0]
    rho1 = np.exp(-x**2 / (2 * sigma**2))
    rho1 -= rho1.mean()
    k = 2 * np.pi * np.fft.fftfreq(x.size, d=dx)
    V1 = np.zeros_like(x)
    nz = k != 0
    V1[nz] = np.real(np.fft.ifft(-np.fft.fft(rho1)[nz] / k[nz]**2))

    r = np.linspace(0, 40 * sigma, 400001)
    rho3 = np.exp(-r**2 / (2 * sigma**2))
    Mr = _cumtrapz(4 * np.pi * r**2 * rho3, r)
    Gtot = _cumtrapz(4 * np.pi * r * rho3, r)
    Phi3 = -(Mr / np.maximum(r, 1e-12)) - (Gtot[-1] - Gtot)

    E1 = np.abs(np.gradient(V1, dx))
    E3 = np.abs(Mr / np.maximum(r, 1e-12)**2)
    force = []
    for rr in [2.0, 4.0, 6.0, 8.0]:
        force.append(dict(r=rr,
                          f1=float(np.interp(rr, x, E1)),
                          f3=float(np.interp(rr, r, E3))))
    return dict(V1_depth=float(V1.max() - V1.min()),
                Phi3_depth=float(Phi3.max() - Phi3.min()),
                total_mass=float(Mr[-1]),
                force=force)


def box_dependence_1d(sigma: float = 1.0, Ls=(5.0, 10.0, 20.0, 40.0, 80.0)) -> list[dict]:
    """The 1-D periodic potential depth grows ~ linearly with box length, i.e.
    the 'gravity' is set by the arbitrary box, not by the lump."""
    out = []
    for L in Ls:
        x = np.linspace(-L / 2, L / 2, 8192, endpoint=False)
        dx = x[1] - x[0]
        rho = np.exp(-x**2 / (2 * sigma**2))
        rho -= rho.mean()
        k = 2 * np.pi * np.fft.fftfreq(x.size, d=dx)
        V = np.zeros_like(x)
        nz = k != 0
        V[nz] = np.real(np.fft.ifft(-np.fft.fft(rho)[nz] / k[nz]**2))
        depth = float(V.max() - V.min())
        out.append(dict(L=L, depth=depth, depth_times_L=depth * L))
    return out


# --------------------------------------------------------------------------- #
# 3. The "trapz over a random batch" artefact
# --------------------------------------------------------------------------- #
def madelung_trapz(N: int = 10000, Nb: int = 1000, seed: int = 0) -> dict:
    """The notebook computes int rho dx as trapz(rho_batch, dx=0.001, dim=0).
    Because the batch is an unsorted random sample of the domain, this is a
    scalar with a multiplicative bias dx*N/L and an O(1) batch-to-batch
    mismatch between the interior and boundary branches."""
    rng = np.random.default_rng(seed)
    true = float(np.trapezoid(np.exp(2 * np.linspace(-1, 1, 200001)**2),
                              np.linspace(-1, 1, 200001)))
    xb = rng.uniform(-1, 1, N)
    rho = np.exp(2 * xb**2)
    est = float(np.sum(0.5 * (rho[1:] + rho[:-1])) * 0.001)
    xb2 = rng.uniform(-1, 1, Nb)
    rho2 = np.exp(2 * xb2**2)
    est_b = float(np.sum(0.5 * (rho2[1:] + rho2[:-1])) * 0.001)
    return dict(true_integral=true, interior_estimate=est,
                boundary_estimate=est_b,
                bias=est / true, interior_boundary_mismatch=est / est_b)


# --------------------------------------------------------------------------- #
# 4. Ground-state soliton (matches PyUltraLight's own profile)
# --------------------------------------------------------------------------- #
def soliton_profile_metrics(profile_path: str) -> dict:
    """Load a radial ground-state profile (psi(r), sampled at dr = 1e-5) and
    return M_code, the half-density radius, and their product."""
    f = np.load(profile_path)
    dr = 1.0e-5
    r = np.arange(f.size) * dr
    rho = f**2
    M = 4 * np.pi * _trapz(rho * r**2, r)
    r_half = float(r[int(np.argmin(np.abs(rho - 0.5 * rho[0])))])
    return dict(M_code=M, r_half=r_half, M_times_r=M * r_half)


DEFAULT_PROFILE = os.path.join(
    "external", "PyUltraLight", "Soliton Profile Files", "initial_f.npy"
)
