"""
Physical units and scaling relations for the fuzzy-dark-matter (FDM)
Schrödinger–Poisson system.

The FDM equations in physical units are (psi is the macroscopic field, rho = m |psi|^2):

    i hbar d_t psi = - (hbar^2 / 2m) nabla^2 psi + m Phi psi
    nabla^2 Phi    = 4 pi G m |psi|^2

Two unit conventions are provided:

1. `code_units()` — the "BECDM" units of Mocz et al. (2017), also used by the
   PyUltraLight reference code: hbar = m = 1 and 4 pi G = 1.  These are the units
   in which the reference soliton profile (the ground state) lives.

2. `de_broglie_parameter()` — the single dimensionless number that SPINN
   (Mishra & Tolley 2025) calls `lambda`.  Rewriting the physical system with
   x = L x~, t = T t~, |Psi|^2 = rho / rho_bar, Phi = 4 pi G rho_bar L^2 V and
   choosing T = 1 / sqrt(4 pi G rho_bar) gives

        i d_t Psi = - (lambda/2) nabla^2 Psi + (1/lambda) V Psi
        nabla^2 V = |Psi|^2 - 1

   with the **single** free parameter

        lambda = hbar / ( m L^2 sqrt(4 pi G rho_bar) ).

   lambda encodes the particle mass m for a given box size L and background
   density rho_bar.  lambda -> infinity is the free Schrödinger equation,
   lambda -> 0 is strongly self-gravitating.  This is the number a PINN must
   treat as its physical input if it is to "model ultralight DM candidates".
"""
from __future__ import annotations

import numpy as np

HBAR = 1.054571817e-34      # J s
G = 6.67430e-11             # m^3 kg^-1 s^-2
EV = 1.78266192e-36         # kg per eV
PARSEC = 3.0857e16          # m
KPC = 1.0e3 * PARSEC        # m
SOLAR_MASS = 1.98892e30     # kg

# Ground-state soliton, computed from the radial SP ODE in code units
# (hbar = m = 1, 4 pi G = 1).  SOLITON_MASS0 is the mass of the unit-amplitude
# (psi(0) = 1) ground state; SOLITON_MR is the invariant mass x half-density
# radius.  These two numbers pin down the physical scaling r_c = K hbar^2 / (G m^2 M).
SOLITON_MASS0 = 3.882765
SOLITON_MR = 2.679419
# rho_c = RHO_C_COEF * M / r_c^3 for the ground state.
# Derivation: in code units rho_c = 1, M = SOLITON_MASS0, r_c = SOLITON_MR/SOLITON_MASS0,
# so RHO_C_COEF = r_c^3 / M = (SOLITON_MR/SOLITON_MASS0)^3 / SOLITON_MASS0.
RHO_C_COEF = SOLITON_MR**3 / SOLITON_MASS0**4


def code_units(axion_mass_eV: float = 1e-22,
               H0_km_s_Mpc: float = 67.7,
               omega_m0: float = 0.31) -> tuple[float, float, float]:
    """
    Return (length_unit, time_unit, mass_unit) in SI for the BECDM code units,
    matching PyUltraLight's definitions (its module-level constants).

    length_unit = (8 pi hbar^2 / (3 m^2 H0^2 Omega_m))^{1/4}
    time_unit   = (3 H0^2 Omega_m / (8 pi))^{-1/2}
    mass_unit   = (3 H0^2 Omega_m / (8 pi))^{1/4} hbar^{3/2} / (m^{3/2} G)
    """
    m = axion_mass_eV * EV
    H0 = H0_km_s_Mpc * 1e3 / (PARSEC * 1e6)  # s^-1
    omega = 3.0 * H0**2 * omega_m0 / (8.0 * np.pi)
    length = (8.0 * np.pi * HBAR**2 / (3.0 * m**2 * H0**2 * omega_m0)) ** 0.25
    time = omega**-0.5
    mass = omega**0.25 * HBAR**1.5 / (m**1.5 * G)
    return length, time, mass


def de_broglie_parameter(axion_mass_eV: float,
                         box_length_m: float,
                         rho_bar_kg_m3: float) -> float:
    """SPINN's lambda = hbar / (m L^2 sqrt(4 pi G rho_bar))."""
    m = axion_mass_eV * EV
    return HBAR / (m * box_length_m**2 * np.sqrt(4.0 * np.pi * G * rho_bar_kg_m3))


def soliton_core_radius(axion_mass_eV: float, soliton_mass_Msun: float) -> float:
    """Half-density core radius in metres: r_c = K hbar^2 / (G m^2 M)."""
    m = axion_mass_eV * EV
    M = soliton_mass_Msun * SOLAR_MASS
    return SOLITON_MR * HBAR**2 / (G * m**2 * M)


def soliton_central_density(axion_mass_eV: float, soliton_mass_Msun: float) -> float:
    """Central mass density in kg/m^3 for the ground-state soliton."""
    r_c = soliton_core_radius(axion_mass_eV, soliton_mass_Msun)
    return RHO_C_COEF * soliton_mass_Msun * SOLAR_MASS / r_c**3


def soliton_profile(r, r_c):
    """Schive, Yu & Chiueh (2014) fit, normalised to rho_c at r = 0.

    rho(r) = rho_c [1 + 0.091 (r/r_c)^2]^-8
    """
    return (1.0 + 0.091 * (np.asarray(r, dtype=float) / r_c) ** 2) ** -8
