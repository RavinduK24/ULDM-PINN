"""
Self-contained sanity tests for fdm.units (run without pytest):

    python tests/test_units.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fdm import units  # noqa: E402


def approx(a, b, rtol=1e-9):
    assert abs(a - b) <= rtol * max(abs(a), abs(b)), (a, b)


def test_code_units_match_pyultralight():
    # Reproduce PyUltraLight.py's module-level constants (lines 11-18).
    L, T, M = units.code_units(1e-22)
    hbar = 1.0545718e-34
    parsec = 3.0857e16
    G = 6.67e-11
    m = 1e-22 * 1.783e-36
    omega_m0 = 0.31
    H0 = 67.7e3 / (parsec * 1e6)
    Lref = (8 * np.pi * hbar**2 / (3 * m**2 * H0**2 * omega_m0)) ** 0.25
    Tref = (3 * H0**2 * omega_m0 / (8 * np.pi)) ** -0.5
    Mref = (3 * H0**2 * omega_m0 / (8 * np.pi)) ** 0.25 * hbar**1.5 / (m**1.5 * G)
    # 1e-3 rtol absorbs the small constant differences (G 6.67e-11 vs 6.67430e-11).
    approx(L, Lref, rtol=1e-3)
    approx(T, Tref, rtol=1e-3)
    approx(M, Mref, rtol=1e-3)


def test_soliton_scaling():
    r1 = units.soliton_core_radius(1e-22, 1e9)
    r2 = units.soliton_core_radius(1e-22, 1e10)
    approx(r1 / r2, 10.0, rtol=1e-6)        # 10x mass -> 0.1x radius
    r3 = units.soliton_core_radius(1e-21, 1e9)
    approx(r1 / r3, 100.0, rtol=1e-6)       # 10x heavier boson -> 1/100 radius


def test_de_broglie_scaling():
    rho = 7.13e-22  # kg/m^3 ~ 0.4 GeV/cm^3 (local DM density)
    lam = units.de_broglie_parameter(1e-22, units.KPC, rho)
    lam2 = units.de_broglie_parameter(2e-22, units.KPC, rho)
    approx(lam / lam2, 2.0, rtol=1e-9)      # lambda ∝ 1/m
    assert 0.1 < lam < 10, lam              # O(1) for FDM on kpc scales


def test_soliton_profile_normalization():
    assert abs(units.soliton_profile(0.0, 1.0) - 1.0) < 1e-12
    # r_c is the half-density radius: rho(r_c) = rho_c / 2
    assert abs(units.soliton_profile(1.0, 1.0) - 0.5) < 0.01


def main():
    test_code_units_match_pyultralight()
    print("ok: code_units matches PyUltraLight")
    test_soliton_scaling()
    print("ok: soliton scaling  r_c ~ 1/(m^2 M)")
    test_de_broglie_scaling()
    print("ok: de Broglie parameter  lambda ~ 1/m,  lambda ~ O(1) for fiducial FDM")
    test_soliton_profile_normalization()
    print("ok: soliton profile normalisation")
    print("\nall tests passed")


if __name__ == "__main__":
    main()
