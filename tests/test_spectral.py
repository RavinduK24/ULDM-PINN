"""Tests for the periodic split-step Fourier reference solver."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fdm import spectral  # noqa: E402


def test_periodic_poisson_solution():
    length = 8.0
    n_x = 128
    x = np.linspace(0.0, length, n_x, endpoint=False)
    psi = spectral.initial_wavefunction(x, length)
    potential = spectral.periodic_potential(psi, length)
    k = spectral.wave_numbers(n_x, length)
    laplacian = np.fft.ifft(-(k**2) * np.fft.fft(potential)).real
    source = np.abs(psi) ** 2 - 1.0
    assert np.max(np.abs(laplacian - source)) < 1.0e-12
    assert abs(float(potential.mean())) < 1.0e-14


def test_split_step_conserves_mass():
    result = spectral.solve(
        length=8.0,
        final_time=0.1,
        coupling=1.0,
        n_x=128,
        dt=1.0e-3,
        output_times=np.linspace(0.0, 0.1, 6),
    )
    drift = np.max(
        np.abs(result["mean_density"] / result["mean_density"][0] - 1.0)
    )
    assert drift < 1.0e-12


def main():
    test_periodic_poisson_solution()
    print("ok: spectral potential solves periodic Poisson")
    test_split_step_conserves_mass()
    print("ok: split-step solver conserves mass")
    print("\nall spectral reference tests passed")


if __name__ == "__main__":
    main()
