"""One-dimensional periodic split-step Fourier Schrödinger--Poisson solver.

The reference system is identical to :mod:`fdm.evolution`:

    i psi_t = -(lambda / 2) psi_xx + (1 / lambda) V psi
    V_xx    = |psi|^2 - 1,

with a zero-mean potential. Time evolution uses a second-order Strang
kick--drift--kick scheme. Each substep is unitary, so mass is conserved to
floating-point precision and the result provides an independent PINN baseline.
"""
from __future__ import annotations

import math

import numpy as np


def wave_numbers(n_x: int, length: float) -> np.ndarray:
    """Angular Fourier wave numbers for an endpoint-excluded periodic grid."""
    return 2.0 * np.pi * np.fft.fftfreq(n_x, d=length / n_x)


def initial_wavefunction(x: np.ndarray, length: float) -> np.ndarray:
    """Normalized pure-sine initial wavefunction used by the PINN."""
    return math.sqrt(2.0) * np.sin(2.0 * np.pi * x / length) + 0.0j


def periodic_potential(psi: np.ndarray, length: float) -> np.ndarray:
    """Solve ``V_xx = |psi|^2 - 1`` spectrally with ``mean(V)=0``."""
    psi = np.asarray(psi, dtype=np.complex128)
    k = wave_numbers(psi.size, length)
    source_hat = np.fft.fft(np.abs(psi) ** 2 - 1.0)
    potential_hat = np.zeros_like(source_hat)
    nonzero = k != 0.0
    potential_hat[nonzero] = -source_hat[nonzero] / k[nonzero] ** 2
    return np.fft.ifft(potential_hat).real


def solve(
    length: float = 8.0,
    final_time: float = 3.0,
    coupling: float = 1.0,
    n_x: int = 512,
    dt: float = 1.0e-3,
    output_times: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Evolve the periodic system and return fields at requested times."""
    if length <= 0 or final_time <= 0 or coupling <= 0 or dt <= 0:
        raise ValueError("length, final_time, coupling, and dt must be positive")
    if n_x < 8:
        raise ValueError("n_x must be at least 8")

    if output_times is None:
        output_times = np.linspace(0.0, final_time, 121)
    times = np.asarray(output_times, dtype=np.float64)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("output_times must be a nonempty one-dimensional array")
    if np.any(np.diff(times) < 0) or times[0] < 0 or times[-1] > final_time + 1e-13:
        raise ValueError("output_times must increase within [0, final_time]")

    x = np.linspace(0.0, length, n_x, endpoint=False)
    k = wave_numbers(n_x, length)
    psi = initial_wavefunction(x, length)
    kinetic_rate = 0.5 * coupling * k**2
    snapshots = np.empty((times.size, n_x), dtype=np.complex128)
    potentials = np.empty((times.size, n_x), dtype=np.float64)
    current_time = 0.0

    for output_index, target_time in enumerate(times):
        while current_time < target_time - 1.0e-14:
            step = min(dt, target_time - current_time)
            potential = periodic_potential(psi, length)
            psi *= np.exp(-0.5j * step * potential / coupling)
            psi = np.fft.ifft(
                np.exp(-1j * step * kinetic_rate) * np.fft.fft(psi)
            )
            potential = periodic_potential(psi, length)
            psi *= np.exp(-0.5j * step * potential / coupling)
            current_time += step
        snapshots[output_index] = psi
        potentials[output_index] = periodic_potential(psi, length)

    density = np.abs(snapshots) ** 2
    return {
        "x": x,
        "t": times,
        "psi": snapshots,
        "potential": potentials,
        "density": density,
        "mean_density": density.mean(axis=1),
    }
