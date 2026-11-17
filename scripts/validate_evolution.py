"""Validate a trained Cartesian PINN against a split-step Fourier solution."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fdm import evolution, spectral  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a trained ULDM PINN with a spectral reference"
    )
    parser.add_argument("--run", required=True, help="training output directory")
    parser.add_argument("--out", default=None, help="validation output directory")
    parser.add_argument("--n-x", type=int, default=512)
    parser.add_argument("--n-t", type=int, default=121)
    parser.add_argument("--dt", type=float, default=1.0e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def load_state(path: str, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def infer_architecture(state: dict[str, torch.Tensor]) -> tuple[int, int]:
    weight_keys = [
        key
        for key in state
        if key.startswith("mlp.net.") and key.endswith(".weight")
    ]
    weight_keys.sort(key=lambda key: int(key.split(".")[2]))
    if len(weight_keys) < 2:
        raise ValueError("checkpoint does not contain the expected MLP weights")
    return int(state[weight_keys[0]].shape[0]), len(weight_keys) - 1


def relative_l2_by_time(predicted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    numerator = np.linalg.norm(predicted - reference, axis=1)
    denominator = np.linalg.norm(reference, axis=1)
    return numerator / np.maximum(denominator, np.finfo(float).eps)


def summarize(error: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(error)),
        "maximum": float(np.max(error)),
        "final": float(error[-1]),
    }


def main() -> None:
    args = parse_args()
    run_dir = os.path.abspath(args.run)
    output_dir = os.path.abspath(args.out or os.path.join(run_dir, "validation"))
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(run_dir, "metrics.json"), encoding="utf-8") as handle:
        training_metrics = json.load(handle)
    device = select_device(args.device)
    state = load_state(os.path.join(run_dir, "model.pt"), device)
    hidden, layers = infer_architecture(state)
    length = float(training_metrics["length"])
    final_time = float(training_metrics["time"])
    coupling = float(training_metrics["lambda"])

    torch.set_default_dtype(torch.float64)
    model = evolution.CartesianEvolutionNet(
        length=length,
        final_time=final_time,
        coupling=coupling,
        n_hidden=hidden,
        n_layers=layers,
    ).to(device=device, dtype=torch.float64)
    model.load_state_dict(state)
    model.eval()

    x = np.linspace(0.0, length, args.n_x, endpoint=False)
    t = np.linspace(0.0, final_time, args.n_t)
    real, imag, potential = evolution.evaluate(model, x, t)
    psi_pinn = real + 1j * imag
    density_pinn = np.abs(psi_pinn) ** 2

    reference = spectral.solve(
        length=length,
        final_time=final_time,
        coupling=coupling,
        n_x=args.n_x,
        dt=args.dt,
        output_times=t,
    )
    psi_ref = reference["psi"]
    density_ref = reference["density"]

    # Poisson fixes V only up to a spatial constant. Compare in the same
    # zero-mean gauge as the spectral reference.
    potential_pinn_zero_mean = potential - potential.mean(axis=1, keepdims=True)
    potential_ref = reference["potential"]

    # A spatially constant potential changes only the wavefunction's global
    # phase. Align that unobservable phase independently at each output time.
    overlap = np.sum(np.conj(psi_pinn) * psi_ref, axis=1)
    phase = np.ones_like(overlap)
    nonzero = np.abs(overlap) > np.finfo(float).eps
    phase[nonzero] = overlap[nonzero] / np.abs(overlap[nonzero])
    psi_pinn_aligned = psi_pinn * phase[:, None]

    density_error = relative_l2_by_time(density_pinn, density_ref)
    wavefunction_error = relative_l2_by_time(psi_pinn_aligned, psi_ref)
    potential_error = relative_l2_by_time(
        potential_pinn_zero_mean, potential_ref
    )
    mass_pinn = density_pinn.mean(axis=1)
    mass_ref = reference["mean_density"]
    mass_drift_pinn = np.max(np.abs(mass_pinn / mass_pinn[0] - 1.0))
    mass_drift_ref = np.max(np.abs(mass_ref / mass_ref[0] - 1.0))

    metrics = {
        "checkpoint": run_dir,
        "device": str(device),
        "hidden_width": hidden,
        "hidden_layers": layers,
        "reference_n_x": args.n_x,
        "reference_dt": args.dt,
        "density_relative_l2": summarize(density_error),
        "phase_aligned_wavefunction_relative_l2": summarize(wavefunction_error),
        "zero_mean_potential_relative_l2": summarize(potential_error),
        "pinn_max_relative_mass_drift": float(mass_drift_pinn),
        "reference_max_relative_mass_drift": float(mass_drift_ref),
    }
    with open(
        os.path.join(output_dir, "validation_metrics.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metrics, handle, indent=2)
    np.savez(
        os.path.join(output_dir, "validation.npz"),
        x=x,
        t=t,
        density_pinn=density_pinn,
        density_reference=density_ref,
        psi_pinn=psi_pinn,
        psi_pinn_phase_aligned=psi_pinn_aligned,
        psi_reference=psi_ref,
        potential_pinn=potential_pinn_zero_mean,
        potential_reference=potential_ref,
        density_relative_l2=density_error,
        wavefunction_relative_l2=wavefunction_error,
        potential_relative_l2=potential_error,
        mass_pinn=mass_pinn,
        mass_reference=mass_ref,
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    extent = (0.0, length, 0.0, final_time)
    image = axes[0, 0].imshow(
        density_ref, origin="lower", aspect="auto", extent=extent, cmap="magma"
    )
    axes[0, 0].set(title="Spectral density", xlabel="x", ylabel="t")
    fig.colorbar(image, ax=axes[0, 0])
    image = axes[0, 1].imshow(
        density_pinn, origin="lower", aspect="auto", extent=extent, cmap="magma"
    )
    axes[0, 1].set(title="PINN density", xlabel="x", ylabel="t")
    fig.colorbar(image, ax=axes[0, 1])
    image = axes[0, 2].imshow(
        np.abs(density_pinn - density_ref),
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
    )
    axes[0, 2].set(title="Absolute density error", xlabel="x", ylabel="t")
    fig.colorbar(image, ax=axes[0, 2])

    axes[1, 0].plot(x, density_ref[-1], label="Spectral")
    axes[1, 0].plot(x, density_pinn[-1], "--", label="PINN")
    axes[1, 0].set(title=f"Final density, t={final_time:g}", xlabel="x", ylabel="rho")
    axes[1, 0].legend()
    axes[1, 1].semilogy(t, density_error, label="density")
    axes[1, 1].semilogy(t, wavefunction_error, label="wavefunction (phase aligned)")
    axes[1, 1].semilogy(t, potential_error, label="potential (zero mean)")
    axes[1, 1].set(title="Relative L2 error", xlabel="t", ylabel="relative L2")
    axes[1, 1].legend(fontsize=8)
    axes[1, 2].plot(t, mass_ref / mass_ref[0] - 1.0, label="Spectral")
    axes[1, 2].plot(t, mass_pinn / mass_pinn[0] - 1.0, label="PINN")
    axes[1, 2].set(title="Mass conservation", xlabel="t", ylabel="relative drift")
    axes[1, 2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1, 2].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "validation.png"), dpi=170)
    plt.close(fig)

    print(f"density relative L2 (final)={density_error[-1]:.3e}")
    print(
        "wavefunction relative L2, phase aligned "
        f"(final)={wavefunction_error[-1]:.3e}"
    )
    print(f"potential relative L2, zero mean (final)={potential_error[-1]:.3e}")
    print(f"PINN max mass drift={mass_drift_pinn:.3e}")
    print(f"spectral max mass drift={mass_drift_ref:.3e}")
    print(f"[io] wrote {output_dir}")


if __name__ == "__main__":
    main()
