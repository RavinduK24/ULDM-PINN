"""Train and visualize the paper-style Cartesian ULDM SPINN."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fdm import evolution  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Periodic Cartesian ULDM Schrödinger--Poisson PINN"
    )
    parser.add_argument("--out", default="data/evolution")
    parser.add_argument("--length", type=float, default=8.0)
    parser.add_argument("--time", type=float, default=3.0)
    parser.add_argument("--lambda", dest="coupling", type=float, default=1.0)
    parser.add_argument("--adam", type=int, default=2000)
    parser.add_argument("--lbfgs", type=int, default=100)
    parser.add_argument("--colloc", type=int, default=10_000)
    parser.add_argument(
        "--initial-points",
        type=int,
        default=None,
        help="IC samples per Adam step (default: same as --colloc)",
    )
    parser.add_argument(
        "--boundary-points",
        type=int,
        default=None,
        help="periodic BC time samples per Adam step (default: same as --colloc)",
    )
    parser.add_argument("--ic-weight", type=float, default=1.0)
    parser.add_argument("--bc-weight", type=float, default=1.0)
    parser.add_argument(
        "--mass-weight", type=float, default=5.0,
        help="weight of the global mass-conservation loss",
    )
    parser.add_argument(
        "--mass-times", type=int, default=8,
        help="times sampled for the mass loss per optimizer step",
    )
    parser.add_argument(
        "--mass-grid", type=int, default=256,
        help="periodic spatial grid points used by the mass loss",
    )
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--n-x", type=int, default=401)
    parser.add_argument("--n-t", type=int, default=121)
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    if args.adam < 0 or args.lbfgs < 0 or args.colloc < 1:
        raise ValueError("training step counts must be non-negative and colloc positive")
    if args.initial_points is not None and args.initial_points < 1:
        raise ValueError("initial-points must be positive")
    if args.boundary_points is not None and args.boundary_points < 1:
        raise ValueError("boundary-points must be positive")
    if args.mass_times < 1 or args.mass_grid < 2:
        raise ValueError("mass-times must be positive and mass-grid >= 2")
    if args.ic_weight < 0 or args.bc_weight < 0 or args.mass_weight < 0:
        raise ValueError("loss weights must be non-negative")

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = select_device(args.device)
    os.makedirs(args.out, exist_ok=True)

    model = evolution.CartesianEvolutionNet(
        length=args.length,
        final_time=args.time,
        coupling=args.coupling,
        n_hidden=args.hidden,
        n_layers=args.layers,
    ).to(device=device, dtype=torch.float64)
    print(
        f"device={device}  parameters={sum(p.numel() for p in model.parameters()):,}  "
        f"domain=[0,{args.length}]x[0,{args.time}]  lambda={args.coupling}  "
        "IC=sqrt(2)*sin(2*pi*x/L)",
        flush=True,
    )

    started = time.perf_counter()
    history = evolution.train(
        model,
        n_adam=args.adam,
        n_lbfgs=args.lbfgs,
        n_colloc=args.colloc,
        n_initial=args.initial_points,
        n_boundary=args.boundary_points,
        initial_weight=args.ic_weight,
        boundary_weight=args.bc_weight,
        mass_weight=args.mass_weight,
        n_mass_times=args.mass_times,
        n_mass_x=args.mass_grid,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - started

    # Include both endpoints to measure periodic agreement explicitly.
    x = np.linspace(0.0, args.length, args.n_x)
    t = np.linspace(0.0, args.time, args.n_t)
    real, imag, potential = evolution.evaluate(model, x, t)
    density = real**2 + imag**2
    mean_density = np.trapezoid(density, x, axis=1) / args.length
    mass_drift = float(np.max(np.abs(mean_density / mean_density[0] - 1.0)))
    periodic_error = float(
        max(
            np.max(np.abs(real[:, 0] - real[:, -1])),
            np.max(np.abs(imag[:, 0] - imag[:, -1])),
            np.max(np.abs(potential[:, 0] - potential[:, -1])),
        )
    )

    metrics = {
        "elapsed_seconds": elapsed,
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "hidden_width": args.hidden,
        "hidden_layers": args.layers,
        "initial_mean_density": float(mean_density[0]),
        "final_mean_density": float(mean_density[-1]),
        "max_relative_mass_drift": mass_drift,
        "max_periodic_value_error": periodic_error,
        "final_losses": history[-1] if history else None,
        "length": args.length,
        "time": args.time,
        "lambda": args.coupling,
        "initial_condition": "sqrt(2)*sin(2*pi*x/L)",
        "adam_steps": args.adam,
        "lbfgs_steps": args.lbfgs,
        "collocation_points": args.colloc,
        "initial_points": args.initial_points or args.colloc,
        "boundary_points": args.boundary_points or args.colloc,
        "initial_weight": args.ic_weight,
        "boundary_weight": args.bc_weight,
        "mass_weight": args.mass_weight,
        "mass_times": args.mass_times,
        "mass_grid": args.mass_grid,
    }
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    np.savez(
        os.path.join(args.out, "solution.npz"),
        x=x,
        t=t,
        real=real,
        imag=imag,
        potential=potential,
        density=density,
        mean_density=mean_density,
    )
    torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for index in np.linspace(0, args.n_t - 1, 5).astype(int):
        axes[0].plot(x, density[index], label=f"t={t[index]:.2f}")
    axes[0].set(xlabel="x", ylabel=r"$|\psi|^2$", title="Density snapshots")
    axes[0].legend(fontsize=8)

    image = axes[1].imshow(
        density,
        origin="lower",
        aspect="auto",
        extent=(x[0], x[-1], t[0], t[-1]),
        cmap="magma",
    )
    axes[1].set(xlabel="x", ylabel="t", title="Cartesian density evolution")
    fig.colorbar(image, ax=axes[1], label=r"$|\psi|^2$")

    axes[2].plot(t, mean_density / mean_density[0] - 1.0)
    axes[2].axhline(0.0, color="black", linewidth=0.7)
    axes[2].set(
        xlabel="t",
        ylabel=r"$\langle\rho\rangle/\langle\rho_0\rangle-1$",
        title="Mass conservation",
    )
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "evolution.png"), dpi=170)
    plt.close(fig)

    print(f"elapsed={elapsed:.1f}s")
    print(f"max relative mass drift={mass_drift:.3e}")
    print(f"max periodic value error={periodic_error:.3e}")
    print(f"[io] wrote {args.out}")


if __name__ == "__main__":
    main()
