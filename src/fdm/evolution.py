"""One-dimensional Cartesian SPINN for periodic ultralight dark matter.

The main time-dependent model implements the Cartesian Schrödinger--Poisson
equations used by Mishra & Tolley (2025):

    i psi_t = -(lambda / 2) psi_xx + (1 / lambda) V psi
    V_xx    = |psi|^2 - 1

on a periodic interval. Initial and periodic boundary conditions are imposed
softly through explicit loss terms, matching the conventional PINN formulation.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


class Sine(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sin(value)


class EvolutionMLP(nn.Module):
    """Paper-sized sine MLP with Xavier-uniform initialization."""

    def __init__(self, n_hidden: int = 32, n_layers: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        width = 2  # normalized Cartesian x and t
        for _ in range(n_layers):
            linear = nn.Linear(width, n_hidden)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.extend((linear, Sine()))
            width = n_hidden
        output = nn.Linear(width, 3)
        nn.init.xavier_uniform_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class CartesianEvolutionNet(nn.Module):
    """Periodic 1-D Cartesian Schrödinger--Poisson PINN.

    The initial condition is ``psi(x,0)=sqrt(2)*sin(2*pi*x/L)``. The ``sqrt(2)``
    ensures that the spatially averaged density is one, as required by the
    periodic Poisson equation ``V_xx = |psi|^2 - 1``.
    """

    def __init__(
        self,
        length: float = 8.0,
        final_time: float = 3.0,
        coupling: float = 1.0,
        n_hidden: int = 32,
        n_layers: int = 2,
    ):
        super().__init__()
        if length <= 0 or final_time <= 0 or coupling <= 0:
            raise ValueError("length, final_time, and coupling must be positive")
        self.length = float(length)
        self.final_time = float(final_time)
        self.coupling = float(coupling)
        self.mlp = EvolutionMLP(n_hidden=n_hidden, n_layers=n_layers)

    @property
    def wave_number(self) -> float:
        return 2.0 * math.pi / self.length

    def initial_fields(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exact initial real field, imaginary field, and zero-mean potential."""
        phase = self.wave_number * x
        # A raw sin(kx) has <|psi|^2> = 1/2 and is incompatible with
        # V_xx = |psi|^2 - 1 on a periodic domain. sqrt(2) normalizes the mean
        # density to one: rho - 1 = -cos(2kx).
        real = math.sqrt(2.0) * torch.sin(phase)
        imag = torch.zeros_like(real)
        # This zero-mean potential exactly solves V_xx = rho - 1 at t=0.
        potential = torch.cos(2.0 * phase) / (4.0 * self.wave_number**2)
        return real, imag, potential

    def forward(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_scaled = 2.0 * x / self.length - 1.0
        t_scaled = 2.0 * t / self.final_time - 1.0
        output = self.mlp(torch.cat((x_scaled, t_scaled), dim=1))
        return output[:, 0:1], output[:, 1:2], output[:, 2:3]


def _first_and_second(
    value: torch.Tensor, coordinate: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    first = torch.autograd.grad(
        value,
        coordinate,
        grad_outputs=torch.ones_like(value),
        retain_graph=True,
        create_graph=True,
    )[0]
    second = torch.autograd.grad(
        first,
        coordinate,
        grad_outputs=torch.ones_like(first),
        retain_graph=True,
        create_graph=True,
    )[0]
    return first, second


def residuals(
    model: CartesianEvolutionNet, x: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Real Schrödinger, imaginary Schrödinger, and Poisson residuals."""
    real, imag, potential = model(x, t)
    _, real_xx = _first_and_second(real, x)
    _, imag_xx = _first_and_second(imag, x)
    _, potential_xx = _first_and_second(potential, x)
    real_t = torch.autograd.grad(
        real, t, torch.ones_like(real), retain_graph=True, create_graph=True
    )[0]
    imag_t = torch.autograd.grad(
        imag, t, torch.ones_like(imag), retain_graph=True, create_graph=True
    )[0]

    lam = model.coupling
    sch_real = real_t + 0.5 * lam * imag_xx - potential * imag / lam
    sch_imag = imag_t - 0.5 * lam * real_xx + potential * real / lam
    poisson = potential_xx - (real**2 + imag**2 - 1.0)
    return sch_real, sch_imag, poisson


def loss_terms(
    model: CartesianEvolutionNet,
    x: torch.Tensor,
    t: torch.Tensor,
    x_initial: torch.Tensor,
    t_boundary: torch.Tensor,
    t_mass: torch.Tensor,
    n_mass_x: int = 256,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """PDE, IC, periodic BC, and global mass-conservation MSE losses."""
    if n_mass_x < 2:
        raise ValueError("n_mass_x must be at least two")
    sch_real, sch_imag, poisson = residuals(model, x, t)
    schrodinger = (sch_real**2 + sch_imag**2).mean()
    poisson_loss = (poisson**2).mean()

    zero_time = torch.zeros_like(x_initial)
    predicted_initial = model(x_initial, zero_time)
    target_initial = model.initial_fields(x_initial)
    initial = sum(
        ((predicted - target) ** 2).mean()
        for predicted, target in zip(predicted_initial, target_initial)
    )

    x_left = torch.zeros_like(t_boundary, requires_grad=True)
    x_right = torch.full_like(t_boundary, model.length, requires_grad=True)
    fields_left = model(x_left, t_boundary)
    fields_right = model(x_right, t_boundary)
    boundary_terms = []
    for left, right in zip(fields_left, fields_right):
        boundary_terms.append(((left - right) ** 2).mean())
        left_x = torch.autograd.grad(
            left,
            x_left,
            grad_outputs=torch.ones_like(left),
            retain_graph=True,
            create_graph=True,
        )[0]
        right_x = torch.autograd.grad(
            right,
            x_right,
            grad_outputs=torch.ones_like(right),
            retain_graph=True,
            create_graph=True,
        )[0]
        boundary_terms.append(((left_x - right_x) ** 2).mean())
    boundary = torch.stack(boundary_terms).mean()

    # The endpoint is excluded so each point represents one equal-width cell
    # on the periodic domain.  Since the IC is normalized to unit mean density,
    # conservation of total mass is equivalent to <|psi|^2>_x = 1.
    x_mass_line = model.length * torch.arange(
        n_mass_x, dtype=t_mass.dtype, device=t_mass.device
    ).reshape(1, -1) / n_mass_x
    x_mass = x_mass_line.expand(t_mass.shape[0], -1).reshape(-1, 1)
    mass_times = t_mass.reshape(-1, 1).expand(-1, n_mass_x).reshape(-1, 1)
    mass_real, mass_imag, _ = model(x_mass, mass_times)
    mean_density = (mass_real**2 + mass_imag**2).reshape(-1, n_mass_x).mean(dim=1)
    mass = ((mean_density - 1.0) ** 2).mean()
    return schrodinger, poisson_loss, initial, boundary, mass


def train(
    model: CartesianEvolutionNet,
    n_adam: int = 2000,
    n_lbfgs: int = 100,
    n_colloc: int = 10_000,
    n_initial: int | None = None,
    n_boundary: int | None = None,
    learning_rate: float = 1.0e-3,
    initial_weight: float = 1.0,
    boundary_weight: float = 1.0,
    mass_weight: float = 5.0,
    n_mass_times: int = 8,
    n_mass_x: int = 256,
    seed: int = 0,
    log_every: int = 100,
) -> list[dict[str, float]]:
    """Train over the full time domain, then polish globally with L-BFGS."""
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    adam = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    n_initial = n_colloc if n_initial is None else n_initial
    n_boundary = n_colloc if n_boundary is None else n_boundary
    if n_initial < 1 or n_boundary < 1:
        raise ValueError("n_initial and n_boundary must be positive")
    if initial_weight < 0 or boundary_weight < 0 or mass_weight < 0:
        raise ValueError("constraint weights must be non-negative")
    if n_mass_times < 1 or n_mass_x < 2:
        raise ValueError("n_mass_times must be positive and n_mass_x at least two")

    def sample(count: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = (model.length * torch.rand(
            count, 1, dtype=dtype, device=device
        )).requires_grad_(True)
        t = (model.final_time * torch.rand(
            count, 1, dtype=dtype, device=device
        )).requires_grad_(True)
        return x, t

    def sample_constraints() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_initial = model.length * torch.rand(
            n_initial, 1, dtype=dtype, device=device
        )
        t_boundary = model.final_time * torch.rand(
            n_boundary, 1, dtype=dtype, device=device
        )
        t_mass = model.final_time * torch.rand(
            n_mass_times, 1, dtype=dtype, device=device
        )
        return x_initial, t_boundary, t_mass

    def objective(
        x: torch.Tensor,
        t: torch.Tensor,
        x_initial: torch.Tensor,
        t_boundary: torch.Tensor,
        t_mass: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ]:
        pieces = loss_terms(
            model, x, t, x_initial, t_boundary, t_mass, n_mass_x=n_mass_x
        )
        total = (
            pieces[0]
            + pieces[1]
            + initial_weight * pieces[2]
            + boundary_weight * pieces[3]
            + mass_weight * pieces[4]
        )
        return total, pieces

    for step in range(n_adam):
        x, t = sample(n_colloc)
        x_initial, t_boundary, t_mass = sample_constraints()
        adam.zero_grad(set_to_none=True)
        total, pieces = objective(x, t, x_initial, t_boundary, t_mass)
        total.backward()
        adam.step()
        record = {
            "total": float(total.detach()),
            "schrodinger": float(pieces[0].detach()),
            "poisson": float(pieces[1].detach()),
            "initial": float(pieces[2].detach()),
            "boundary": float(pieces[3].detach()),
            "mass": float(pieces[4].detach()),
        }
        history.append(record)
        if step % log_every == 0 or step == n_adam - 1:
            print(
                f"adam {step:5d}  L={record['total']:.3e}  "
                f"sch={record['schrodinger']:.3e}  "
                f"poisson={record['poisson']:.3e}  "
                f"ic={record['initial']:.3e}  bc={record['boundary']:.3e}  "
                f"mass={record['mass']:.3e}",
                flush=True,
            )

    if n_lbfgs > 0:
        x_fixed, t_fixed = sample(max(n_colloc, 10_000))
        x_initial_fixed, t_boundary_fixed, t_mass_fixed = sample_constraints()
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=n_lbfgs,
            history_size=min(50, n_lbfgs),
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            total, _ = objective(
                x_fixed, t_fixed, x_initial_fixed, t_boundary_fixed, t_mass_fixed
            )
            total.backward()
            return total

        before = float(closure().detach())
        lbfgs.step(closure)
        after, pieces = objective(
            x_fixed, t_fixed, x_initial_fixed, t_boundary_fixed, t_mass_fixed
        )
        record = {
            "total": float(after.detach()),
            "schrodinger": float(pieces[0].detach()),
            "poisson": float(pieces[1].detach()),
            "initial": float(pieces[2].detach()),
            "boundary": float(pieces[3].detach()),
            "mass": float(pieces[4].detach()),
        }
        history.append(record)
        print(
            f"lbfgs       L={before:.3e} -> {record['total']:.3e}  "
            f"sch={record['schrodinger']:.3e}  "
            f"poisson={record['poisson']:.3e}  "
            f"ic={record['initial']:.3e}  bc={record['boundary']:.3e}  "
            f"mass={record['mass']:.3e}",
            flush=True,
        )
    return history


@torch.no_grad()
def evaluate(
    model: CartesianEvolutionNet, x: np.ndarray, t: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate fields on the Cartesian product of time and position arrays."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    x_array = np.asarray(x, dtype=np.float64)
    t_array = np.asarray(t, dtype=np.float64)
    xx, tt = np.meshgrid(x_array, t_array)
    position = torch.as_tensor(xx.reshape(-1, 1), dtype=dtype, device=device)
    time = torch.as_tensor(tt.reshape(-1, 1), dtype=dtype, device=device)
    real, imag, potential = model(position, time)
    shape = (t_array.size, x_array.size)
    return (
        real.cpu().numpy().reshape(shape),
        imag.cpu().numpy().reshape(shape),
        potential.cpu().numpy().reshape(shape),
    )
