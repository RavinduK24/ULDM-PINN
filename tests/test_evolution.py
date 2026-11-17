"""Sanity tests for the soft-constrained Cartesian SPINN."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fdm import evolution  # noqa: E402


torch.set_default_dtype(torch.float64)


def make_model() -> evolution.CartesianEvolutionNet:
    torch.manual_seed(7)
    return evolution.CartesianEvolutionNet(n_hidden=8, n_layers=2).double()


def test_wavefunction_sine_is_normalized_and_solves_poisson():
    model = make_model()
    x_line = torch.linspace(0.0, model.length, 2001).reshape(-1, 1)
    real_line, imag_line, _ = model.initial_fields(x_line)
    density = (real_line**2 + imag_line**2)[:, 0]
    mean_density = torch.trapezoid(density, x_line[:, 0]) / model.length
    assert abs(float(mean_density) - 1.0) < 1.0e-12

    x = torch.linspace(0.1, model.length - 0.1, 19).reshape(-1, 1)
    x.requires_grad_(True)
    real, imag, potential = model.initial_fields(x)
    first = torch.autograd.grad(potential.sum(), x, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), x)[0]
    residual = second - (real**2 + imag**2 - 1.0)
    assert float(residual.abs().max()) < 1.0e-12


def make_loss_inputs(model: evolution.CartesianEvolutionNet):
    x = torch.linspace(0.1, model.length - 0.1, 11).reshape(-1, 1)
    t = torch.linspace(0.1, model.final_time, 11).reshape(-1, 1)
    x.requires_grad_(True)
    t.requires_grad_(True)
    x_initial = torch.linspace(0.0, model.length, 13).reshape(-1, 1)
    t_boundary = torch.linspace(0.0, model.final_time, 9).reshape(-1, 1)
    t_mass = torch.linspace(0.0, model.final_time, 5).reshape(-1, 1)
    return x, t, x_initial, t_boundary, t_mass


def test_all_five_losses_are_active_and_finite():
    model = make_model()
    losses = evolution.loss_terms(model, *make_loss_inputs(model), n_mass_x=32)
    assert len(losses) == 5
    assert all(bool(torch.isfinite(loss)) for loss in losses)
    # The unconstrained random network initially violates both ICs and BCs,
    # confirming that these terms provide nonzero optimization signals.
    assert float(losses[2]) > 1.0e-8
    assert float(losses[3]) > 1.0e-12
    assert float(losses[4]) > 1.0e-8


def test_combined_loss_backpropagates():
    model = make_model()
    losses = evolution.loss_terms(model, *make_loss_inputs(model), n_mass_x=32)
    total = sum(losses)
    total.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)


def main():
    test_wavefunction_sine_is_normalized_and_solves_poisson()
    print("ok: wavefunction sine is normalized and solves Poisson")
    test_all_five_losses_are_active_and_finite()
    print("ok: PDE, IC, periodic BC, and mass losses are active and finite")
    test_combined_loss_backpropagates()
    print("ok: combined loss backpropagates")
    print("\nall soft-constrained evolution tests passed")


if __name__ == "__main__":
    main()
