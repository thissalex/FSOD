import pytest
import torch

from fsod_experiments.adaptation import fit, match


def data():
    g = torch.Generator().manual_seed(123)
    return torch.randn(6, 5, 12, generator=g), torch.tensor([0, 0, 1, 1, 2, 2])


def test_identity_and_noncommuting_closed_form():
    x, y = data()
    a0, p, _, _ = fit(x, y, 0, 10)
    assert torch.allclose(a0, torch.eye(12, dtype=a0.dtype), atol=1e-12)
    ae, _, _, _ = fit(x[:, :1].expand(-1, 5, -1), y)
    assert torch.allclose(ae, torch.eye(12, dtype=ae.dtype), atol=1e-12)
    a, _, _, stats = fit(x, y, 10, 3)
    assert stats["solve_relative_residual"] < 1e-12
    # Noncommuting C and S produce a nonsymmetric A; a transpose bug is detectable.
    assert not torch.allclose(a, a.T, atol=1e-6)
    scores = match(x[:, 0], p, a)
    assert scores.shape == (6, 3)
    assert scores.abs().max() <= 1 + 1e-12


def test_view_replication_does_not_change_statistics():
    x, y = data()
    a, p, _, _ = fit(x, y)
    ar, pr, _, _ = fit(x.repeat_interleave(2, dim=1), y)
    assert torch.allclose(a, ar, atol=1e-12)
    assert torch.allclose(p, pr, atol=1e-12)


def test_single_class_and_zero_features():
    x, y = data()
    _, _, _, stats = fit(x, torch.zeros_like(y))
    assert stats["class_trace"] == 0
    with pytest.raises(ValueError, match="Near-zero"):
        fit(torch.zeros_like(x), y)


def test_mu_zero_matches_regularized_suppression():
    x, y = data()
    a, _, _, _ = fit(x, y, 10, 0)
    z = torch.nn.functional.normalize(x.double(), dim=-1)
    e = (z - z.mean(1, keepdim=True)).flatten(0, 1).T / (30**0.5)
    eye = torch.eye(12, dtype=torch.float64)
    assert torch.allclose(a @ (eye + 10 * e @ e.T), eye, atol=1e-12)
