"""One-step map vs multi-step integration of the diagonal field: consistent fields agree, inconsistent ones are flagged."""
import torch
from torch import nn

from pgofp.density.diagnostics import generate_multistep, one_step_gap
from pgofp.networks.mean_flow import MeanFlowNet


def test_consistent_straight_flow_has_zero_gap():
    """u(z,r,t) = z - m for a straight line to m: diagonal field v(z,t)=(z-m)/t, average velocity = (z-m) (r=0,t=1)."""

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = torch.tensor([[0.7, -0.3]])

        def forward(self, z, r, t, c):
            # average velocity over [r,t] of the straight path to m: constant (z_t - m)/t  (independent of r)
            return (z - self.m) / t.clamp_min(1e-6)

        def one_step(self, eps, cond):
            n = eps.shape[0]
            zero, one = torch.zeros(n, 1), torch.ones(n, 1)
            return eps - self(eps, zero, one, cond)

    net = Net()
    eps = torch.randn(64, 2)
    out = one_step_gap(net, eps, torch.zeros(64, 0), steps=200)
    assert out["flow/one_step_gap_latent"] < 5e-2  # integration error only
    z1 = net.one_step(eps, torch.zeros(64, 0))
    assert torch.allclose(z1, net.m.expand(64, 2), atol=1e-6)  # every source sample lands exactly on m in one step


def test_inconsistent_network_gap_is_detected():
    class Bad(nn.Module):
        """Average velocity at (0,1) is unrelated to the diagonal field -> one-step output disagrees with multi-step."""

        def forward(self, z, r, t, c):
            diag = (r == t).all()
            return 0.2 * z if bool(diag) else 0.9 * z + 0.5

        def one_step(self, eps, cond):
            n = eps.shape[0]
            return eps - self(eps, torch.zeros(n, 1), torch.ones(n, 1), cond)

    out = one_step_gap(Bad(), torch.randn(64, 2), torch.zeros(64, 0), steps=50)
    assert out["flow/one_step_gap_latent"] > 0.3


def test_zero_init_network_is_consistent():
    net = MeanFlowNet(2, 3, (16,), 8, "silu")
    eps, c = torch.randn(32, 2), torch.randn(32, 3)
    out = one_step_gap(net, eps, c, steps=10)
    assert out["flow/one_step_gap_latent"] < 1e-6 and out["flow/one_step_gap_action"] < 1e-6
    assert torch.allclose(generate_multistep(net, eps, c, 5), eps)
