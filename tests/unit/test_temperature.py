import math

import pytest
import torch

from pgofp.algorithms.temperature import Temperature


def _t(mode="auto", alpha=0.2, d=3):
    return Temperature(mode, alpha, "auto", d, lr=0.1, alpha_min=1e-3, alpha_max=10.0)


def test_low_entropy_raises_alpha_high_entropy_lowers_it():
    t = _t()
    a0 = t.alpha
    t.step(torch.tensor(-10.0))  # entropy far BELOW target (-d = -3)
    assert t.alpha > a0
    t2 = _t()
    t2.step(torch.tensor(+10.0))  # entropy far ABOVE target
    assert t2.alpha < a0


def test_target_entropy_default_is_minus_action_dim():
    assert _t(d=5).target_entropy == -5.0


def test_gradient_never_reaches_actor_graph():
    t = _t()
    x = torch.randn(4, requires_grad=True)
    ent = (x * 2).sum()  # stand-in for the entropy estimate of the online actor
    t.step(ent)
    assert x.grad is None


def test_fixed_mode_does_not_move_and_has_no_optimizer():
    t = _t("fixed", 0.1)
    for _ in range(5):
        t.step(torch.tensor(-100.0))
    assert abs(t.alpha - 0.1) < 1e-6 and t.opt is None


def test_bounds_and_bound_hits_recorded():
    t = Temperature("auto", 0.5, -2.0, 2, lr=5.0, alpha_min=0.1, alpha_max=1.0)
    for _ in range(10):
        m = t.step(torch.tensor(-50.0))
    assert t.alpha <= 1.0 + 1e-6
    assert m["alpha_at_bound_frac"] > 0.5


def test_state_roundtrip():
    t = _t()
    t.step(torch.tensor(-5.0))
    t2 = _t()
    t2.load_state_dict(t.state_dict())
    assert abs(t2.alpha - t.alpha) < 1e-9
