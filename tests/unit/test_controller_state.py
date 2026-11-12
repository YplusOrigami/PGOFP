import numpy as np
import pytest

from pgofp.controllers.manager import ControllerManager
from pgofp.controllers.pid import InvertedPendulumPID, ReacherPID
from pgofp.core.exceptions import PriorUnavailableError
from pgofp.core.types import ControllerPrediction
from pgofp.controllers.base import BaseController


def _obs(n=3, seed=0):
    return np.random.default_rng(seed).normal(size=(n, 4)).astype(np.float32) * 0.1


def _pid(**g):
    return InvertedPendulumPID({"gains": {"theta": 10.0, "theta_dot": 1.0, "x": 0.5, "x_dot": 0.5, "integral_x": 1.0, **g}}, 1)


def test_predict_is_pure_and_repeatable():
    c = _pid()
    st = c.initial_state(3)
    o = _obs()
    before = {k: v.copy() for k, v in st.items()}
    a = c.predict(o, None, st).center_norm
    for _ in range(5):
        b = c.predict(o, None, st).center_norm
    assert np.array_equal(a, b)
    assert all(np.array_equal(before[k], st[k]) for k in st)


def test_advance_updates_integral_exactly_once_per_call():
    c = _pid()
    st = c.initial_state(2)
    o = np.array([[0.5, 0.0, 0.0, 0.0], [-0.2, 0, 0, 0]], np.float32)
    s1 = c.advance(st, o, None, np.zeros((2, 1)))
    assert np.allclose(s1["integral"][:, 0], o[:, 0] * c.dt)
    s2 = c.advance(s1, o, None, np.zeros((2, 1)))
    assert np.allclose(s2["integral"][:, 0], 2 * o[:, 0] * c.dt)


def test_anti_windup_clamps_integral():
    c = _pid()
    st = c.initial_state(1)
    o = np.array([[2.0, 0.5, 0.0, 0.0]], np.float32)  # large, saturating command
    for _ in range(2000):
        st = c.advance(st, o, None, np.zeros((1, 1)))
    assert abs(st["integral"][0, 0]) <= c.i_limit + 1e-9


def test_manager_reset_only_touches_masked_slots():
    c = _pid()
    mgr = ControllerManager([c], 3, 1)
    o = np.array([[0.5, 0, 0, 0]] * 3, np.float32)
    for _ in range(5):
        mgr.advance(o, np.zeros((3, 0)), np.zeros((3, 1)))
    before = mgr.states[0]["integral"].copy()
    assert (before > 0).all()
    mgr.reset_slots(np.array([False, True, False]))
    after = mgr.states[0]["integral"]
    assert after[1, 0] == 0 and after[0, 0] == before[0, 0] and after[2, 0] == before[2, 0]


def test_manager_predict_does_not_mutate_and_supports_subset_idx():
    mgr = ControllerManager([_pid()], 3, 1)
    mgr.advance(_obs(), np.zeros((3, 0)), np.zeros((3, 1)))
    snap = mgr.states[0]["integral"].copy()
    out_all = mgr.predict(_obs(), np.zeros((3, 0)))
    out_sub = mgr.predict(_obs()[[1]], np.zeros((1, 0)), idx=np.array([1]))
    assert np.array_equal(mgr.states[0]["integral"], snap)
    assert np.allclose(out_sub["prior_mu"][0], out_all["prior_mu"][1])
    assert out_all["prior_mu"].shape == (3, 1, 1) and out_all["prior_valid"].shape == (3, 1)


class _Boom(BaseController):
    n_components, version = 1, "boom"

    def predict(self, obs, ref, state):
        raise RuntimeError("solver crashed")


class _NaN(BaseController):
    n_components, version = 1, "nan"

    def predict(self, obs, ref, state):
        n = obs.shape[0]
        return ControllerPrediction(center_norm=np.full((n, 1, 1), np.nan))


def test_runtime_failure_becomes_explicit_invalid_flag_not_silent_zero_prior():
    for ctrl in (_Boom(), _NaN()):
        mgr = ControllerManager([ctrl], 2, 1)
        out = mgr.predict(_obs(2), np.zeros((2, 0)))
        assert (out["prior_valid"] == 0).all()
        assert np.isfinite(out["prior_mu"]).all()
        assert mgr.stats()["prior/fallback_rate"] == 1.0


def test_configuration_errors_are_not_swallowed():
    class Bad(BaseController):
        n_components, version = 1, "bad"

        def predict(self, obs, ref, state):
            return ControllerPrediction()  # neither center

    with pytest.raises(PriorUnavailableError):
        ControllerManager([Bad()], 1, 1).predict(_obs(1), np.zeros((1, 0)))


def test_failed_component_weight_moves_to_base_in_both_sampling_and_density(ctx_factory):
    """invalid prior_valid reaches the source builder and zeroes the prior weight consistently (see test_sources too)."""
    import torch

    from pgofp.core.specs import SourceSpec
    from pgofp.priors.mixture import build_source

    ctx = ctx_factory(B=2, d=1, M=1)
    ctx.prior_valid[0] = 0.0
    src = build_source(SourceSpec(kind="prior_mixture", beta=0.8), ctx, 1)
    w = torch.softmax(src.log_weights, -1)
    assert w[0, 0] == 1.0 and abs(float(w[1, 1]) - 0.8) < 1e-6


def test_state_dict_version_mismatch_rejected():
    mgr = ControllerManager([_pid()], 1, 1)
    sd = mgr.state_dict()
    sd["versions"] = ["other"]
    with pytest.raises(PriorUnavailableError):
        mgr.load_state_dict(sd)


def test_reacher_pid_outputs_bounded_and_wraps_angles():
    c = ReacherPID({"gains": {"kp": [2.0, 2.0], "kd": [0.3, 0.3]}}, 2)
    obs = np.zeros((1, 10), np.float32)
    obs[0, [0, 1]] = 1.0  # cos
    obs[0, 4:6] = [0.15, 0.05]
    p = c.predict(obs, None, c.initial_state(1))
    assert np.all(np.abs(p.center_norm) <= 1.0 + 1e-12) and p.valid.all()
