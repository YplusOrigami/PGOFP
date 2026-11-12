"""Controller sign/behaviour against the real MuJoCo tasks (these fix the signs empirically)."""
import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("mujoco")

from pgofp.controllers.pid import InvertedPendulumPID, ReacherPID  # noqa: E402

pytestmark = pytest.mark.mujoco


def _run_ipend(ctrl, seed, n=1000):
    e = gym.make("InvertedPendulum-v5")
    o, _ = e.reset(seed=seed)
    st = ctrl.initial_state(1)
    t = 0
    for t in range(n):
        a = ctrl.predict(o[None], None, st).center_norm[0, 0] * 3.0
        o2, r, te, tr, _ = e.step(a.astype(np.float32))
        st = ctrl.advance(st, o[None], None, np.zeros((1, 1)))
        o = o2
        if te or tr:
            break
    return t + 1


def test_pid_default_gains_balance_pendulum():
    c = InvertedPendulumPID({"gains": {"theta": 10.0, "theta_dot": 1.0, "x": 0.5, "x_dot": 0.5}}, 1)
    assert all(_run_ipend(c, s) == 1000 for s in (50000, 50001, 50002))


def test_flipped_sign_fails_quickly():
    c = InvertedPendulumPID({"gains": {"theta": -10.0, "theta_dot": -1.0, "x": -0.5, "x_dot": -0.5}}, 1)
    assert np.mean([_run_ipend(c, s) for s in (50000, 50001, 50002)]) < 50


def test_reacher_pd_reaches_target():
    e = gym.make("Reacher-v5")
    c = ReacherPID({"gains": {"kp": [2.0, 2.0], "kd": [0.3, 0.3]}}, 2)
    dists = []
    for s in range(50000, 50004):
        o, _ = e.reset(seed=s)
        st = c.initial_state(1)
        for _ in range(50):
            a = c.predict(o[None], None, st).center_norm[0, 0]
            o2, r, te, tr, info = e.step(a.astype(np.float32))
            st = c.advance(st, o[None], None, a[None])
            o = o2
        dists.append(-info["reward_dist"])
    assert np.mean(dists) < 0.02


def test_mpc_balances_pendulum_and_is_deterministic_and_does_not_touch_env():
    from pgofp.controllers.mpc import CEMMPC
    from pgofp.core.specs import ActionSpec, EnvSpec
    from pgofp.core.exceptions import UnsupportedCombinationError

    spec = EnvSpec("InvertedPendulum-v5", 4, ActionSpec((-3.0,), (3.0,)))
    cfg = {"horizon": 15, "population": 64, "iterations": 3, "init_std": 0.4, "elite_fraction": 0.1,
           "cost": {"w_x": 3.0, "w_xdot": 1.0, "w_theta": 20.0}}
    m = CEMMPC(cfg, spec)
    e = gym.make("InvertedPendulum-v5")
    o, _ = e.reset(seed=50000)
    qpos_before = e.unwrapped.data.qpos.copy()
    a1 = m.predict(o[None], None, {}).center_norm
    a2 = m.predict(o[None], None, {}).center_norm
    assert np.array_equal(a1, a2)  # fixed query randomness
    assert np.array_equal(e.unwrapped.data.qpos, qpos_before)  # planning used a separate model/data
    with pytest.raises(UnsupportedCombinationError):
        CEMMPC({**cfg, "model_access": "privileged_sim_state"}, spec)
    n = 0
    for _ in range(120):
        o, r, te, tr, _ = e.step((m.predict(o[None], None, {}).center_norm[0, 0] * 3.0).astype(np.float32))
        n += 1
        if te:
            break
    assert n == 120 and m.model_rollouts > 0
