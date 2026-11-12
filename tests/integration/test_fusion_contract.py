import numpy as np
import pytest

from pgofp.config.compose import compose
from pgofp.config.validate import validate_config
from pgofp.core.exceptions import BackendUnavailableError, ConfigError
from pgofp.envs.fusion.actuator import ActuatorWrapper, CoilMap
from pgofp.envs.fusion.adapter import FusionEnv, make_fusion_env_fn
from pgofp.envs.fusion.mock_backend import MockFusionBackend
from pgofp.envs.fusion.observation import ObservationAssembler
from pgofp.envs.fusion.protocol import ActuatorOwnership, FieldSpec, ObservationSchema
from pgofp.envs.fusion.reference import ReferenceTrack
from pgofp.envs.fusion.safety import SafetyFilter


def _env(**kw):
    return FusionEnv(MockFusionBackend(horizon=kw.pop("horizon", 20), **kw), CoilMap.mock_symmetric())


def test_seven_to_twelve_symmetric_expansion():
    cm = CoilMap.mock_symmetric()
    v = cm.expand(np.array([0.5, 0.2, -0.3, 0.1, 0.0, 0.9, -1.0]))
    assert v.shape == (12,)
    assert v[1] == v[2] and v[3] == v[4] and v[5] == v[6] and v[7] == v[8] and v[9] == v[10]  # up/down pairs identical
    assert np.all(np.abs(v) <= 1499.0 + 1e-9)
    assert np.allclose(cm.expand(np.full(7, 5.0)), cm.expand(np.ones(7)))  # commands clipped to [-1, 1]


def test_unverified_real_coil_map_refused(tmp_path):
    with pytest.raises(BackendUnavailableError):
        CoilMap.from_yaml(__import__("pathlib").Path(__file__).parents[2] / "configs" / "fusion" / "coil_map.yaml")


def test_real_backend_selection_refused_until_verified():
    cfg = compose(["env=fusion_stub", "prior=gaussian", "algorithm=gaussian_flame_m", "env.backend=fusion"])
    with pytest.raises(BackendUnavailableError):
        validate_config(cfg)
    with pytest.raises(BackendUnavailableError):
        make_fusion_env_fn(cfg, {})  # no SDK factory provided


def test_mock_env_spec_and_runs_without_any_sdk():
    e = _env()
    assert e.spec.action_dim == 7 and e.spec.obs_dim == 6 and e.spec.ref_dim == 3 and e.spec.phase_dim == 1
    obs, _ = e.reset(seed=0)
    assert obs.shape == (6,)
    obs, r, term, trunc, info = e.step(np.zeros(7))
    assert {"raw_command", "filtered_command", "intervention", "reward_terms", "termination_reason", "applied_voltage"} <= set(info)
    assert info["applied_voltage"].shape == (12,)


def test_reference_is_for_the_queried_step_not_a_stale_one():
    e = _env(horizon=10)
    e.reset(seed=0)
    r0 = e.reference().copy()
    for _ in range(3):
        e.step(np.zeros(7))
    r3 = e.reference()
    assert not np.allclose(r0, r3)
    assert np.allclose(r3, e.backend.get_reference(3, 0))


def test_ownership_prevents_double_application():
    own_backend = ActuatorWrapper(ActuatorOwnership("backend", "backend", "backend"), delay_steps=3, rate_limit_frac=0.01)
    v = np.full(12, 1000.0)
    assert np.array_equal(own_backend.apply(v), v)  # wrapper stays out of stages the backend owns
    own_wrapper = ActuatorWrapper(ActuatorOwnership("wrapper", "wrapper", "backend"), delay_steps=2, rate_limit_frac=0.1)
    outs = [own_wrapper.apply(v) for _ in range(5)]
    assert np.all(outs[0] == 0) and np.all(outs[1] == 0)  # delayed
    assert outs[2][0] <= 0.1 * 1499 + 1e-9  # rate limited
    with pytest.raises(ValueError):
        ActuatorOwnership("nobody", "backend", "backend").validate()


def test_terminal_semantics_truncation_vs_termination():
    e = _env(horizon=5, terminal_on_horizon=False)
    e.reset(seed=0)
    for i in range(5):
        _, _, term, trunc, info = e.step(np.zeros(7))
    assert trunc and not term and info["termination_reason"] == "time_limit"
    e2 = _env(horizon=5, terminal_on_horizon=True)
    e2.reset(seed=0)
    for i in range(5):
        _, _, term, trunc, info = e2.step(np.zeros(7))
    assert term and not trunc and info["termination_reason"] == "horizon_terminal"
    e3 = _env(horizon=50, error_limit=0.05)
    e3.reset(seed=0)
    for _ in range(50):
        _, _, term, trunc, info = e3.step(np.ones(7))
        if term or trunc:
            break
    assert term and info["termination_reason"] == "divergence"


def test_missing_observation_fields_are_masked_not_zero_filled_silently():
    schema = ObservationSchema((FieldSpec("a", "m", 0, 2), FieldSpec("b", "A", 2, 1)), 3)
    asm = ObservationAssembler(schema)
    vec, mask = asm.assemble({"a": np.array([1.0, 2.0])})
    assert mask.tolist() == [1, 1, 0] and vec.tolist() == [1, 2, 0]
    vec, mask = asm.assemble({"a": np.array([1.0, 2.0]), "b": np.array([5.0])}, {"b": np.array([0.0])})
    assert mask[2] == 0
    with pytest.raises(ConfigError):
        asm.assemble({"a": np.array([1.0])})
    with pytest.raises(ConfigError):
        ObservationAssembler(ObservationSchema((FieldSpec("a", "m", 0, 2),), 5))
    with pytest.raises(ConfigError):
        ObservationAssembler(schema, expected_units={"a": "cm"})


def test_safety_filter_reports_intervention_and_does_not_hide_it():
    f = SafetyFilter(box=0.5, max_delta=0.1)
    r1 = f(np.array([0.9, 0.0]))
    assert r1.intervention and "box" in r1.reason and np.allclose(r1.raw_command, [0.9, 0.0]) and np.allclose(r1.filtered_command, [0.5, 0.0])
    r2 = f(np.array([-0.5, 0.0]))
    assert "rate" in r2.reason
    assert not SafetyFilter(enabled=False)(np.array([5.0])).intervention


def test_reference_track_hash_and_lookahead_clamp():
    tr = ReferenceTrack(np.arange(10, dtype=np.float32).reshape(5, 2), "F1", 0.01, lookahead=2)
    assert tr.window(0).tolist() == [0, 1, 2, 3, 4, 5]
    assert tr.window(4).tolist() == [8, 9, 8, 9, 8, 9]  # clamped at the final step
    assert tr.hash == ReferenceTrack(tr.data.copy(), "F1", 0.01).hash


def test_mock_snapshot_roundtrip_is_exact():
    e = _env()
    e.reset(seed=3)
    for _ in range(4):
        e.step(np.ones(7) * 0.3)
    snap = e.get_snapshot()
    a = [e.step(np.ones(7) * 0.1)[0] for _ in range(3)]
    e.set_snapshot(snap)
    b = [e.step(np.ones(7) * 0.1)[0] for _ in range(3)]
    assert all(np.allclose(x, y) for x, y in zip(a, b))


def test_fusion_mock_trains_end_to_end(tmp_path):
    from pgofp.training.runner import Runner

    cfg = compose(["env=fusion_stub", "prior=gaussian", "algorithm=gaussian_flame_m", "experiment=smoke", f"output_dir={tmp_path}",
                   "train.total_env_steps=400", "train.learning_starts=100", "eval.interval=200", "eval.episodes=1", "checkpoint.interval=400",
                   "algorithm.K_train=2", "network.actor_hidden=[16]", "network.critic_hidden=[16]", "algorithm.density.steps=1",
                   "train.log_interval=100"])
    out = Runner(cfg).build().train()
    assert out["outcome"] == "completed"
