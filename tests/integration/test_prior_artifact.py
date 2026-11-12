import json
import shutil

import numpy as np
import pytest
import torch

pytest.importorskip("gymnasium")

from pgofp.config.compose import compose
from pgofp.controllers.artifacts import load_prior_artifact
from pgofp.controllers.ppo import PPOPriorController, build_ppo_controller
from pgofp.core.exceptions import PriorUnavailableError
from pgofp.core.specs import ActionSpec, EnvSpec
from pgofp.networks.gaussian_policy import TanhGaussianPolicy
from pgofp.training.prior_trainer import PPOTrainer, assign_tiers


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    d = tmp_path_factory.mktemp("prior")
    cfg = compose(["env=inverted_pendulum", "algorithm=ppo", "prior=gaussian", "algorithm.params.rollout_steps=128",
                   "algorithm.params.epochs=2", "algorithm.params.hidden=[16,16]"])
    tr = PPOTrainer(cfg, seed=100, device="cpu")
    tr.run(256)
    path = tr.export(d / "art", {"stochastic": {"return_mean": 1.0}}, {"prior_pretrain_env_steps": tr.env_steps})
    spec = tr.spec
    pol_sd = {k: v.clone() for k, v in tr.policy.state_dict().items()}
    mean, std = tr.norm.mean.copy(), np.sqrt(tr.norm.var + 1e-8)
    tr.close()
    return path, spec, pol_sd, mean, std


def test_export_load_roundtrip_matches_distribution(artifact):
    path, spec, sd, mean, std = artifact
    ctrl = build_ppo_controller({}, spec, "native", str(path))
    obs = np.random.default_rng(0).normal(size=(5, spec.obs_dim)).astype(np.float32)
    pred = ctrl.predict(obs, None, {})
    pol = TanhGaussianPolicy(spec.obs_dim, spec.action_dim, (16, 16), "tanh")
    pol.load_state_dict(sd)
    x = torch.as_tensor(np.clip((obs - mean.astype(np.float32)) / std.astype(np.float32), -10, 10))
    mu, ls = pol.dist_params(x)
    assert np.allclose(pred.center_latent[:, 0], mu.detach().numpy(), atol=1e-5)
    assert np.allclose(pred.log_std_latent[:, 0], ls.detach().numpy(), atol=1e-5)
    assert pred.center_norm is None and ctrl.capability.latent_coordinates


def test_center_gaussian_mode_hides_native_std(artifact):
    path, spec, *_ = artifact
    ctrl = build_ppo_controller({}, spec, "center_gaussian", str(path))
    pred = ctrl.predict(np.zeros((2, spec.obs_dim), np.float32), None, {})
    assert pred.log_std_latent is None and not ctrl.capability.provides_std


def test_controller_is_frozen_and_never_trained(artifact):
    path, spec, *_ = artifact
    ctrl = build_ppo_controller({}, spec, "native", str(path))
    assert all(not p.requires_grad for p in ctrl.policy.parameters())
    before = [p.clone() for p in ctrl.policy.parameters()]
    ctrl.predict(np.ones((3, spec.obs_dim), np.float32), None, {})
    assert all(torch.equal(a, b) for a, b in zip(before, ctrl.policy.parameters()))


def test_uses_artifact_normalizer_not_a_new_one(artifact):
    path, spec, *_ = artifact
    ctrl = build_ppo_controller({}, spec, "native", str(path))
    obs = np.full((1, spec.obs_dim), 100.0, np.float32)
    p_big = ctrl.predict(obs, None, {}).center_latent
    # clip at +-10 after the FROZEN normalization: values beyond the clip give identical output
    assert np.array_equal(p_big, ctrl.predict(obs * 2, None, {}).center_latent) or np.allclose(p_big, ctrl.predict(obs * 2, None, {}).center_latent)


def test_hash_tamper_wrong_env_dims_and_schema_rejected(artifact, tmp_path):
    path, spec, *_ = artifact
    bad = tmp_path / "tampered"
    shutil.copytree(path, bad)
    w = torch.load(bad / "weights.pt", weights_only=False)
    w["policy"]["log_std"] = w["policy"]["log_std"] + 1
    torch.save(w, bad / "weights.pt")
    with pytest.raises(PriorUnavailableError):
        load_prior_artifact(bad)
    with pytest.raises(PriorUnavailableError):
        load_prior_artifact(path, expect_env="Hopper-v5")
    with pytest.raises(PriorUnavailableError):
        load_prior_artifact(path, expect_obs_dim=spec.obs_dim + 1)
    with pytest.raises(PriorUnavailableError):
        load_prior_artifact(path, expect_hash="deadbeef")
    inc = tmp_path / "incomplete"
    shutil.copytree(path, inc)
    (inc / "normalizer.npz").unlink()
    with pytest.raises(PriorUnavailableError):
        load_prior_artifact(inc)
    stateful = tmp_path / "stateful"
    shutil.copytree(path, stateful)
    m = json.loads((stateful / "manifest.json").read_text())
    m["stateful"] = True
    (stateful / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(PriorUnavailableError):
        load_prior_artifact(stateful)


def test_tier_assignment_uses_calibration_only():
    tiers = assign_tiers({"a": {"return_mean": 10.0}, "b": {"return_mean": 55.0}, "c": {"return_mean": 100.0}}, random_return=5.0)
    assert tiers == {"a": "poor", "b": "medium", "c": "good"}


def test_ppo_baseline_records_native_and_center_policies_without_mutation(artifact, tmp_path):
    from pgofp.config.compose import to_container
    from pgofp.evaluation.prior_baseline import evaluate_prior
    from pgofp.logging.logger import Logger

    path, *_ = artifact
    before = load_prior_artifact(path).content_hash
    cfg = compose(["env=inverted_pendulum", "prior=ppo", f"prior.artifact_path={path}",
                   "source=constant", "algorithm=prior_flame_m", "logging=local", "eval.episodes=2", "eval.max_steps=4"])
    logger = Logger(tmp_path, cfg, to_container(cfg), "ppo-baseline")
    state = torch.get_rng_state()
    try:
        result = evaluate_prior(cfg, tmp_path, logger, 0)
    finally:
        logger.close()
    assert result["primary_policy"] == "native_stochastic"
    assert "prior_eval_deterministic" in result and result["prior_eval"]["summary"]["prior_eval/episodes"] == 2
    assert torch.equal(state, torch.get_rng_state())
    assert load_prior_artifact(path).content_hash == before
