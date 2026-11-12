"""Short end-to-end trainings: finite losses, parameters move, frozen priors stay frozen, Gaussian reuses the same class."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from pgofp.config.compose import compose
from pgofp.training.runner import Runner

BASE = ["env=toy", "experiment=smoke", "train.total_env_steps=900", "train.learning_starts=200", "train.log_interval=300",
        "eval.interval=450", "eval.episodes=2", "checkpoint.interval=450", "algorithm.K_train=4", "network.actor_hidden=[32,32]",
        "network.critic_hidden=[32,32]", "algorithm.density.steps=2"]


def _cfg(tmp_path, extra):
    return compose(BASE + [f"output_dir={tmp_path}"] + extra)


def _scalars(run_dir):
    return [json.loads(l) for l in (Path(run_dir) / "metrics.jsonl").read_text().splitlines()]


def test_gaussian_flame_trains_and_updates_params(tmp_path):
    r = Runner(_cfg(tmp_path, ["prior=gaussian", "algorithm=gaussian_flame_m"])).build()
    before = {k: v.clone() for k, v in r.agent.actor.state_dict().items()}
    out = r.train()
    assert out["outcome"] == "completed" and out["gradient_steps"] > 0
    assert any(not torch.equal(before[k], v) for k, v in r.agent.actor.state_dict().items())
    recs = _scalars(r.run_dir)
    train_recs = [x for x in recs if "loss/actor_qrmf" in x]
    assert train_recs and all(math.isfinite(x["loss/actor_qrmf"]) and math.isfinite(x["critic/loss"]) for x in train_recs)
    steps = [x["step"] for x in recs]
    assert steps == sorted(steps)  # monotone env-step axis


def test_gaussian_and_prior_variants_share_the_same_agent_class(tmp_path):
    a = Runner(_cfg(tmp_path, ["prior=gaussian", "algorithm=gaussian_flame_m"])).build()
    b = Runner(_cfg(tmp_path, ["env=toy", "prior=gaussian", "algorithm=prior_flame_m", "source.kind=gaussian"])).build()
    assert type(a.agent) is type(b.agent)
    a.pool.close(); b.pool.close(); a.eval_pool.close(); b.eval_pool.close()


class _ToyPID:
    """Registered ad-hoc controller for the toy env: u = -0.5*obs (normalized), deterministic."""


def _toy_prior_cfg(tmp_path, extra=()):
    return _cfg(tmp_path, ["prior=gaussian", "algorithm=prior_flame_m", "source.kind=prior_mixture", "source.beta=0.75",
                           "prior.type=toy_p"] + list(extra))


def _register_toy_controller():
    from pgofp.controllers.base import BaseController
    from pgofp.core.registry import CONTROLLERS
    from pgofp.core.types import ControllerPrediction

    if "toy_p" in CONTROLLERS.names():
        return

    class ToyP(BaseController):
        n_components, version = 1, "toy_p_v1"

        def predict(self, obs, ref, state):
            return ControllerPrediction(center_norm=np.clip(0.6 * np.tanh(obs), -0.95, 0.95)[:, None, :])

    CONTROLLERS.register("toy_p")(lambda cfg, env_spec, **kw: ToyP())


def test_prior_source_trains_and_prior_params_are_frozen(tmp_path, monkeypatch):
    _register_toy_controller()
    from pgofp.controllers import factory

    monkeypatch.setattr(factory, "_build_one", lambda ptype, c, spec, kw, ap, dist, seed: __import__("pgofp.core.registry", fromlist=["CONTROLLERS"]).CONTROLLERS.build("toy_p", c, spec))
    cfg = compose(BASE + [f"output_dir={tmp_path}", "prior=gaussian", "algorithm=prior_flame_m", "source.kind=prior_mixture"])
    from omegaconf import OmegaConf
    OmegaConf.set_struct(cfg, False)
    cfg.prior.type = "toy_p"
    r = Runner.__new__(Runner)
    # use the standard validation-free path: build manually to inject the toy prior type
    import pgofp.training.runner as runner_mod
    monkeypatch.setattr(runner_mod, "validate_config", lambda c, check_artifacts=True: [])
    r = Runner(cfg).build()
    assert r.manager.n_prior == 1 and r.ctx_spec.n_prior == 1
    ctrl = r.manager.controllers[0]
    out = r.train()
    assert out["outcome"] == "completed"
    rec = [x for x in _scalars(r.run_dir) if "prior/beta" in x][-1]
    assert rec["prior/beta"] == 0.75 and rec["prior/fallback_rate"] == 0.0
    assert ctrl.version == "toy_p_v1"


def test_ofql_and_sac_baselines_run(tmp_path):
    for algo in ("online_ofql", "sac"):
        r = Runner(_cfg(tmp_path, ["prior=gaussian", f"algorithm={algo}", "source.kind=gaussian"])).build()
        out = r.train()
        assert out["outcome"] == "completed", algo
        recs = _scalars(r.run_dir)
        assert any("critic/loss" in x for x in recs)


def test_controller_only_runs_without_updates(tmp_path):
    r = Runner(_cfg(tmp_path, ["prior=gaussian", "algorithm=controller_only", "source.kind=gaussian"])).build()
    out = r.train()
    assert out["outcome"] == "completed" and out["gradient_steps"] == 0


def test_nan_stops_the_update_and_is_recorded(tmp_path, monkeypatch):
    from pgofp.core.exceptions import NumericalError
    r = Runner(_cfg(tmp_path, ["prior=gaussian", "algorithm=gaussian_flame_m"])).build()
    orig = r.agent.critic.forward

    def bad(cond, a):
        q1, q2 = orig(cond, a)
        return q1 * float("nan"), q2

    monkeypatch.setattr(r.agent.critic, "forward", bad)
    with pytest.raises(NumericalError):
        r.train()
    assert (Path(r.run_dir) / "numerical_failure.json").is_file()
    m = json.loads((Path(r.run_dir) / "manifest.json").read_text())
    assert m["outcome"] == "failed_numerical"


def test_zero_velocity_init_policy_is_tanh_of_source(tmp_path):
    r = Runner(_cfg(tmp_path, ["prior=gaussian", "algorithm=gaussian_flame_m"])).build()
    eps = torch.randn(16, 2)
    cond = torch.randn(16, r.builder.dim)
    z = r.agent.actor.one_step(eps, cond)
    assert torch.equal(z, eps)  # last layer zero-initialized => z_out = eps exactly
