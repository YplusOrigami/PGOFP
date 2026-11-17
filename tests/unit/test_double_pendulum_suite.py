"""The new 300k suite preserves the old learning protocol and uses three worker slots."""
import json
from pathlib import Path

import pytest

from pgofp.config.compose import compose
from pgofp.experiments import benchmark_suite as suite

SPEC = Path(__file__).resolve().parents[2] / "configs/experiment/double_pendulum_prior_beta_300k.yaml"


def test_exact_18_runs_and_300k_budget_and_same_learning_protocol(tmp_path):
    new = suite.load_suite(SPEC)
    old = suite.load_suite()
    plans = suite.build_plans(new, suite.default_prior_paths(new, tmp_path), device="cuda")
    assert len(plans) == len({p.run_key for p in plans}) == 18
    assert {(p.axes["prior"], p.axes["source.beta"], p.seed) for p in plans} == {
        (prior, beta, seed) for prior in ("ppo", "mpc") for beta in (0, .5, 1) for seed in (0, 1, 2)}
    assert {p.axes["env"] for p in plans} == {"inverted_double_pendulum"}
    assert new["max_parallel"] == 3
    old_plans = suite.build_plans(old, suite.default_prior_paths(old, tmp_path), device="cuda")
    for p in plans:
        c = compose(p.overrides)
        previous = compose(next(q for q in old_plans if q.axes == p.axes and q.seed == p.seed).overrides)
        assert c.train.total_env_steps == 300000
        for group in ("algorithm", "network", "optimizer", "source", "condition", "eval", "checkpoint", "replay", "env"):
            assert c[group] == previous[group], group
        for key in c.train:
            if key != "total_env_steps":
                assert c.train[key] == previous.train[key]
        assert c.logging.wandb.mode == "disabled"
        assert c.logging.jsonl and c.logging.tensorboard and c.logging.console_progress
        assert c.logging.wandb.project == "pgofp-inverted_double_pendulum-300k"
        assert c.eval.prior_before_training and c.checkpoint.save_replay
        if p.axes["prior"] == "mpc":
            assert c.prior.controller.timeout_s == .5
            assert c.prior.controller.version == "double_pendulum_cem_v1"


def test_reuse_previous_suite_prior_portably_and_fallback_to_fresh_path(tmp_path):
    s = suite.load_suite(SPEC)
    env = "inverted_double_pendulum"
    fresh = suite.default_prior_paths(s, tmp_path)[env]
    assert fresh.is_relative_to(tmp_path / "artifacts/priors" / s["name"])
    previous = tmp_path / s["reuse_prior_paths"][env]
    previous.mkdir(parents=True)
    (previous / "manifest.json").write_text("{}")
    assert suite.default_prior_paths(s, tmp_path)[env] == previous


def test_cli_plan_records_correct_budget_and_does_not_start_training(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, "execute", lambda *a, **k: pytest.fail("Plan started training"))
    assert suite.main(["--spec", str(SPEC), "--plan", "--output-dir", str(tmp_path)]) == 0
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["flow_steps_each"] == 300000 and plan["total_flow_steps"] == 5400000
    assert plan["max_parallel_default"] == 3 and len(plan["runs"]) == 18
    assert not (tmp_path / "runs").exists()


def test_cli_uses_suite_device_and_parallel_defaults_without_running(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setattr(suite, "execute", lambda s, out, **kw: calls.append(kw) or 0)
    assert suite.main(["--spec", str(SPEC), "--run", "--output-dir", str(tmp_path)]) == 0
    assert calls[0]["max_parallel"] == 3 and calls[0]["device"] == "cuda"
    assert calls[0]["online"] is False


def test_short_validation_has_six_jobs_and_isolated_validation_priors(tmp_path):
    s = suite.load_suite(SPEC)
    paths = {e: suite.artifact_for(dict(s, prior_steps=64), e, tmp_path / "validation_priors") for e in s["ppo_envs"]}
    plans = suite.build_plans(s, paths, smoke=True, device="cpu")
    assert len(plans) == 6
    for p in plans:
        c = compose(p.overrides)
        assert c.smoke and c.train.total_env_steps == 24 and c.eval.max_steps == 8
        if p.axes["prior"] == "ppo":
            assert Path(c.prior.artifact_path).is_relative_to(tmp_path / "validation_priors")


def test_execution_acceptance_uses_300k_not_previous_30k(monkeypatch, tmp_path):
    s = suite.load_suite(SPEC)
    seen = {}
    paths = suite.default_prior_paths(s, tmp_path)
    monkeypatch.setattr(suite, "prepare_priors", lambda *a, **k: paths)
    monkeypatch.setattr(suite, "check_artifact", lambda *a, **k: None)
    monkeypatch.setattr(suite, "validate_config", lambda *a, **k: None)
    monkeypatch.setattr(suite, "aggregate", lambda *a, **k: None)

    class FakeLauncher:
        def __init__(self, man, root, **kwargs):
            seen["parallel"] = kwargs["max_parallel"]

        def run(self, **kwargs):
            pass

    def verify(man, expected_steps):
        seen["steps"] = expected_steps
        return {"all_passed": True, "checks": []}

    monkeypatch.setattr(suite, "Launcher", FakeLauncher)
    monkeypatch.setattr(suite, "verify_local", verify)
    assert suite.execute(s, tmp_path, device="cpu", online=False, entity=None, smoke=False, max_parallel=3) == 0
    assert seen == {"parallel": 3, "steps": 300000}
    preflight = json.loads((tmp_path / "preflight.json").read_text())
    assert preflight["max_parallel"] == 3 and preflight["flow_steps_each"] == 300000
