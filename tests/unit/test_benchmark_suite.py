"""The requested matrix is exact, portable and isolated from validation artifacts."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pgofp.config.compose import compose
from pgofp.core.exceptions import ConfigError
from pgofp.experiments.benchmark_suite import (
    artifact_for,
    build_plans,
    check_artifact,
    default_prior_paths,
    load_suite,
    recover_manifest,
    suite_lock,
)
from pgofp.experiments.launcher import Launcher
from pgofp.experiments.manifest import MatrixManifest


def test_exact_requested_150_run_matrix_and_uniform_flow_budget(tmp_path):
    suite = load_suite()
    paths = default_prior_paths(suite, tmp_path)
    plans = build_plans(suite, paths)
    assert len(plans) == len({p.run_key for p in plans}) == 150
    assert sum(p.axes["prior"] == "ppo" for p in plans) == 90
    assert sum(p.axes["prior"] == "pid" for p in plans) == 30
    assert sum(p.axes["prior"] == "mpc" for p in plans) == 30
    combos = {(p.axes["env"], p.axes["prior"]) for p in plans}
    assert len(combos) == 10
    for env, prior in combos:
        group = [p for p in plans if p.axes["env"] == env and p.axes["prior"] == prior]
        assert {(p.seed, p.axes["source.beta"]) for p in group} == {(s, b) for s in [0, 1, 2] for b in [0, .25, .5, .75, 1]}
    for p in plans:
        cfg = compose(p.overrides)
        assert p.status == "planned" and cfg.train.total_env_steps == 30000
        assert cfg.train.batch_size == 256 and cfg.train.learning_starts == 5000 and cfg.train.utd_ratio == 1
        assert list(cfg.network.actor_hidden) == list(cfg.network.critic_hidden) == [256] * 3
        assert cfg.algorithm.K_train == 32 and cfg.algorithm.K_local == 16
        assert cfg.algorithm.use_entropy and cfg.algorithm.density.steps == 5
        assert cfg.logging.wandb.mode == "disabled" and not cfg.logging.wandb.require_online
        assert not cfg.logging.wandb.upload_artifacts
        assert cfg.logging.wandb.scalar_source == "direct"
        assert cfg.logging.tensorboard and cfg.logging.jsonl
        assert cfg.eval.prior_before_training and cfg.eval.representative_action
        assert cfg.logging.console_progress
        if p.axes["env"] == "inverted_double_pendulum" and p.axes["prior"] == "mpc":
            assert cfg.prior.controller.version == "double_pendulum_cem_v1"
            assert cfg.prior.controller.cost.healthy_reward == 10


def test_portable_prior_paths_reuse_only_existing_canonical_final(tmp_path):
    suite = load_suite()
    old = artifact_for(suite, "half_cheetah", tmp_path / "artifacts/priors")
    old.mkdir(parents=True)
    paths = default_prior_paths(suite, tmp_path)
    assert paths["half_cheetah"] == old
    assert paths["reacher"].is_relative_to(tmp_path / "artifacts/priors" / suite["name"])
    assert all(str(p).startswith(str(tmp_path)) for p in paths.values())


def test_validation_runs_cover_sources_and_endpoints_without_changing_model(tmp_path):
    suite = load_suite()
    plans = build_plans(suite, default_prior_paths(suite, tmp_path), smoke=True, online=False)
    assert len(plans) == 30
    assert {p.axes["source.beta"] for p in plans} == {0, .5, 1}
    for p in plans:
        cfg = compose(p.overrides)
        assert cfg.smoke and cfg.train.total_env_steps == 24
        assert cfg.train.batch_size == 256 and cfg.algorithm.K_train + cfg.algorithm.K_local == 48
        assert cfg.eval.max_steps == 8
        assert cfg.logging.wandb.mode == "disabled"


def test_reject_low_budget_or_smoke_priors_for_formal_experiments(monkeypatch, tmp_path):
    import pgofp.experiments.benchmark_suite as suite

    art = SimpleNamespace(manifest={"distribution": "tanh_gaussian", "prior_seed": 100, "smoke": True},
                          budget={"prior_pretrain_env_steps": 64})
    monkeypatch.setattr(suite, "load_prior_artifact", lambda *a, **k: art)
    with pytest.raises(ConfigError, match="fewer"):
        check_artifact(tmp_path, "reacher", min_steps=1000000, prior_seed=100)
    art.budget["prior_pretrain_env_steps"] = 1001472
    with pytest.raises(ConfigError, match="Validation priors"):
        check_artifact(tmp_path, "reacher", min_steps=1000000, prior_seed=100)


def test_interrupted_run_resumes_checkpoint_and_completed_run_stays_complete(tmp_path):
    suite = load_suite()
    plans = build_plans(suite, default_prior_paths(suite, tmp_path))[:2]
    man = MatrixManifest(tmp_path, suite["name"], [p.to_record() for p in plans])
    man.runs[0]["status"] = "interrupted"
    man.runs[1]["status"] = "completed"
    d = tmp_path / "runs" / (plans[0].run_key + "__abcdef")
    (d / "checkpoints").mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"outcome": "interrupted"}))
    (d / "checkpoints/last.pt").write_bytes(b"fixture")
    recover_manifest(man, plans)
    assert man.runs[0]["status"] == "planned" and man.runs[1]["status"] == "completed"
    command = Launcher(man, Path.cwd())._cmd(man.runs[0])
    assert f"resume={d / 'checkpoints/last.pt'}" in command
    assert "resume_mode=episode_restart" in command


def test_duplicate_suite_launcher_is_rejected(tmp_path):
    with suite_lock(tmp_path), pytest.raises(ConfigError, match="already holds"), suite_lock(tmp_path):
        pass


def test_summary_uses_last_duplicate_eval_without_altering_raw_log(tmp_path):
    from pgofp.experiments.aggregate import _eval_curve

    rows = [{"step": 24, "eval/return_mean": 1}, {"step": 24, "eval/return_mean": 2}]
    path = tmp_path / "metrics.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert _eval_curve(tmp_path) == ([24], [2])
    assert len(path.read_text().splitlines()) == 2


def test_cloud_readback_checks_sparse_eval_separately_from_training(monkeypatch):
    import sys

    from pgofp.experiments.benchmark_suite import verify_cloud

    requested = []

    def history(*, keys, samples, pandas):
        requested.append(keys)
        assert samples > 0 and pandas is False
        return [{keys[0]: 1., "charts/env_steps": 30000}]

    run = SimpleNamespace(history=history, state="finished")
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Api=lambda **kw: SimpleNamespace(run=lambda path: run)))
    local = {"checks": [{"run_key": "sparse", "wandb": {"entity": "e", "project": "p", "run_id": "id", "url": "test"}}]}
    assert verify_cloud(local)["all_passed"]
    assert requested == [["eval/return_mean", "charts/env_steps"], ["loss/actor_qrmf", "charts/env_steps"]]


@pytest.mark.parametrize("args,expected", [
    (["--run"], False), (["--run", "--local-only"], False),
    (["--validate"], False), (["--run", "--wandb-online"], True),
])
def test_cli_defaults_to_local_even_with_online_environment(monkeypatch, tmp_path, args, expected):
    import pgofp.experiments.benchmark_suite as suite

    calls = []
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setattr(suite, "execute", lambda s, out, **kwargs: calls.append(kwargs) or 0)
    assert suite.main([*args, "--output-dir", str(tmp_path)]) == 0
    assert calls[0]["online"] is expected


def test_disable_cloud_preserves_completed_and_resumes_interrupted(tmp_path):
    suite = load_suite()
    paths = default_prior_paths(suite, tmp_path)
    old = build_plans(suite, paths, online=True)[:2]
    new = build_plans(suite, paths)[:2]
    man = MatrixManifest(tmp_path, suite["name"], [p.to_record() for p in old])
    man.runs[0]["status"] = "completed"
    man.runs[1]["status"] = "interrupted"
    directory = tmp_path / "runs" / (old[1].run_key + "__fixture")
    (directory / "checkpoints").mkdir(parents=True)
    (directory / "checkpoints/last.pt").write_bytes(b"fixture")
    recover_manifest(man, new)
    assert man.runs[0]["status"] == "completed"
    assert man.runs[1]["status"] == "planned" and man.runs[1]["resume_checkpoint"].endswith("last.pt")
    assert all(compose(r["overrides"]).logging.wandb.mode == "disabled" for r in man.runs)
    assert all(r["logging_history"][0]["previous_mode"] == "online" for r in man.runs)


def test_only_zero_data_wandb_startup_failures_are_requeued(tmp_path):
    suite = load_suite()
    paths = default_prior_paths(suite, tmp_path)
    old = build_plans(suite, paths, online=True)[:3]
    man = MatrixManifest(tmp_path, suite["name"], [p.to_record() for p in old])
    for index, rec in enumerate(man.runs):
        rec["status"] = "failed"
        directory = tmp_path / "runs" / (rec["run_key"] + "__fixture")
        (directory / "checkpoints").mkdir(parents=True)
        (directory / "events.jsonl").write_text(json.dumps({"kind": "wandb_required_online_failed"}) + "\n")
        if index == 1:
            (directory / "numerical_failure.json").write_text("{}")
        if index == 2:
            (directory / "metrics.jsonl").write_text(json.dumps({"step": 100, "critic/loss": 1}) + "\n")
    recover_manifest(man, build_plans(suite, paths)[:3])
    assert [r["status"] for r in man.runs] == ["planned", "failed", "failed"]


def test_logging_migration_rejects_live_children(monkeypatch, tmp_path):
    import pgofp.experiments.benchmark_suite as module

    suite = load_suite()
    paths = default_prior_paths(suite, tmp_path)
    old = build_plans(suite, paths, online=True)[:1]
    man = MatrixManifest(tmp_path, suite["name"], [p.to_record() for p in old])
    man.runs[0].update(status="running", pid=123)
    monkeypatch.setattr(module, "_process_alive", lambda pid: True)
    with pytest.raises(ConfigError, match="stop it before relaunch"):
        recover_manifest(man, build_plans(suite, paths)[:1])
