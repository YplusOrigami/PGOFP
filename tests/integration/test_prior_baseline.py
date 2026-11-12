"""Baseline evaluation cannot change training RNG, budget or the frozen prior."""
import json
import random

import numpy as np
import pytest
import torch

pytest.importorskip("mujoco")

from pgofp.config.compose import compose, to_container
from pgofp.evaluation.prior_baseline import evaluate_prior
from pgofp.logging.jsonl import read_jsonl
from pgofp.logging.logger import Logger
from pgofp.training.runner import Runner


def cfg_for(path, extra=()):
    return compose(["env=inverted_pendulum", "prior=pid", "algorithm=prior_flame_m", "source=constant",
                    "logging=local", "logging.console_progress=true", "device=cpu", f"output_dir={path}",
                    "eval.episodes=2", "eval.max_steps=4", "eval.representative_action=false", *extra])


def test_baseline_is_independent_of_beta_and_preserves_global_rng(tmp_path):
    means = []
    for beta in (0, 1):
        cfg = cfg_for(tmp_path, [f"source.beta={beta}"])
        output = tmp_path / str(beta)
        logger = Logger(output, cfg, to_container(cfg), "baseline")
        py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            report = evaluate_prior(cfg, output, logger, 0)
            means.append(report["prior_eval"]["summary"]["prior_eval/return_mean"])
            assert random.getstate() == py and torch.equal(cpu, torch.get_rng_state())
            assert np.array_equal(np_state[1], np.random.get_state()[1])
        finally:
            logger.close()
        record = read_jsonl(output / "metrics.jsonl")[0]
        assert record["step"] == 0 and record["prior_eval/episodes"] == 2
        assert [e["seed"] for e in report["prior_eval"]["episodes"]] == [10000, 10001]
        assert (output / "evaluation/prior_baseline.json").is_file()
        assert list((output / "evaluation/prior_baseline").glob("attempt_*/episodes.csv"))
    assert means[0] == means[1]


def test_baseline_does_not_change_a_short_training_trajectory(tmp_path):
    actors = []
    for enabled in (False, True):
        cfg = cfg_for(tmp_path / str(enabled), [f"eval.prior_before_training={str(enabled).lower()}",
                      "seed=7", "train.total_env_steps=12", "train.learning_starts=4", "train.batch_size=4",
                      "train.log_interval=4", "replay.capacity=16", "network.actor_hidden=[8,8]",
                      "network.critic_hidden=[8,8]", "algorithm.K_train=2", "algorithm.K_local=0",
                      "algorithm.density.steps=1", "eval.interval=12", "checkpoint.interval=12"])
        runner = Runner(cfg)
        summary = runner.train()
        assert summary["outcome"] == "completed" and summary["env_transitions"] == 12
        actors.append({k: v.clone() for k, v in runner.agent.actor.state_dict().items()})
        if enabled:
            records = read_jsonl(runner.run_dir / "metrics.jsonl")
            assert "prior_eval/return_mean" in records[0] and records[0]["step"] == 0
            assert json.loads((runner.run_dir / "progress.json").read_text())["status"] == "completed"
    assert all(torch.equal(actors[0][k], value) for k, value in actors[1].items())


def test_failed_baseline_never_starts_collection(tmp_path, monkeypatch):
    def failed(cfg, run_dir, logger, step):
        logger.progress("prior_evaluation", 0, 2, unit="episodes")
        raise RuntimeError("controlled baseline failure")

    monkeypatch.setattr("pgofp.evaluation.prior_baseline.evaluate_prior", failed)
    runner = Runner(cfg_for(tmp_path, ["eval.prior_before_training=true", "replay.capacity=16"]))
    with pytest.raises(RuntimeError, match="controlled baseline failure"):
        runner.train()
    meta = json.loads((runner.run_dir / "manifest.json").read_text())
    assert meta["outcome"] == "failed" and meta["summary"]["env_transitions"] == 0
    assert meta["summary"]["gradient_steps"] == 0 and meta["summary"]["failure_stage"] == "prior_evaluation"
