import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pgofp.config.compose import compose
from pgofp.core.exceptions import ResumeIncompatibleError
from pgofp.training.checkpoint import CheckpointManager
from pgofp.training.runner import Runner

BASE = ["env=toy", "prior=gaussian", "algorithm=gaussian_flame_m", "experiment=smoke", "train.learning_starts=100", "train.log_interval=100",
        "eval.interval=100000", "checkpoint.interval=300", "algorithm.K_train=4", "network.actor_hidden=[32,32]", "network.critic_hidden=[32,32]",
        "algorithm.density.steps=2", "eval.episodes=2"]


def _cfg(tmp_path, total, extra=()):
    return compose(BASE + [f"output_dir={tmp_path}", f"train.total_env_steps={total}", *extra])


def _weights(r):
    return {k: v.clone() for k, v in r.agent.actor.state_dict().items()}


def test_exact_resume_matches_uninterrupted_run(tmp_path):
    """Toy env supports exact snapshots: stop at 600, resume to 900 == one straight run to 900 (weights equal)."""
    straight = Runner(_cfg(tmp_path / "a", 900)).build()
    straight.train()
    w_ref = _weights(straight)

    part1 = Runner(_cfg(tmp_path / "b", 600)).build()
    part1.train()
    ckpt = part1.run_dir / "checkpoints" / "last.pt"
    cfg2 = _cfg(tmp_path / "b", 900, ["resume_mode=exact_resume"])
    part2 = Runner(cfg2, run_dir=part1.run_dir).build()
    state = CheckpointManager(part1.run_dir).load(ckpt)
    part2.load_state(state, "exact_resume")
    assert part2.env_transitions == 600
    part2.train()
    w2 = _weights(part2)
    assert all(torch.allclose(w_ref[k], w2[k], atol=1e-6) for k in w_ref)
    assert part2.env_transitions == 900


def test_resume_log_axis_monotone_and_no_overwrite(tmp_path):
    p1 = Runner(_cfg(tmp_path, 600)).build()
    p1.train()
    n_before = len((p1.run_dir / "metrics.jsonl").read_text().splitlines())
    p2 = Runner(_cfg(tmp_path, 900, ["resume_mode=episode_restart"]), run_dir=p1.run_dir).build()
    p2.load_state(CheckpointManager(p1.run_dir).load(p1.run_dir / "checkpoints" / "last.pt"), "episode_restart")
    p2.train()
    steps = [json.loads(l)["step"] for l in (p2.run_dir / "metrics.jsonl").read_text().splitlines()]
    assert steps == sorted(steps) and steps[-1] >= 900 and len(steps) > n_before
    events = [json.loads(l)["kind"] for l in (p2.run_dir / "events.jsonl").read_text().splitlines()]
    assert "resume" in events


def test_incompatible_changes_reject_resume_but_weights_only_loads(tmp_path):
    p1 = Runner(_cfg(tmp_path, 400)).build()
    p1.train()
    state = CheckpointManager(p1.run_dir).load(p1.run_dir / "checkpoints" / "last.pt")
    for bad in (["network.actor_hidden=[16,16]"], ["env.kwargs.action_dim=3"], ["algorithm.actor_objective=reverse_no_q"]):
        r = Runner(_cfg(tmp_path / "x", 800, bad)).build()
        with pytest.raises((ResumeIncompatibleError, RuntimeError, KeyError)):
            r.load_state(state, "episode_restart")
    r = Runner(_cfg(tmp_path / "y", 800)).build()
    r.load_state(state, "weights_only")
    assert r.env_transitions == 0  # fresh budget/clock
    assert all(torch.allclose(p1.agent.actor.state_dict()[k], v) for k, v in r.agent.actor.state_dict().items())


def test_exact_resume_refused_without_env_snapshot(tmp_path):
    """MuJoCo has no snapshot API in the adapter: exact_resume must refuse instead of pretending."""
    pytest.importorskip("gymnasium")
    cfg = compose(["env=inverted_pendulum", "prior=gaussian", "algorithm=gaussian_flame_m", "experiment=smoke", f"output_dir={tmp_path}",
                   "train.total_env_steps=300", "train.learning_starts=100", "eval.interval=100000", "checkpoint.interval=300",
                   "algorithm.K_train=2", "network.actor_hidden=[16]", "network.critic_hidden=[16]", "algorithm.density.steps=1",
                   "eval.episodes=1", "train.log_interval=100"])
    r = Runner(cfg).build()
    r.train()
    cfg2 = compose(["env=inverted_pendulum", "prior=gaussian", "algorithm=gaussian_flame_m", "experiment=smoke", f"output_dir={tmp_path}",
                    "train.total_env_steps=600", "resume_mode=exact_resume", "algorithm.K_train=2", "network.actor_hidden=[16]",
                    "network.critic_hidden=[16]", "algorithm.density.steps=1", "eval.episodes=1", "train.learning_starts=100",
                    "eval.interval=100000", "train.log_interval=100"])
    r2 = Runner(cfg2, run_dir=r.run_dir).build()
    r2.load_state(CheckpointManager(r.run_dir).load(r.run_dir / "checkpoints" / "last.pt"), "exact_resume")
    with pytest.raises(ResumeIncompatibleError):
        r2.train()


def test_checkpoint_checksum_and_atomic_save(tmp_path):
    p1 = Runner(_cfg(tmp_path, 300)).build()
    p1.train()
    ck = p1.run_dir / "checkpoints" / "last.pt"
    good = ck.read_bytes()
    ck.write_bytes(good[:-100] + b"x" * 100)
    with pytest.raises(ResumeIncompatibleError):
        CheckpointManager(p1.run_dir).load(ck)
    ck.write_bytes(good)
    assert not list((p1.run_dir / "checkpoints").glob("*.tmp"))
    CheckpointManager(p1.run_dir).load(ck)
