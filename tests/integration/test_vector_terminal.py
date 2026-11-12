"""Termination semantics and per-slot controller state under explicit-reset vector pools."""
import numpy as np
import pytest
import torch

from pgofp.config.compose import compose
from pgofp.controllers.manager import ControllerManager
from pgofp.core.types import ContextSpec
from pgofp.data.replay_buffer import ReplayBuffer
from pgofp.envs.factory import make_pool
from pgofp.envs.toy import ToyReach
from pgofp.envs.vector import SyncEnvPool
from pgofp.training.collector import Collector
from pgofp.utils.normalization import RunningNorm
from pgofp.utils.seed import RngStreams
from pgofp.controllers.base import BaseController
from pgofp.core.types import ControllerPrediction
from pgofp.core.specs import SourceSpec


class _CountingController(BaseController):
    """state = number of advances since reset; center encodes it so replay can verify the pairing."""
    n_components, ctrl_dim, version = 1, 1, "counting"

    def initial_state(self, n):
        return {"k": np.zeros((n, 1))}

    def predict(self, obs, ref, state):
        n = obs.shape[0]
        return ControllerPrediction(center_norm=np.tile(np.tanh(state["k"] / 10.0)[:, None, :], (1, 1, 2)))

    def advance(self, state, obs, ref, cmd):
        return {"k": state["k"] + 1}

    def features(self, state):
        return state["k"].astype(np.float32)


class _Agent:
    def act(self, ctx, spec, norm, gen, deterministic=False, K=1):
        from pgofp.core.types import ActionResult
        a = torch.zeros(len(ctx), 2)
        return ActionResult(a, a, a)


def _make(n_envs=3, horizon=5):
    cfg = compose(["env=toy", "prior=gaussian", "algorithm=gaussian_flame_m"])
    pool = SyncEnvPool(lambda i: ToyReach(2, horizon=horizon, seed=i), n_envs, 0)
    mgr = ControllerManager([_CountingController()], n_envs, 2)
    cs = ContextSpec(obs_dim=2, action_dim=2, ctrl_dim=1, n_prior=1)
    replay = ReplayBuffer(500, cs)
    col = Collector(cfg, pool, mgr, pool.spec, replay, RunningNorm(2), RngStreams(0), torch.device("cpu"))
    col.reset()
    return col, replay, pool, mgr


def test_truncation_keeps_bootstrap_and_uses_final_obs_not_reset_obs():
    col, replay, pool, _ = _make(1, horizon=4)
    spec = SourceSpec()
    final_obs = None
    for _ in range(4):
        col.step(_Agent(), spec, RunningNorm(2).snapshot(), 0, False)
    i = 3  # last transition: time-limit truncation
    assert replay.data["truncated"][i] == 1 and replay.data["terminated"][i] == 0
    assert replay.data["bootstrap_mask"][i] == 1.0  # truncated -> bootstrap
    # next_obs of the last transition is the real final observation and differs from the fresh reset obs that starts the next episode
    assert not np.allclose(replay.data["next_obs_raw"][i], col.cur["obs_raw"][0])


def test_termination_stops_bootstrap():
    cfg = compose(["env=toy", "prior=gaussian", "algorithm=gaussian_flame_m"])
    pool = SyncEnvPool(lambda i: ToyReach(2, horizon=50, seed=i), 1, 0)
    mgr = ControllerManager([], 1, 2)
    replay = ReplayBuffer(100, ContextSpec(obs_dim=2, action_dim=2, n_prior=0))
    col = Collector(cfg, pool, mgr, pool.spec, replay, RunningNorm(2), RngStreams(0), torch.device("cpu"))
    col.reset()
    pool.envs[0].s = np.array([7.0, 0.0], np.float32)  # |s| stays > 6 after one step -> genuine termination

    class Push:
        def act(self, ctx, spec, norm, gen, deterministic=False, K=1):
            from pgofp.core.types import ActionResult
            a = torch.ones(len(ctx), 2)
            return ActionResult(a, a, a)

    for _ in range(30):
        col.step(Push(), SourceSpec(), RunningNorm(2).snapshot(), 0, False)
        if replay.data["terminated"][: len(replay)].any():
            break
    idx = int(np.argmax(replay.data["terminated"][: len(replay)]))
    assert replay.data["terminated"][idx] == 1 and replay.data["bootstrap_mask"][idx] == 0.0


def test_controller_state_per_slot_and_condition_pairing():
    col, replay, pool, mgr = _make(3, horizon=5)
    for _ in range(7):
        col.step(_Agent(), SourceSpec(), RunningNorm(2).snapshot(), 0, False)
    k = replay.data["ctrl_feat"][: len(replay), 0]
    nk = replay.data["next_ctrl_feat"][: len(replay), 0]
    env_ids = replay.data["env_id"][: len(replay)]
    ep = replay.data["step_in_episode"][: len(replay)]
    # every slot restarts its own counter at an episode boundary: the stored counter always equals step_in_episode - 1
    assert np.array_equal(k, ep - 1)
    # next-condition counter is one more (true next state, computed BEFORE any reset)
    assert np.array_equal(nk, k + 1)
    # slots hit the horizon simultaneously here; after reset all counters are back at 0 -> no cross-slot leakage
    assert np.array_equal(np.unique(env_ids), np.arange(3))


def test_episode_stats_are_per_episode_weighted():
    col, replay, pool, _ = _make(2, horizon=4)
    for _ in range(9):
        col.step(_Agent(), SourceSpec(), RunningNorm(2).snapshot(), 0, False)
    stats = col.drain_episode_stats()
    assert stats["charts/episodes"] == 4  # 2 envs x 2 finished episodes
    assert stats["charts/episode_length"] == 4.0
    assert col.drain_episode_stats().get("charts/episodes") is None  # drained


def test_next_prior_matches_requery_of_true_next_state():
    """prior params stored for next condition == controller queried at the true next state (not the reset one)."""
    col, replay, pool, mgr = _make(1, horizon=3)
    for _ in range(3):
        col.step(_Agent(), SourceSpec(), RunningNorm(2).snapshot(), 0, False)
    # on the truncating step the stored next prior reflects counter 3 (not 0 as after reset)
    mu_next = replay.data["next_prior_mu"][2, 0, 0]
    assert abs(mu_next - np.arctanh(np.tanh(3 / 10.0))) < 1e-4
    assert abs(col.cur["prior_mu"][0, 0, 0]) < 1e-6  # fresh episode starts from counter 0
