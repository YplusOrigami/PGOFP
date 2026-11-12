import numpy as np
import pytest
import torch

from pgofp.core.exceptions import ConfigError
from pgofp.core.types import CONTEXT_FIELDS, ContextSpec
from pgofp.data.replay_buffer import ReplayBuffer


def _rec(n, d=2, o=3, M=1, base=0.0):
    r = {}
    shapes = ContextSpec(obs_dim=o, action_dim=d, n_prior=M).shapes()
    for k, shp in shapes.items():
        r[k] = np.full((n, *shp), base, np.float32)
        r["next_" + k] = np.full((n, *shp), base + 0.5, np.float32)
    r.update({"action_policy_norm": np.zeros((n, d), np.float32), "action_command_norm": np.zeros((n, d), np.float32),
              "reward": np.arange(n, dtype=np.float32) + base, "terminated": np.zeros(n, np.float32), "truncated": np.zeros(n, np.float32),
              "bootstrap_mask": np.ones(n, np.float32)})
    return r


def test_ring_overwrite_and_size():
    buf = ReplayBuffer(5, ContextSpec(obs_dim=3, action_dim=2, n_prior=1))
    buf.add_batch(_rec(4, base=0))
    buf.add_batch(_rec(3, base=100))
    assert len(buf) == 5 and buf.ptr == 2 and buf.total_added == 7
    assert sorted(buf.data["reward"].tolist()).count(100.0) == 1  # newest entries overwrote the oldest
    # capacity 5: batch 2 wrote slots 4, 0, 1
    assert buf.data["obs_raw"][[4, 0, 1], 0].tolist() == [100.0] * 3 and buf.data["obs_raw"][[2, 3], 0].tolist() == [0.0, 0.0]


def test_current_and_next_conditions_are_paired():
    buf = ReplayBuffer(10, ContextSpec(obs_dim=3, action_dim=2, n_prior=1))
    buf.add_batch(_rec(10, base=7))
    b = buf.sample(6, np.random.default_rng(0))
    assert torch.all(b.ctx.obs_raw == 7.0) and torch.all(b.next_ctx.obs_raw == 7.5)
    assert b.ctx.prior_mu.shape == (6, 1, 2) and b.reward.shape == (6,) and b.bootstrap_mask.shape == (6,)


def test_rejects_wrong_shape_and_missing_field():
    buf = ReplayBuffer(10, ContextSpec(obs_dim=3, action_dim=2, n_prior=1))
    bad = _rec(4)
    bad["obs_raw"] = np.zeros((4, 5), np.float32)
    with pytest.raises(ConfigError):
        buf.add_batch(bad)
    miss = _rec(4)
    del miss["reward"]
    with pytest.raises(ConfigError):
        buf.add_batch(miss)
    with pytest.raises(ConfigError):
        ReplayBuffer(3, ContextSpec(obs_dim=3, action_dim=2)).sample(2, np.random.default_rng(0))


def test_zero_width_fields_roundtrip(tmp_path):
    buf = ReplayBuffer(8, ContextSpec(obs_dim=3, action_dim=2, n_prior=0))
    buf.add_batch(_rec(5, M=0))
    buf.save(tmp_path / "r.npz")
    b2 = ReplayBuffer(8, ContextSpec(obs_dim=3, action_dim=2, n_prior=0))
    b2.load(tmp_path / "r.npz")
    assert len(b2) == 5 and b2.ptr == buf.ptr
    assert np.array_equal(b2.data["reward"][:5], buf.data["reward"][:5])
    with pytest.raises(ConfigError):
        ReplayBuffer(9, ContextSpec(obs_dim=3, action_dim=2, n_prior=0)).load(tmp_path / "r.npz")


def test_single_sample_keeps_batch_dim():
    buf = ReplayBuffer(4, ContextSpec(obs_dim=3, action_dim=2, n_prior=1))
    buf.add_batch(_rec(4))
    b = buf.sample(1, np.random.default_rng(0))
    assert b.ctx.obs_raw.shape == (1, 3) and b.reward.shape == (1,)
