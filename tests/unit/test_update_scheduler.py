import pytest

from pgofp.training.update_scheduler import UpdateScheduler


def _run(utd, n_envs, steps, learning_starts=0):
    s = UpdateScheduler(utd, learning_starts)
    total, env = 0, 0
    for _ in range(steps):
        env += n_envs
        total += s.n_updates(n_envs, env)
    return total


@pytest.mark.parametrize("utd", [0.2, 1.0, 2.0])
def test_updates_scale_with_real_transitions_not_vector_steps(utd):
    one = _run(utd, 1, 1000)
    eight = _run(utd, 8, 125)  # same 1000 real transitions
    assert abs(one - 1000 * utd) <= 1 and abs(eight - 1000 * utd) <= 1


def test_fractional_utd_is_never_truncated_to_zero():
    assert _run(0.2, 1, 100) == 20


def test_warmup_leaves_no_update_debt():
    s = UpdateScheduler(1.0, learning_starts=500)
    env, total_after = 0, 0
    for _ in range(500):
        env += 1
        assert s.n_updates(1, env) in (0, 1)
    # first post-warmup call must NOT repay 500 historical updates
    env += 1
    n = s.n_updates(1, env)
    assert n <= 2


def test_actor_interval_and_state_roundtrip():
    s = UpdateScheduler(1.0, 0, actor_update_interval=3)
    flags = []
    for _ in range(6):
        flags.append(s.do_actor())
        s.tick()
    assert flags == [True, False, False, True, False, False]
    s2 = UpdateScheduler(1.0, 0, 3)
    s2.load_state_dict(s.state_dict())
    assert s2.gradient_steps == 6


def test_negative_utd_rejected():
    with pytest.raises(ValueError):
        UpdateScheduler(-1.0, 0)
