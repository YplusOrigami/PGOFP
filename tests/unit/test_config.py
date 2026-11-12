import pytest

from pgofp.config.compose import compose, config_hash
from pgofp.config.validate import validate_config
from pgofp.core.exceptions import (BackendUnavailableError, ConfigError, UnsupportedCombinationError)


def test_default_compose_valid():
    cfg = compose(["env=toy", "prior=gaussian", "algorithm=gaussian_flame_m"])
    validate_config(cfg)
    assert cfg.source.kind == "gaussian" and cfg.algorithm.name == "gaussian_flame_m"


def test_unknown_override_key_rejected():
    with pytest.raises(ConfigError):
        compose(["env=toy", "prior=gaussian", "train.nonexistent_knob=3"])


def test_unknown_config_file_rejected():
    with pytest.raises(ConfigError):
        compose(["env=does_not_exist"])


def test_gaussian_flame_forces_gaussian_source_and_rejects_conflict():
    cfg = compose(["env=inverted_pendulum", "algorithm=gaussian_flame_m", "prior=gaussian"])
    assert cfg.source.kind == "gaussian"
    with pytest.raises(ConfigError):
        compose(["env=inverted_pendulum", "algorithm=gaussian_flame_m", "prior=gaussian", "source.kind=prior_mixture"])


def test_gaussian_flame_same_hyperparameters_as_prior_flame():
    a = compose(["env=toy", "prior=gaussian", "algorithm=gaussian_flame_m"])
    b = compose(["env=toy", "prior=gaussian", "algorithm=prior_flame_m", "source.kind=gaussian"])
    ra, rb = dict(a.algorithm), dict(b.algorithm)
    ra.pop("name"), rb.pop("name"), ra.pop("forces"), rb.pop("forces")
    assert ra == rb


def test_prior_source_without_prior_rejected():
    cfg = compose(["env=toy", "prior=gaussian", "algorithm=prior_flame_m"])
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_beta_and_sigma_ranges():
    with pytest.raises(ConfigError):
        validate_config(compose(["env=inverted_pendulum", "prior=pid", "source.beta=1.5"]))
    with pytest.raises(ConfigError):
        validate_config(compose(["env=inverted_pendulum", "prior=pid", "source.sigma_prior=0.001"]))


def test_unsupported_task_prior_combination():
    with pytest.raises(UnsupportedCombinationError):
        validate_config(compose(["env=half_cheetah", "prior=pid"]))
    with pytest.raises(UnsupportedCombinationError):
        validate_config(compose(["env=hopper", "prior=mpc"]))


def test_ppo_prior_requires_existing_artifact():
    with pytest.raises(ConfigError):
        validate_config(compose(["env=half_cheetah", "prior=ppo"]))
    with pytest.raises(ConfigError):
        validate_config(compose(["env=half_cheetah", "prior=ppo", "prior.artifact_path=/nonexistent/dir"]))


def test_prior_features_without_prior_rejected():
    with pytest.raises(ConfigError):
        validate_config(compose(["env=toy", "prior=gaussian", "algorithm=gaussian_flame_m", "condition.mode=prior_features"]))


def test_real_fusion_refuses_unverified_config():
    cfg = compose(["env=fusion_stub", "prior=gaussian", "algorithm=gaussian_flame_m", "env.backend=fusion"])
    with pytest.raises(BackendUnavailableError):
        validate_config(cfg)


def test_hash_ignores_run_name_and_changes_with_hyperparameters():
    a = compose(["env=toy", "prior=gaussian", "algorithm=gaussian_flame_m", "run_name=a"])
    b = compose(["env=toy", "prior=gaussian", "algorithm=gaussian_flame_m", "run_name=b"])
    c = compose(["env=toy", "prior=gaussian", "algorithm=gaussian_flame_m", "algorithm.K_train=7"])
    assert config_hash(a) == config_hash(b) != config_hash(c)


def test_auto_controller_config_resolves_per_env():
    cfg = compose(["env=reacher", "prior=pid"])
    assert cfg.prior.controller.name == "reacher_pid"
    assert compose(["env=inverted_pendulum", "prior=pid"]).prior.controller.name == "inverted_pendulum_pid"


def test_experiment_smoke_marks_run():
    cfg = compose(["env=toy", "prior=gaussian", "algorithm=gaussian_flame_m", "experiment=smoke"])
    assert cfg.smoke is True
    assert any("smoke" in w for w in validate_config(cfg))
