from pathlib import Path

import pytest

from pgofp.config.compose import compose, to_container
from pgofp.core.types import MetricRecord
from pgofp.logging.jsonl import read_jsonl, truncate_after_step
from pgofp.logging.logger import Logger


def _cfg(extra=()):
    return compose(["env=toy", "prior=gaussian", "algorithm=gaussian_flame_m", *extra])


def _tb_scalars(run_dir):
    from tensorboard.backend.event_processing import event_accumulator

    ea = event_accumulator.EventAccumulator(str(Path(run_dir) / "tensorboard"), size_guidance={"scalars": 0})
    ea.Reload()
    return {t: [(e.step, e.value) for e in ea.Scalars(t)] for t in ea.Tags()["scalars"]}


def test_jsonl_and_tensorboard_agree_on_values_and_env_step_axis(tmp_path):
    cfg = _cfg()
    lg = Logger(tmp_path, cfg, to_container(cfg), "t")
    for s in (100, 200, 300):
        lg.write_scalars(MetricRecord(s, {"charts/episode_return": s / 10.0, "loss/critic": 1.0 / s}))
    lg.close()
    recs = read_jsonl(tmp_path / "metrics.jsonl")
    tb = _tb_scalars(tmp_path)
    assert [(r["step"], r["charts/episode_return"]) for r in recs] == [(s, s / 10.0) for s in (100, 200, 300)]
    assert [(s, round(v, 6)) for s, v in tb["charts/episode_return"]] == [(s, s / 10.0) for s in (100, 200, 300)]


def test_nonfinite_scalars_never_reach_tensorboard(tmp_path):
    cfg = _cfg()
    lg = Logger(tmp_path, cfg, to_container(cfg), "t")
    lg.write_scalars(MetricRecord(10, {"a": float("nan"), "b": 1.0, "c": float("inf")}))
    lg.close()
    assert set(_tb_scalars(tmp_path)) == {"b"}


def test_non_monotone_step_is_rejected_and_reported(tmp_path):
    cfg = _cfg()
    lg = Logger(tmp_path, cfg, to_container(cfg), "t")
    lg.write_scalars(MetricRecord(200, {"x": 1.0}))
    lg.write_scalars(MetricRecord(100, {"x": 2.0}))
    lg.close()
    assert [r["step"] for r in read_jsonl(tmp_path / "metrics.jsonl")] == [200]
    assert "non_monotone_step" in [e["kind"] for e in read_jsonl(tmp_path / "events.jsonl")]


def test_resume_truncates_records_past_the_checkpoint_and_keeps_them_aside(tmp_path):
    cfg = _cfg()
    lg = Logger(tmp_path, cfg, to_container(cfg), "t")
    for s in (100, 200, 300, 400):
        lg.write_scalars(MetricRecord(s, {"x": float(s)}))
    lg.close()
    lg2 = Logger(tmp_path, cfg, to_container(cfg), "t", resume_step=200)
    lg2.write_scalars(MetricRecord(300, {"x": -1.0}))
    lg2.close()
    steps = [r["step"] for r in read_jsonl(tmp_path / "metrics.jsonl")]
    assert steps == [100, 200, 300]
    assert [r["step"] for r in read_jsonl(tmp_path / "metrics_discarded.jsonl")] == [300, 400]
    assert read_jsonl(tmp_path / "metrics.jsonl")[-1]["x"] == -1.0


def test_wandb_disabled_and_offline_without_package_degrade_gracefully(tmp_path):
    cfg = _cfg(["logging=wandb_offline"])
    lg = Logger(tmp_path, cfg, to_container(cfg), "t")  # wandb not installed in this env -> must not raise
    lg.write_scalars(MetricRecord(1, {"x": 1.0}))
    lg.close()
    kinds = [e["kind"] for e in read_jsonl(tmp_path / "events.jsonl")]
    try:
        import wandb  # noqa: F401
    except ImportError:
        assert "wandb_degraded" in kinds
    assert read_jsonl(tmp_path / "metrics.jsonl")[0]["x"] == 1.0


def test_truncated_final_line_after_crash_is_tolerated(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text('{"step": 1, "x": 1}\n{"step": 2, "x": 2}\n{"step": 3, "x"')
    assert [r["step"] for r in read_jsonl(p)] == [1, 2]
    assert truncate_after_step(p, 1, tmp_path / "d.jsonl") == 1


def test_required_online_logging_fails_before_training_instead_of_degrading(tmp_path, monkeypatch):
    from pgofp.core.exceptions import ConfigError

    class Unavailable:
        error = "network unavailable"
        active = False

        def __init__(self, *args, **kwargs):
            pass

        def finish(self, code=0):
            pass

    monkeypatch.setattr("pgofp.logging.logger.WandbSession", Unavailable)
    cfg = _cfg(["logging=wandb_online", "logging.wandb.require_online=true"])
    with pytest.raises(ConfigError, match="No training was started"):
        Logger(tmp_path, cfg, to_container(cfg), "required-cloud")
    events = read_jsonl(tmp_path / "events.jsonl")
    assert any(e["kind"] == "wandb_required_online_failed" for e in events)


def test_artifact_event_does_not_conflict_with_event_kind(tmp_path):
    cfg = _cfg()
    lg = Logger(tmp_path, cfg, to_container(cfg), "artifact")
    lg.write_artifact("example", tmp_path / "example.pt", "model")
    lg.close()
    event = read_jsonl(tmp_path / "events.jsonl")[-1]
    assert event["kind"] == "artifact" and event["artifact_type"] == "model"


def test_required_online_rejects_an_sdk_offline_run(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from pgofp.core.exceptions import ConfigError

    class Offline:
        error = None
        active = True
        run = SimpleNamespace(settings=SimpleNamespace(mode="offline"))

        def __init__(self, *args, **kwargs):
            pass

        def finish(self, code=0):
            pass

    monkeypatch.setattr("pgofp.logging.logger.WandbSession", Offline)
    cfg = _cfg(["logging=wandb_online", "logging.wandb.require_online=true"])
    with pytest.raises(ConfigError, match="online logging is required"):
        Logger(tmp_path, cfg, to_container(cfg), "required-cloud")


def test_direct_cloud_scalars_disable_tb_sync_and_preserve_duplicate_steps(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from pgofp.logging.wandb import WandbSession

    calls, rows, axes = [], [], []
    fake_run = SimpleNamespace(id="test", log=rows.append, define_metric=lambda *a, **kw: axes.append((a, kw)))

    def init(**kwargs):
        calls.append(kwargs)
        return fake_run

    def forbidden_patch(**kwargs):
        pytest.fail("Direct logging must not enable the TensorBoard watcher")

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init, tensorboard=SimpleNamespace(patch=forbidden_patch)))
    session = WandbSession("online", "test", None, None, [], "test", {}, tmp_path,
                           tb_root=tmp_path / "tensorboard", scalar_source="direct")
    session.scalars(24, {"loss/critic": 1., "bad": float("nan")})
    session.scalars(24, {"eval/return_mean": 2.})
    assert calls[0]["sync_tensorboard"] is False
    assert rows == [{"loss/critic": 1., "charts/env_steps": 24}, {"eval/return_mean": 2., "charts/env_steps": 24}]
    assert axes[-1] == (("*",), {"step_metric": "charts/env_steps"})


def test_disabled_logging_does_not_call_wandb_even_when_environment_requests_online(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    def forbidden(*args, **kwargs):
        pytest.fail("Disabled logging must never authenticate, initialize or upload")

    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=forbidden, Api=forbidden, login=forbidden))
    cfg = _cfg(["logging=local"])
    logger = Logger(tmp_path, cfg, to_container(cfg), "no-cloud")
    logger.write_scalars(MetricRecord(1, {"loss/critic": 2.}))
    logger.close()
    assert not logger.wandb.active
    assert read_jsonl(tmp_path / "metrics.jsonl")[0]["loss/critic"] == 2.
    assert _tb_scalars(tmp_path)["loss/critic"] == [(1, 2.)]


def test_broken_preinstalled_tensorflow_cannot_break_logging_or_preflight(tmp_path):
    """Reproduce the server's ABI error in fresh interpreters, then verify both entry paths."""
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    fake = tmp_path / "preinstalled" / "tensorflow"
    fake.mkdir(parents=True)
    attempted = tmp_path / "tensorflow_import_attempted"
    (fake / "__init__.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(attempted)!r}).write_text('attempted')\n"
        "raise ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(fake.parent), str(root / "src"), env.get("PYTHONPATH", "")])

    def run(code):
        return subprocess.run([sys.executable, "-c", code], env=env, cwd=tmp_path,
                              capture_output=True, text=True, timeout=60, check=False)

    broken = run("from torch.utils.tensorboard import SummaryWriter")
    assert broken.returncode != 0 and "numpy.dtype size changed" in broken.stderr
    assert attempted.is_file()
    attempted.unlink()

    fixed = run(
        "from pathlib import Path\n"
        "import numpy as np\n"
        "from pgofp.config.compose import compose,to_container\n"
        "from pgofp.logging.logger import Logger\n"
        "from pgofp.core.types import MetricRecord\n"
        "from tensorboard.backend.event_processing.event_accumulator import EventAccumulator\n"
        "cfg=compose(['env=toy','prior=gaussian','algorithm=gaussian_flame_m','logging=local'])\n"
        "logger=Logger(Path('run'),cfg,to_container(cfg),'tf-isolation')\n"
        "logger.write_scalars(MetricRecord(9,{'loss/critic':3.5}))\n"
        "logger.histogram('test/histogram',np.asarray([0.,1.,2.]),9)\n"
        "logger.close()\n"
        "ea=EventAccumulator(str(next(Path('run/tensorboard').glob('events.out.tfevents.*'))));ea.Reload()\n"
        "assert [(x.step,x.value) for x in ea.Scalars('loss/critic')]==[(9,3.5)]\n"
        "assert len(ea.Histograms('test/histogram'))==1\n"
    )
    assert fixed.returncode == 0, fixed.stderr
    assert not attempted.exists()

    preflight = run(
        "import importlib.util\n"
        f"spec=importlib.util.spec_from_file_location('bootstrap',{str(root / 'scripts/bootstrap_dependencies.py')!r})\n"
        "module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)\n"
        "module.verify_tensorboard()\n"
    )
    assert preflight.returncode == 0, preflight.stderr
    assert "TensorBoard检查" in preflight.stdout and not attempted.exists()


def test_old_tensorboard_is_rejected_with_an_upgrade_instruction(tmp_path, monkeypatch):
    import importlib.metadata

    from pgofp.logging.tensorboard import TensorBoardWriter

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "2.14.0")
    with pytest.raises(RuntimeError, match="install tensorboard>=2.18"):
        TensorBoardWriter(tmp_path)
