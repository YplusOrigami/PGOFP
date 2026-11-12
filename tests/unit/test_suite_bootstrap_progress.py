import importlib.util
import json
import sys
from pathlib import Path

from pgofp.experiments.benchmark_suite import build_plans, default_prior_paths, load_suite, phase
from pgofp.experiments.launcher import Launcher
from pgofp.experiments.manifest import MatrixManifest


def bootstrap():
    path = Path(__file__).resolve().parents[2] / "scripts/bootstrap_dependencies.py"
    spec = importlib.util.spec_from_file_location("pgofp_bootstrap_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bootstrap_does_not_install_satisfied_dependencies(monkeypatch):
    module = bootstrap()
    monkeypatch.setattr(module, "required_packages", lambda: ["torch>=2.3", "numpy>=1.26"])
    monkeypatch.setattr(module, "pip_install", lambda *a: (_ for _ in ()).throw(AssertionError("Unexpected pip invocation")))
    assert module.main() == 0


def test_missing_dependency_install_preserves_existing_cuda_torch(monkeypatch):
    module = bootstrap()
    monkeypatch.setattr(module, "required_packages", lambda: ["scipy"])
    monkeypatch.setattr(module, "missing_requirements", lambda requirements: requirements)
    original = module.importlib.metadata.version
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "2.5.1+cu121" if name == "torch" else original(name))
    calls = []
    monkeypatch.setattr(module, "pip_install", lambda requirements, constraints=None: calls.append((requirements, Path(constraints).read_text())))
    assert module.main() == 0
    assert calls == [(["scipy"], "torch==2.5.1+cu121\n")]


def test_progress_reports_failed_child_stage_and_log_path(tmp_path, capsys):
    suite = load_suite()
    plan = build_plans(suite, default_prior_paths(suite, tmp_path))[0]
    man = MatrixManifest(tmp_path / "out", "test", [plan.to_record()])
    script = tmp_path / "fake_child.py"
    # Simulate an owned child failing during baseline evaluation; no training occurs.
    script.write_text("import pathlib,json,sys\n"
                      "args=dict(x.split('=',1) for x in sys.argv[1:] if '=' in x)\n"
                      "d=pathlib.Path(args['output_dir'])/(args['run_name']+'__fixture');d.mkdir(parents=True)\n"
                      "(d/'manifest.json').write_text(json.dumps({'outcome':'failed'}))\n"
                      "(d/'progress.json').write_text(json.dumps({'stage':'prior_evaluation','current':1,'total':10,'status':'failed','message':'test error'}))\n"
                      "print('Controlled child error',flush=True)\nsys.exit(7)\n")
    Launcher(man, tmp_path, python=sys.executable, script=str(script), max_retries=0,
             show_progress=True).run(poll_s=.01)
    text = capsys.readouterr().out
    assert man.runs[0]["status"] == "failed" and man.runs[0]["failure_stage"] == "prior_evaluation"
    assert "出错步骤=prior_evaluation" in text and "Controlled child error" in text
    assert str(tmp_path / "out/logs") in text and "失败=1" in text
    assert json.loads((tmp_path / "out/manifest.json").read_text())["runs"][0]["exit_code"] == 7


def test_failed_phase_is_recorded_with_a_redacted_error(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setenv("WANDB_API_KEY", "test-secret")
    with pytest.raises(RuntimeError), phase(tmp_path, 2, "PPO先验准备"):
        raise RuntimeError("test-secret installation failed")
    state = json.loads((tmp_path / "suite_status.json").read_text())
    assert state["stage"] == 2 and state["status"] == "failed"
    assert "test-secret" not in state["error"] and "[REDACTED]" in state["error"]


def test_short_phases_remain_visible_between_parent_polls(tmp_path, capsys):
    from pgofp.logging.progress import ProgressDisplay, ProgressFile

    path = tmp_path / "progress.json"
    progress = ProgressFile(path, True)
    progress.update("prior_evaluation", 0, 10)
    progress.update("prior_evaluation", 10, 10)
    progress.update("warmup", 0, 5000)
    progress.update("flow_training", 5000, 30000)
    display = ProgressDisplay("test")
    display.poll(path)
    text = capsys.readouterr().out
    assert "先验基线评估" in text and "10/10 (100%)" in text
    assert "热身采样" in text and "Flow训练" in text
