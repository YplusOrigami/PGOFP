"""Install only missing/incompatible project dependencies in the selected Python environment."""
from __future__ import annotations

import importlib
import importlib.metadata
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pip_install(requirements: list[str], constraints: str | None = None) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"]
    if constraints:
        cmd += ["--constraint", constraints]
    print(f"[0/6 依赖安装] 正在安装：{', '.join(requirements)}", flush=True)
    subprocess.run([*cmd, *requirements], check=True)


def required_packages() -> list[str]:
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        try:
            import tomli as tomllib
        except ImportError:
            pip_install(["tomli>=2"])
            import tomli as tomllib
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    return project["dependencies"] + [r for extra in ("mujoco", "wandb", "analysis")
                                      for r in project["optional-dependencies"][extra]]


def missing_requirements(requirements: list[str]) -> list[str]:
    from packaging.requirements import Requirement

    missing = []
    for index, text in enumerate(requirements, 1):
        req = Requirement(text)
        if req.marker and not req.marker.evaluate():
            continue
        try:
            version = importlib.metadata.version(req.name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        ok = version is not None and req.specifier.contains(version, prereleases=True)
        print(f"[0/6 依赖检查 {index}/{len(requirements)}] {req.name}: "
              f"{version or '未安装'} {'OK' if ok else '待安装/补齐'}", flush=True)
        if not ok:
            # Gymnasium's MuJoCo extra is listed explicitly by the project as well.
            missing.append(text)
    return missing


def verify_tensorboard() -> None:
    """Exercise the real logger, not just tensorboard's lightweight top-level import."""
    project_src = str(ROOT / "src")
    if project_src not in sys.path:
        sys.path.insert(0, project_src)
    from pgofp.logging.tensorboard import TensorBoardWriter, use_tensorboard_stub

    use_tensorboard_stub()

    # Import the reader after the logger has selected TensorBoard's TF stub.
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    with tempfile.TemporaryDirectory(prefix="pgofp-tensorboard-check-") as tmp:
        path = Path(tmp)
        writer = TensorBoardWriter(path)
        try:
            writer.scalars(1, {"bootstrap/logging_check": 1.})
        finally:
            writer.close()
        # Select a real event file explicitly, ignoring macOS AppleDouble metadata.
        event_file = next(path.glob("events.out.tfevents.*"))
        reader = EventAccumulator(str(event_file))
        reader.Reload()
        records = reader.Scalars("bootstrap/logging_check")
        if len(records) != 1 or records[0].step != 1 or records[0].value != 1.:
            raise RuntimeError("TensorBoard event write/read verification failed")
    print("[0/6 TensorBoard检查] 实际写入/回读通过，使用无TensorFlow的轻量后端。", flush=True)


def main() -> int:
    if not (3, 10) <= sys.version_info[:2] < (3, 13):
        print("[ERROR 0/6 依赖检查] 本项目需要 Python 3.10–3.12；请设置 PGOFP_PYTHON。", file=sys.stderr)
        return 1
    print(f"[0/6 依赖检查] Python={sys.executable}", flush=True)
    try:
        try:
            import packaging.requirements  # noqa: F401
        except ImportError:
            pip_install(["packaging>=23"])
        requirements = required_packages()
        missing = missing_requirements(requirements)
        if missing:
            # Preserve an installed compatible CUDA build even when another
            # package's dependency resolver considers replacing torch.
            try:
                torch_version = importlib.metadata.version("torch")
            except importlib.metadata.PackageNotFoundError:
                torch_version = None
            with tempfile.TemporaryDirectory(prefix="pgofp-dependencies-") as tmp:
                constraint = Path(tmp) / "constraints.txt"
                preserve_torch = torch_version and not any(r.split(">=")[0] == "torch" for r in missing)
                if preserve_torch:
                    constraint.write_text(f"torch=={torch_version}\n")
                pip_install(missing, str(constraint) if preserve_torch else None)
        # Metadata alone can hide a broken binary installation. Report its exact
        # module rather than silently attempting to replace a working CUDA stack.
        for name in ("numpy", "torch", "omegaconf", "yaml", "tensorboard", "gymnasium", "mujoco", "wandb", "pandas", "matplotlib", "scipy"):
            try:
                importlib.import_module(name)
            except Exception as exc:
                raise RuntimeError(f"依赖 {name} 无法导入：{type(exc).__name__}: {exc}") from exc
        print("[0/6 TensorBoard检查] 正在测试实际事件写入与回读。", flush=True)
        verify_tensorboard()
        print("[0/6 依赖检查] 完成；所需依赖及TensorBoard写入可用。", flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001 -- bootstrap boundary must identify dependency failures
        print(f"[ERROR 0/6 依赖安装] {type(exc).__name__}: {exc}\n"
              f"当前 Python：{sys.executable}；请检查该环境的安装权限、pip 与下载网络。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
