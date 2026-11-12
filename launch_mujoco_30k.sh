#!/usr/bin/env bash
# Default only writes the plan. Explicit --run prepares priors and starts training.
set -euo pipefail
PGOFP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PGOFP_ROOT"
if [[ -n "${PGOFP_PYTHON:-}" ]]; then
  PGOFP_PYTHON_BIN="$PGOFP_PYTHON"
elif [[ -n "${VIRTUAL_ENV:-}${CONDA_PREFIX:-}" ]]; then
  PGOFP_PYTHON_BIN="$(command -v python)"
elif [[ -x "$PGOFP_ROOT/.venv/bin/python" ]]; then
  PGOFP_PYTHON_BIN="$PGOFP_ROOT/.venv/bin/python"
else
  echo '[0/6 环境准备] 正在创建项目 .venv，并复用平台已有依赖。'
  python3 -m venv --system-site-packages "$PGOFP_ROOT/.venv"
  PGOFP_PYTHON_BIN="$PGOFP_ROOT/.venv/bin/python"
fi
export PYTHONPATH="$PGOFP_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PGOFP_ROOT/outputs/.cache/matplotlib}"
"$PGOFP_PYTHON_BIN" "$PGOFP_ROOT/scripts/bootstrap_dependencies.py"
if [[ $# -eq 0 ]]; then
  set -- --plan
fi
exec "$PGOFP_PYTHON_BIN" "$PGOFP_ROOT/scripts/run_mujoco_suite.py" "$@"
