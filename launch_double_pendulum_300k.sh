#!/usr/bin/env bash
# No arguments: run the 18-job, 300k suite on CUDA with three concurrent jobs.
# --plan: write configs only. --validate --device cpu: isolated short validation.
set -euo pipefail
PGOFP_DOUBLE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -eq 0 ]]; then
  set -- --run
fi
exec bash "$PGOFP_DOUBLE_ROOT/launch_mujoco_30k.sh" \
  --spec "$PGOFP_DOUBLE_ROOT/configs/experiment/double_pendulum_prior_beta_300k.yaml" "$@"
