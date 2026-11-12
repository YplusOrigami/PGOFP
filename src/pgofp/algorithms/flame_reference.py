"""flame_reference: adapter around a PINNED upstream FLAME checkout (docs/reproduction_notes.md).

Refuses to start without an explicit upstream commit and an importable pinned checkout: a floating `main` or an
unpinned install is never silently treated as the paper version.  No upstream code is vendored or fabricated here.
"""
from __future__ import annotations

from ..core.exceptions import BackendUnavailableError, ConfigError


class FlameReferenceAgent:
    def __init__(self, cfg, env_spec, builder, device):
        params = dict(cfg.algorithm.params)
        commit = params.get("upstream_commit")
        path = params.get("upstream_path")
        if not commit or not path:
            raise ConfigError("flame_reference requires algorithm.params.upstream_commit and algorithm.params.upstream_path "
                              "(a pinned checkout of github.com/lzqw/FLAME); refusing to start with an unpinned reference")
        raise BackendUnavailableError(
            f"flame_reference adapter is not implemented in this repository version (pinned commit {commit}); "
            "see docs/reproduction_notes.md. Use gaussian_flame_m as the same-implementation Gaussian control.")
