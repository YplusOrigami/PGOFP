from .base import check_finite  # noqa: F401
from .continuous_proxy import ContinuousInverseProxy  # noqa: F401
from .local_meanflow_proxy import LocalMeanFlowProxy  # noqa: F401


def make_estimator(method: str, **kw):
    if method == "continuous_inverse_proxy":
        keys = ("steps", "solver", "trace", "n_trace_samples", "chunk_size")
        return ContinuousInverseProxy(**{k: kw[k] for k in keys if k in kw})
    if method == "local_meanflow_proxy":
        keys = ("grid", "logdet", "max_iter", "tol")
        return LocalMeanFlowProxy(**{k: kw[k] for k in keys if k in kw})
    from ..core.exceptions import ConfigError

    raise ConfigError(f"unknown density method {method!r}")
