"""Read-only density instrumentation for the beta=1 entropy ablation."""
import sys
from pgofp.algorithms.prior_flame import PriorFlameAgent
from pgofp.density.continuous_proxy import ContinuousInverseProxy
from pgofp.cli import COMMANDS
import torch
_stats = {}
_estimate = ContinuousInverseProxy.estimate_at
_update = PriorFlameAgent.update

def estimate(self, *args, **kwargs):
    out = _estimate(self, *args, **kwargs)
    if out.log_prob_norm.numel():
        _stats['diagnostic/logpi_min'] = float(out.log_prob_norm.min())
        _stats['diagnostic/logpi_max'] = float(out.log_prob_norm.max())
        _stats['diagnostic/logpi_abs_max'] = float(out.log_prob_norm.abs().max())
        _stats['diagnostic/logpi_nonfinite_count'] = float((~torch.isfinite(out.log_prob_norm)).sum())
    if out.inverse_residual is not None:
        _stats['diagnostic/inverse_residual_max'] = float(out.inverse_residual.max())
    return out

def update(self, *args, **kwargs):
    _stats.clear()
    out = _update(self, *args, **kwargs)
    out.metrics.update(_stats)
    return out

ContinuousInverseProxy.estimate_at = estimate
PriorFlameAgent.update = update
if __name__ == '__main__':
    sys.exit(COMMANDS['train'](sys.argv[1:]))
