"""continuous_inverse_proxy (default, docs section 7.2).

Given the latent action z_a (t=0):
  1. integrate dz/dt = f(z,t) = u(z,t,t;c) from t=0 to t=1 -> eps_hat (source end),
     together with D = int_0^1 div_z f dt.
  2. log pi_z(z_a|c) = log p_src(eps_hat|c) + D
  3. log pi_norm(a|c) = log pi_z(z_a|c) - log|J_tanh(z_a)|
Direction check (analytic): f = k z in 1-D gives z(1) = z0 e^k, D = k, log p_0(z0) = log p_1(z0 e^k) + k.
Heun with N steps costs 2N field evaluations (logged as `nfe`, never conflated with N).
"""
from __future__ import annotations

from typing import Optional

import torch

from ..core.types import DensityResult
from ..math.action_transform import log_abs_det_tanh
from ..math.integration import integrate_with_divergence


class ContinuousInverseProxy:
    method = "continuous_inverse_proxy"

    def __init__(self, steps: int = 5, solver: str = "heun", trace: str = "hutchinson", n_trace_samples: int = 1,
                 chunk_size: int = 8192):
        self.steps = steps
        self.solver = solver
        self.trace = trace
        self.n_trace_samples = n_trace_samples
        self.chunk_size = chunk_size

    def _one_chunk(self, z_a, cond, net, source, generator, eps_ref):
        def field(z, t):
            tt = torch.full((z.shape[0], 1), t, device=z.device, dtype=z.dtype)
            return net(z, tt, tt, cond)

        eps_hat, D, nfe = integrate_with_divergence(
            field, z_a, steps=self.steps, solver=self.solver, trace=self.trace,
            n_trace_samples=self.n_trace_samples, generator=generator,
        )
        with torch.no_grad():
            lp_src = source.log_prob(eps_hat)
            lp_z = lp_src + D
            lp_norm = lp_z - log_abs_det_tanh(z_a)
            resid = None if eps_ref is None else (eps_hat - eps_ref).norm(dim=-1)
        return lp_norm, lp_z, eps_hat, resid, nfe

    def estimate_at(self, z_a: torch.Tensor, cond: torch.Tensor, net, source, generator: Optional[torch.Generator] = None,
                    eps_ref: Optional[torch.Tensor] = None) -> DensityResult:
        B = z_a.shape[0]
        outs = []
        for s in range(0, B, self.chunk_size):
            sl = slice(s, min(B, s + self.chunk_size))
            src = source if B <= self.chunk_size else source.index(sl)
            outs.append(self._one_chunk(z_a[sl].detach(), cond[sl], net, src, generator,
                                        None if eps_ref is None else eps_ref[sl]))
        lp_norm = torch.cat([o[0] for o in outs])
        lp_z = torch.cat([o[1] for o in outs])
        eps_hat = torch.cat([o[2] for o in outs])
        resid = None if outs[0][3] is None else torch.cat([o[3] for o in outs])
        finite = torch.isfinite(lp_norm)
        return DensityResult(
            log_prob_norm=lp_norm.detach(), method=self.method, is_exact_for_sampled_policy=False,
            nfe=outs[0][4], inverse_residual=resid, finite_mask=finite, log_prob_latent=lp_z.detach(),
            eps_hat=eps_hat.detach(),
        )
