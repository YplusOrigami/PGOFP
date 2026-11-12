"""local_meanflow_proxy (optional, docs section 7.3).

Discrete map from source time 1 down to 0 over intervals [r_i, t_i] with h_i = t_i - r_i:
    T_i(z_t) = z_t - h_i * u(z_t, r_i, t_i; c).
For a QUERIED action z_a we invert from the target end: for i descending from the interval touching 0,
solve  y - h_i u(y, r_i, t_i) = z_r  for y = z_t (fixed-point iteration y <- z_r + h_i u(y, ...)), recover
the source end, and
    log pi_z(z_a) = log p_src(z_1) - sum_i log|det(I - h_i J_z u_i)|.
Local inversion is unique only when h_i * Lip(u) < 1 (contraction); this is NOT assumed globally: the
iteration reports a residual and failures are counted instead of being hidden. `is_exact_for_sampled_policy`
is False by default (a one-interval map [0,1] with small h*Lip would be exact, otherwise it is not).
The first implementation uses the full Jacobian (low-dimensional action spaces); a trace approximation
-h*trace(J) is available through `logdet="trace"` with its truncation error left to the caller to diagnose.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from ..core.types import DensityResult
from ..math.action_transform import log_abs_det_tanh


class LocalMeanFlowProxy:
    method = "local_meanflow_proxy"

    def __init__(self, grid: Tuple[float, ...] = (0.0, 0.5, 1.0), logdet: str = "full", max_iter: int = 50,
                 tol: float = 1e-6):
        if logdet not in ("full", "trace"):
            raise ValueError("logdet must be full|trace")
        self.grid = tuple(grid)
        self.logdet = logdet
        self.max_iter = max_iter
        self.tol = tol

    def _invert_interval(self, net, z_r, r, t, cond):
        """Fixed-point solve y - h u(y) = z_r. Returns y, residual."""
        h = (t - r).view(-1, 1)
        y = z_r.clone()
        for _ in range(self.max_iter):
            with torch.no_grad():
                y_new = z_r + h * net(y, r, t, cond)
            if (y_new - y).abs().max() < self.tol:
                y = y_new
                break
            y = y_new
        with torch.no_grad():
            res = (y - h * net(y, r, t, cond) - z_r).norm(dim=-1)
        return y, res

    def _logdet_interval(self, net, y, r, t, cond):
        h = (t - r).view(-1, 1)
        B, d = y.shape
        with torch.enable_grad():
            yr = y.detach().requires_grad_(True)
            u = net(yr, r, t, cond)
            cols = [torch.autograd.grad(u[:, i].sum(), yr, retain_graph=True)[0] for i in range(d)]  # rows of J
        J = torch.stack(cols, dim=1)  # [B, d, d] with J[b,i,j] = du_i/dz_j
        if self.logdet == "trace":
            return (-(h.squeeze(-1)) * torch.diagonal(J, dim1=1, dim2=2).sum(-1)).detach()
        eye = torch.eye(d, device=y.device, dtype=y.dtype).unsqueeze(0)
        M = eye - h.unsqueeze(-1) * J.detach()
        sign, logabs = torch.linalg.slogdet(M)
        return logabs

    def estimate_at(self, z_a: torch.Tensor, cond: torch.Tensor, net, source, generator: Optional[torch.Generator] = None,
                    eps_ref: Optional[torch.Tensor] = None) -> DensityResult:
        B = z_a.shape[0]
        z = z_a.detach()
        total_logdet = torch.zeros(B, device=z.device, dtype=z.dtype)
        worst = torch.zeros(B, device=z.device, dtype=z.dtype)
        # walk source-side intervals from the target end: grid[0]=0 ... grid[-1]=1; interval j = [g_j, g_{j+1}]
        for j in range(len(self.grid) - 1):
            r = torch.full((B, 1), self.grid[j], device=z.device, dtype=z.dtype)
            t = torch.full((B, 1), self.grid[j + 1], device=z.device, dtype=z.dtype)
            y, res = self._invert_interval(net, z, r, t, cond)
            total_logdet = total_logdet + self._logdet_interval(net, y, r, t, cond)
            worst = torch.maximum(worst, res)
            z = y.detach()
        with torch.no_grad():
            lp_z = source.log_prob(z) - total_logdet
            lp_norm = lp_z - log_abs_det_tanh(z_a)
        resid = worst if eps_ref is None else torch.maximum(worst, (z - eps_ref).norm(dim=-1))
        return DensityResult(
            log_prob_norm=lp_norm, method=self.method, is_exact_for_sampled_policy=False,
            nfe=(len(self.grid) - 1), inverse_residual=resid, finite_mask=torch.isfinite(lp_norm),
            log_prob_latent=lp_z, eps_hat=z,
        )
