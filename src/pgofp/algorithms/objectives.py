"""Loss builders (docs/algorithm_contract.md sections 6-7). Returns scalars + diagnostics; never steps optimizers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from ..core.types import DensityResult
from ..density.base import check_finite
from ..math.action_transform import log_abs_det_tanh
from ..math.importance import candidate_weights
from ..math.meanflow_target import meanflow_target
from ..math.reverse_sampling import mixture_reverse_candidates, reverse_candidates
from ..math.time_sampling import sample_time_pairs
from ..utils.seed import RngStreams


@dataclass
class FlowSettings:
    time_delta: float = 0.02
    flow_ratio: float = 0.5
    time_sampling: str = "uniform"
    logit_mean: float = -0.4
    logit_std: float = 1.0


# ---------------------------------------------------------------------------------------------------- critic
def soft_critic_loss(
    critic,
    critic_target,
    actor_target,
    estimator,
    *,
    cond: torch.Tensor,
    cond_next: torch.Tensor,
    source_next,
    a_cmd: torch.Tensor,
    reward: torch.Tensor,
    bootstrap_mask: torch.Tensor,
    gamma: float,
    alpha: float,
    use_entropy: bool,
    rng: RngStreams,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """y = r + gamma * mask * (min Q'(c', a') - alpha * logpi_proxy(a'|c')),  a' ~ target one-step policy.

    Everything on the target side is stop-gradient; a' is drawn from the SAME conditional source as the behavior
    and Actor sides (identical `source_next` construction in the caller).
    """
    with torch.no_grad():
        eps_n = source_next.sample((), rng.torch("target_source"))
        z_n = actor_target.one_step(eps_n, cond_next)
        a_n = torch.tanh(z_n)
        q_t = critic_target.min_q(cond_next, a_n)
        logpi = torch.zeros_like(q_t)
        dres: Optional[DensityResult] = None
        if use_entropy:
            dres = estimator.estimate_at(z_n, cond_next, actor_target, source_next, rng.torch("trace"), eps_ref=eps_n)
            check_finite(dres, "critic target entropy")
            logpi = dres.log_prob_norm
        y = reward + gamma * bootstrap_mask * (q_t - alpha * logpi)
    q1, q2 = critic(cond, a_cmd)
    loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
    m = {
        "critic/loss": float(loss.detach()),
        "critic/q1_mean": float(q1.detach().mean()), "critic/q1_std": float(q1.detach().std()) if q1.numel() > 1 else 0.0,
        "critic/q2_mean": float(q2.detach().mean()),
        "critic/target_mean": float(y.mean()), "critic/target_std": float(y.std()) if y.numel() > 1 else 0.0,
        "critic/td_error": float((q1.detach() - y).abs().mean()),
        "critic/q_gap": float((q1.detach() - q2.detach()).abs().mean()),
        "entropy/target_logpi_mean": float(logpi.mean()),
    }
    if dres is not None:
        m["entropy/nfe"] = float(dres.nfe)
        if dres.inverse_residual is not None:
            m["entropy/inverse_residual"] = float(dres.inverse_residual.mean())
    return loss, m


# ---------------------------------------------------------------------------------------------------- actor
def _expand_candidates(B: int, K: int, z_t, r, t, cond):
    d = z_t.shape[-1]
    zt = z_t.unsqueeze(1).expand(B, K, d).reshape(B * K, d)
    rr = r.unsqueeze(1).expand(B, K, 1).reshape(B * K, 1)
    tt = t.unsqueeze(1).expand(B, K, 1).reshape(B * K, 1)
    cc = cond.unsqueeze(1).expand(B, K, cond.shape[-1]).reshape(B * K, cond.shape[-1])
    return zt, rr, tt, cc


def actor_loss_reverse(
    net,
    critic,
    cond: torch.Tensor,
    source,
    alpha: float,
    *,
    K: int,
    flow: FlowSettings,
    objective: str,
    zt_proposal: str,
    a_replay_latent: Optional[torch.Tensor],
    rng: RngStreams,
    K_local: int = 0,
    sigma_local: float = 0.3,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Main objective: conditional reverse-sampling Q-reweighted MeanFlow (section 6.2/6.3).

    For each (c, t, z_t): K reverse candidates z_a_k from the source-driven kernel, self-normalized weights
    softmax_k( min Q(c, tanh z_a_k)/alpha + log|J_tanh(z_a_k)| ) (per condition only), MeanFlow JVP targets.

    K_local > 0 adds a defensive local proposal around K_local samples of the CURRENT policy and corrects the
    weights with log phi - log g_mix (balance-heuristic MIS), see math.reverse_sampling.mixture_reverse_candidates.
    The target posterior is unchanged; only the estimator variance at large t drops (K_local = 0 is the exact
    spec-main path).
    """
    B, d = cond.shape[0], source.dim
    dev = cond.device
    ts = sample_time_pairs(B, delta=flow.time_delta, flow_ratio=flow.flow_ratio, kind=flow.time_sampling,
                           logit_mean=flow.logit_mean, logit_std=flow.logit_std, generator=rng.torch("time"),
                           device=dev, dtype=cond.dtype)
    gz = rng.torch("candidates")
    with torch.no_grad():
        if zt_proposal == "standard_normal":
            z_t = torch.randn(B, d, generator=gz, device=dev, dtype=cond.dtype)
        else:
            eps0 = source.sample((), gz)
            z_t = (1.0 - ts.t) * a_replay_latent + ts.t * eps0
        if K_local > 0:
            cond_c = cond.unsqueeze(1).expand(B, K_local, cond.shape[-1]).reshape(B * K_local, -1)
            eps_c = source.sample((K_local,), gz).reshape(B * K_local, d)
            centers = net.one_step(eps_c, cond_c).reshape(B, K_local, d)
        else:
            centers = torch.zeros(B, 0, d, device=dev, dtype=cond.dtype)
        eps, z_a, v, log_corr = mixture_reverse_candidates(z_t, ts.t, source, K, centers, sigma_local, gz)
        Kt = z_a.shape[1]
        a = torch.tanh(z_a)
        qmin = critic.min_q(cond, a)  # [B, Kt]
        logJ = log_abs_det_tanh(z_a)
        if objective == "reverse_qrmf":
            logw = qmin / alpha + logJ
        elif objective == "reverse_no_q":
            logw = logJ
        elif objective == "reverse_uniform":
            logw = torch.zeros_like(qmin)
        else:
            raise ValueError(f"unknown reverse objective {objective!r}")
        w, wd = candidate_weights(logw + log_corr)
        K = Kt
        zt_e, r_e, t_e, c_e = _expand_candidates(B, K, z_t, ts.r, ts.t, cond)
        u_target, jn = meanflow_target(lambda z, r, t: net(z, r, t, c_e), zt_e, r_e, t_e, v.reshape(B * K, d))
    u_pred = net(z_t, ts.r, ts.t, cond)  # [B, d]  (the only differentiable branch)
    per = ((u_pred.unsqueeze(1) - u_target.reshape(B, K, d)) ** 2).mean(-1)  # [B, K]
    per_b = (w * per).sum(-1)
    loss = per_b.mean()
    m = _metrics(loss, per_b, ts, wd, u_pred, u_target.reshape(B, K, d), v, jn, qmin)
    return loss, m


def actor_loss_endpoint_is(
    net,
    critic,
    cond: torch.Tensor,
    source,
    alpha: float,
    *,
    K: int,
    flow: FlowSettings,
    rng: RngStreams,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Validation objective (section 6.5): known uniform-action proposal; weights exp(Q/alpha) per condition.

    a_k ~ U((-1,1)^d), eps_k ~ p_src independent, z_t_k = (1-t) atanh(a_k) + t eps_k. With q_z = 2^-d |J_tanh|
    the importance weight exp(Q/alpha + logJ - log q_z) reduces to exp(Q/alpha) (constants cancel in softmax).
    """
    B, d = cond.shape[0], source.dim
    dev, dt = cond.device, cond.dtype
    ts = sample_time_pairs(B, delta=flow.time_delta, flow_ratio=flow.flow_ratio, kind=flow.time_sampling,
                           logit_mean=flow.logit_mean, logit_std=flow.logit_std, generator=rng.torch("time"),
                           device=dev, dtype=dt)
    gz = rng.torch("candidates")
    with torch.no_grad():
        lim = 1.0 - 1e-6
        a = (torch.rand(B, K, d, generator=gz, device=dev, dtype=dt) * 2 - 1).clamp(-lim, lim)
        z_a = torch.atanh(a)
        eps = source.sample((K,), gz)
        tt = ts.t.unsqueeze(1)
        z_t = (1 - tt) * z_a + tt * eps  # [B,K,d]
        v = eps - z_a
        qmin = critic.min_q(cond, a)
        w, wd = candidate_weights(qmin / alpha)
        zt_f = z_t.reshape(B * K, d)
        r_f = ts.r.unsqueeze(1).expand(B, K, 1).reshape(B * K, 1)
        t_f = ts.t.unsqueeze(1).expand(B, K, 1).reshape(B * K, 1)
        c_f = cond.unsqueeze(1).expand(B, K, cond.shape[-1]).reshape(B * K, -1)
        u_target, jn = meanflow_target(lambda z, r, t: net(z, r, t, c_f), zt_f, r_f, t_f, v.reshape(B * K, d))
    u_pred = net(zt_f, r_f, t_f, c_f).reshape(B, K, d)
    per = ((u_pred - u_target.reshape(B, K, d)) ** 2).mean(-1)
    per_b = (w * per).sum(-1)
    loss = per_b.mean()
    m = _metrics(loss, per_b, ts, wd, u_pred.reshape(B * K, d), u_target.reshape(B, K, d), v, jn, qmin)
    return loss, m


def _metrics(loss, per_b, ts, wd, u_pred, u_target, v, jn, qmin):
    diag = ts.is_diag
    zero = 0.0
    return {
        "loss/actor_qrmf": float(loss.detach()),
        "loss/meanflow_diag": float(per_b.detach()[diag].mean()) if bool(diag.any()) else zero,
        "loss/meanflow_offdiag": float(per_b.detach()[~diag].mean()) if bool((~diag).any()) else zero,
        "weights/ess_mean": float(wd["ess"].mean()),
        "weights/ess_p10": float(torch.quantile(wd["ess"], 0.1)),
        "weights/max_weight_mean": float(wd["max_weight"].mean()),
        "weights/finite_ratio": float(wd["finite_ratio"]),
        "weights/logw_std": float(wd["logw_std"].mean()),
        "weights/candidate_q_std": float(qmin.std(-1).mean()),
        "flow/target_velocity_norm": float(u_target.norm(dim=-1).mean()),
        "flow/displacement_norm": float(v.norm(dim=-1).mean()),
        "flow/jvp_norm": float(jn.mean()),
        "flow/pred_velocity_norm": float(u_pred.detach().norm(dim=-1).mean()),
        "flow/t_mean": float(ts.t.mean()),
        "flow/diag_fraction": float(ts.is_diag.float().mean()),
    }
