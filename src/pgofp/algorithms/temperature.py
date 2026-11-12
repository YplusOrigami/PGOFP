"""Entropy temperature alpha_ent (never confused with the OFQL Q-guidance coefficient)."""
from __future__ import annotations

import math
from typing import Any, Dict

import torch


class Temperature:
    """mode=fixed: constant alpha (still used as the Q-reweighting temperature).
    mode=auto: alpha = exp(log_alpha), L = log_alpha * stop_grad(H_hat - H_target).

    Descending L: if H_hat < H_target (entropy too low) d L/d log_alpha < 0 so log_alpha INCREASES;
    if entropy is above target log_alpha decreases.  H_hat = mean(-logpi_proxy) from the ONLINE actor.
    """

    def __init__(self, mode: str, alpha: float, target_entropy, action_dim: int, lr: float,
                 alpha_min: float, alpha_max: float, device="cpu"):
        if mode not in ("fixed", "auto"):
            raise ValueError("temperature.mode must be fixed|auto")
        self.mode = mode
        self.alpha_min, self.alpha_max = alpha_min, alpha_max
        self.target_entropy = float(-action_dim) if target_entropy == "auto" else float(target_entropy)
        self.log_alpha = torch.tensor(math.log(alpha), device=device, requires_grad=(mode == "auto"))
        self.opt = torch.optim.Adam([self.log_alpha], lr=lr) if mode == "auto" else None
        self.at_bound_count = 0
        self.updates = 0

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.detach().exp())

    def loss(self, entropy_hat: torch.Tensor) -> torch.Tensor:
        return self.log_alpha * (entropy_hat.detach() - self.target_entropy)

    def step(self, entropy_hat: torch.Tensor) -> Dict[str, float]:
        if self.mode != "auto":
            return {"alpha": self.alpha}
        self.opt.zero_grad(set_to_none=True)
        loss = self.loss(entropy_hat)
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            lo, hi = math.log(self.alpha_min), math.log(self.alpha_max)
            at_bound = bool((self.log_alpha.detach() <= lo) | (self.log_alpha.detach() >= hi))
            self.log_alpha.clamp_(lo, hi)
        self.at_bound_count += int(at_bound)
        self.updates += 1
        return {"alpha": self.alpha, "alpha_loss": float(loss.detach()), "alpha_at_bound_frac": self.at_bound_count / self.updates}

    def state_dict(self) -> Dict[str, Any]:
        return {"log_alpha": self.log_alpha.detach().cpu(), "opt": None if self.opt is None else self.opt.state_dict(),
                "at_bound_count": self.at_bound_count, "updates": self.updates}

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        with torch.no_grad():
            self.log_alpha.copy_(sd["log_alpha"].to(self.log_alpha.device))
        if self.opt is not None and sd.get("opt") is not None:
            self.opt.load_state_dict(sd["opt"])
        self.at_bound_count, self.updates = sd["at_bound_count"], sd["updates"]
