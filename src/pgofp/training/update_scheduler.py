"""Fractional UTD driven by REAL environment transitions."""
from __future__ import annotations

from typing import Any, Dict


class UpdateScheduler:
    """credit += new_real_transitions * utd_ratio once learning has started; run floor(credit) updates.

    * utd_ratio < 1 (e.g. 0.2) works through the float credit (never range(0.2)).
    * Before `learning_starts` no credit accumulates, so warmup leaves no debt to repay afterwards.
    * Reset slots of a vector env (no action taken) are never counted: callers pass only real transitions.
    """

    def __init__(self, utd_ratio: float, learning_starts: int, actor_update_interval: int = 1):
        if utd_ratio < 0:
            raise ValueError("utd_ratio must be >= 0")
        self.utd = float(utd_ratio)
        self.learning_starts = int(learning_starts)
        self.actor_interval = int(actor_update_interval)
        self.credit = 0.0
        self.gradient_steps = 0

    def n_updates(self, new_transitions: int, env_transitions: int) -> int:
        if env_transitions < self.learning_starts:
            self.credit = 0.0
            return 0
        self.credit += new_transitions * self.utd
        n = int(self.credit)
        self.credit -= n
        return n

    def do_actor(self) -> bool:
        return self.gradient_steps % self.actor_interval == 0

    def tick(self) -> None:
        self.gradient_steps += 1

    def state_dict(self) -> Dict[str, Any]:
        return {"credit": self.credit, "gradient_steps": self.gradient_steps}

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.credit, self.gradient_steps = sd["credit"], sd["gradient_steps"]
