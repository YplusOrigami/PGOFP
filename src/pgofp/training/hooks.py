from __future__ import annotations

from typing import Any, Dict


class IntervalTrigger:
    """Fires once each time the env-step clock crosses a multiple of `interval`; state survives checkpoints."""

    def __init__(self, interval: int):
        self.interval = int(interval)
        self.next = self.interval

    def due(self, step: int) -> bool:
        return self.interval > 0 and step >= self.next

    def fire(self, step: int) -> None:
        while self.next <= step:
            self.next += self.interval

    def state_dict(self) -> Dict[str, Any]:
        return {"next": self.next}

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.next = sd["next"]
