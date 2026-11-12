"""Explicit registries; unknown names raise (no eval / dynamic import by string)."""
from __future__ import annotations

from typing import Any, Callable, Dict

from .exceptions import ConfigError


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str):
        def deco(fn):
            if name in self._items:
                raise ConfigError(f"{self.kind} {name!r} registered twice")
            self._items[name] = fn
            return fn

        return deco

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._items:
            raise ConfigError(f"unknown {self.kind} {name!r}; known: {sorted(self._items)}")
        return self._items[name]

    def build(self, name: str, *args, **kwargs):
        return self.get(name)(*args, **kwargs)

    def names(self):
        return sorted(self._items)


TASKS = Registry("task")
CONTROLLERS = Registry("controller")
AGENTS = Registry("agent")
