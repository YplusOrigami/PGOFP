"""Named-field observation assembly with unit checks and explicit missing-value masks."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from ...core.exceptions import ConfigError
from .protocol import ObservationSchema


class ObservationAssembler:
    """Builds the flat observation vector from named fields in schema order (no hard-coded indices)."""

    def __init__(self, schema: ObservationSchema, expected_units: Dict[str, str] | None = None):
        self.schema = schema
        self.names = [f.name for f in schema.fields]
        self.counts = [f.count for f in schema.fields]
        self.dim = int(sum(self.counts))
        if self.dim != schema.obs_dim:
            raise ConfigError(f"schema field counts sum to {self.dim} but obs_dim={schema.obs_dim}")
        if expected_units:
            for f in schema.fields:
                if f.name in expected_units and expected_units[f.name] != f.unit:
                    raise ConfigError(f"unit mismatch for {f.name}: schema {f.unit!r} vs expected {expected_units[f.name]!r}")

    def assemble(self, values: Dict[str, np.ndarray], masks: Dict[str, np.ndarray] | None = None):
        """Returns (vector [d], mask [d]). Missing fields are masked, never silently turned into 'normal zeros'."""
        vec, mask = [], []
        for name, cnt in zip(self.names, self.counts):
            if name in values:
                v = np.asarray(values[name], np.float32).reshape(-1)
                if v.size != cnt:
                    raise ConfigError(f"field {name}: expected {cnt} values, got {v.size}")
                m = np.ones(cnt, np.float32) if masks is None or name not in masks else np.asarray(masks[name], np.float32).reshape(-1)
                vec.append(np.where(m > 0, v, 0.0))
                mask.append(m)
            else:
                vec.append(np.zeros(cnt, np.float32))
                mask.append(np.zeros(cnt, np.float32))
        return np.concatenate(vec), np.concatenate(mask)
