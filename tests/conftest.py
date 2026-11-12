from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

torch.set_default_dtype(torch.float32)


@pytest.fixture
def gen():
    g = torch.Generator()
    g.manual_seed(1234)
    return g


class AffineField(nn.Module):
    """u(z, r, t, c) = (a + b*t) * z  (a, b scalars or per-dim): analytic flow/density references."""

    def __init__(self, a, b=0.0, dim=2):
        super().__init__()
        self.a = nn.Parameter(torch.as_tensor(a, dtype=torch.float64).expand(dim).clone(), requires_grad=False)
        self.b = nn.Parameter(torch.as_tensor(b, dtype=torch.float64).expand(dim).clone(), requires_grad=False)

    def forward(self, z, r, t, cond):
        return (self.a.to(z.dtype) + self.b.to(z.dtype) * t) * z


@pytest.fixture
def affine_field():
    return AffineField


def make_ctx(B=4, d=2, M=1, obs=3, seed=0, valid=1.0, dtype=torch.float32):
    from pgofp.core.types import ContextBatch

    g = torch.Generator().manual_seed(seed)
    return ContextBatch(
        obs_raw=torch.randn(B, obs, generator=g, dtype=dtype),
        reference=torch.zeros(B, 0, dtype=dtype),
        ctrl_feat=torch.zeros(B, 0, dtype=dtype),
        phase=torch.zeros(B, 0, dtype=dtype),
        prior_mu=torch.randn(B, M, d, generator=g, dtype=dtype),
        prior_logstd=torch.full((B, M, d), math.log(0.3), dtype=dtype),
        prior_valid=torch.full((B, M), valid, dtype=dtype),
    )


@pytest.fixture
def ctx_factory():
    return make_ctx
