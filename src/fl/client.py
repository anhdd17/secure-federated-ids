"""Benign client: local SGD from the global weights, returns the update Δ = w_local − w_global."""
from __future__ import annotations

import torch
from torch import nn

from src.fl.model import get_flat, set_flat


def local_update(
    model: nn.Module,
    w_global: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    lr: float,
    batch_size: int,
    gen: torch.Generator,
) -> torch.Tensor:
    """Run E epochs of minibatch SGD on (X, y) and return Δ as a fresh vector.

    One `model` object is reused by all clients to save memory, which is safe only because
    the weights are overwritten from w_global and a new optimizer is built on every call:
    no parameter or momentum state survives from the previous client.
    """
    set_flat(model, w_global)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    n = len(y)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, batch_size):
            b = perm[start : start + batch_size]
            opt.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(model(X[b]), y[b]).backward()
            opt.step()
    return get_flat(model) - w_global
