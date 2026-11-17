"""FedAvg aggregation (McMahan et al. 2017, Algorithm 1): average weighted by local sample count."""
import torch


def fedavg(updates: torch.Tensor, ctx: dict) -> tuple[torch.Tensor, dict]:
    n_k = torch.as_tensor(ctx["n_samples"], dtype=updates.dtype)
    weights = n_k / n_k.sum()
    # Note: n_k is self-reported. A malicious client that claims a huge n_k dominates the
    # average, one reason robust rules ignore n_k altogether.
    return weights @ updates, {"weights": weights.tolist()}
