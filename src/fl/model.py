"""MLP for tabular flows. No BatchNorm: its running statistics are not averaged correctly by FedAvg."""
import torch
from torch import nn


def make_mlp(d_in: int, n_classes: int, hidden: tuple[int, ...] = (128, 64)) -> nn.Module:
    layers, prev = [], d_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, n_classes))
    return nn.Sequential(*layers)


def get_flat(model: nn.Module) -> torch.Tensor:
    """Model parameters as one detached vector (a copy, never a view of the live weights)."""
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()


def set_flat(model: nn.Module, vec: torch.Tensor) -> None:
    torch.nn.utils.vector_to_parameters(vec, model.parameters())
